#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "layers.h"
#include "volume.h"
//32, 32, 3, 5, 16, 1, 2
conv_layer_t* make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
                              int stride, int pad) {
  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  l->output_depth = num_filters;
  l->filter_width = filter_width;
  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->filter_height = filter_width;
  l->stride        = stride;
  l->pad = pad;

  l->output_width = (input_width + pad * 2 - filter_width) /
                    stride + 1;
  l->output_height = (input_height + pad * 2 - filter_width) /
                     stride + 1;

  l->filters = malloc(sizeof(volume_t*) * num_filters);
  //#pragma omp for
  for (int i = 0; i < num_filters; i++) {
    l->filters[i] = make_volume(filter_width, filter_width,
                                input_depth, 0.0);
  }

  l->bias   = 0.0;
  l->biases = make_volume(1, 1, num_filters, 0.0);

  return l;
}

// Performs the forward pass for a convolutional layer by convolving each one
// of the filters with a particular input, and placing the result in the output
// array.
//
// One way to think about convolution in this case is that we have one of the
// layer's filters (a 3D array) that is superimposed on one of the layer's
// inputs (a second 3D array) that has been implicitly padded with zeros. Since
// convolution is a sum of products (described below), we don't actually have
// to add any zeros to the input volume since those terms will not contribute
// to the convolution. Instead, for each position in the filter, we just make
// sure that we are in bounds for the input volume.
//
// Essentially, the filter is "sliding" across the input, in both the x and y
// directions, where we increment our position in each direction by using the
// stride parameter.
//
// At each position, we compute the sum of the elementwise product of the filter
// and the part of the array it's covering. For instance, let's consider a 2D
// case, where the filter (on the left) is superimposed on some part of the
// input (on the right).
//
//   Filter             Input
//  -1  0  1           1  2  3
//  -1  0  1           4  5  6
//  -1  0  1           7  8  9
//
// Here, the sum of the elementwise product is:
//    Filter[0][0] * Input[0][0] + Filter[0][1] * Input[0][1] + ...
//    = -1 * 1 + 0 * 2 + ... + 0 * 8 + 1 * 9
//    = 6
//
// The 3D case is essentially the same, we just have to sum over the other
// dimension as well. Also, since volumes are internally represented as 1D
// arrays, we must use the volume_get and volume_set commands to access elements
// at a coordinate (x, y, d). Finally, we add the corresponding bias for the
// filter to the sum before putting it into the output volume.
void conv_forward_first(conv_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  //changed y+= stride and x+=stride to y++ and x++. changed xy form -l->pad to -2
  int xy = -l->pad;
  volume_t** filters   = l->filters;
  double* bias_weights = l->biases->weights;

  int output_y = l->output_height;
  int output_x = l->output_width;
  int output_z = l->output_depth;
  //output_x, output_y = 32, output_z = 16
  volume_t* in  = inputs[0];
  volume_t* out = outputs[0];
  int out_width = out->width;
  int out_depth = out->depth;

  int in_height  = in->height;
  int in_width   = in->width;
  int in_depth   = in->depth;
  double* in_w   = in->weights;
  double* out_w  = out->weights;

  #pragma omp parallel for
  for (int f = 0; f < output_z; f++) {
  volume_t* filter = filters[f];
  int filter_x     = filter->width;
  int filter_y     = filter->height;
  int filter_z     = filter->depth;
  double* filter_w = filter->weights;


  int y = xy;
  for (int out_y = 0; out_y < output_y; y++, out_y++) {
    int x = xy;
    for (int out_x = 0; out_x < output_x; x++, out_x++) {

      // Take sum of element-wise product
      double sum = 0.0;
      for (int fx = 0; fx < filter_x; fx++) {
        int in_x = x + fx;
        if (in_x >= 0 && in_x < in_width) {
        for (int fy = 0; fy < filter_y; fy++) {
          int in_y = y + fy;
          if (in_y >= 0 && in_y < in_height) {
            //Filter is 3, so don't need a loop
            sum += filter_w[((filter_x * fy) + fx) * filter_z + 0] * in_w[((in_width * in_y) + in_x) * in_depth + 0] +
                   filter_w[((filter_x * fy) + fx) * filter_z + 1] * in_w[((in_width * in_y) + in_x) * in_depth + 1] +
                   filter_w[((filter_x * fy) + fx) * filter_z + 2] * in_w[((in_width * in_y) + in_x) * in_depth + 2];
            }
          }
          }
        }
        sum += bias_weights[f];
        out_w[((out_width * out_y) + out_x) * out_depth + f] = sum;
      }
    }
  }
}

void conv_forward_second(conv_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  int xy = -l->pad;
  volume_t** filters = l->filters;
  double* bias_weights = l->biases->weights;
  int output_y = l->output_height;
  int output_x = l->output_width;
  int output_z = l->output_depth;
  //output_x, output_y = 16, output_z = 20

  volume_t* in  = inputs[0];
  volume_t* out = outputs[0];
  int out_width = out->width;
  int out_depth = out->depth;

  int in_height  = in->height;
  int in_width   = in->width;
  int in_depth   = in->depth;
  double* in_w   = in->weights;
  double* out_w  = out->weights;

    #pragma omp parallel for
    for (int f = 0; f < output_z; f++) {
      volume_t* filter = filters[f];
      int filter_x     = filter->width;
      int filter_y     = filter->height;
      int filter_z     = filter->depth;
      double* filter_w = filter->weights;


      int y = xy;
      for (int out_y = 0; out_y < output_y; y++, out_y++) {
        int x = xy;
        for (int out_x = 0; out_x < output_x; x++, out_x++) {

          // Take sum of element-wise product
          double sum = 0.0;
          for (int fx = 0; fx < filter_x; fx++) {
            int in_x = x + fx;
            if (in_x >= 0 && in_x < in_width) {
            for (int fy = 0; fy < filter_y; fy++) {
              int in_y = y + fy;
              if (in_y >= 0 && in_y < in_height) {
                //Filter is 16, so don't need a tail case
                __m256d sum_vec  = _mm256_setzero_pd();
                double* fil_vol  = &(filter_w[((filter_x * fy) + fx) * filter_z]);
                double* in_vol   = &(in_w[((in_width * in_y) + in_x) * in_depth]);

                __m256d curr_vec1 = _mm256_loadu_pd((fil_vol));
                __m256d curr_vec2 = _mm256_loadu_pd((in_vol));
                __m256d prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
                sum_vec           = _mm256_add_pd(sum_vec, prod_vec);

                curr_vec1 = _mm256_loadu_pd((fil_vol + 4));
                curr_vec2 = _mm256_loadu_pd((in_vol  + 4));
                prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
                sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

                curr_vec1 = _mm256_loadu_pd((fil_vol + 8));
                curr_vec2 = _mm256_loadu_pd((in_vol  + 8));
                prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
                sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

                curr_vec1 = _mm256_loadu_pd((fil_vol + 12));
                curr_vec2 = _mm256_loadu_pd((in_vol  + 12));
                prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
                sum_vec   = _mm256_add_pd(sum_vec, prod_vec);
                double adder[4];
                _mm256_storeu_pd((double *) adder, sum_vec);
                sum += adder[0] + adder[1] + adder[2] + adder[3];
              }
            }
            }
          }

          sum += bias_weights[f];
          out_w[((out_width * out_y) + out_x) * out_depth + f] = sum;
        }
      }
    }
}

void conv_forward_third(conv_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  int xy = -l->pad;
  volume_t** filters = l->filters;
  double* bias_weights = l->biases->weights;
  int output_y = l->output_height;
  int output_x = l->output_width;
  int output_z = l->output_depth;
  //output_x, output_y = 8, output_z = 20

  volume_t* in  = inputs[0];
  volume_t* out = outputs[0];
  int out_width = out->width;
  int out_depth = out->depth;

  int in_height  = in->height;
  int in_width   = in->width;
  int in_depth   = in->depth;
  double* in_w   = in->weights;
  double* out_w  = out->weights;

  #pragma omp parallel for
  for (int f = 0; f < output_z; f++) {
    volume_t* filter = filters[f];
    int filter_x     = filter->width;
    int filter_y     = filter->height;
    int filter_z     = filter->depth;
    double* filter_w = filter->weights;


    int y = xy;
    for (int out_y = 0; out_y < output_y; y++, out_y++) {
      int x = xy;
      for (int out_x = 0; out_x < output_x; x++, out_x++) {

        // Take sum of element-wise product
        double sum = 0.0;
        for (int fx = 0; fx < filter_x; fx++) {
          int in_x = x + fx;
          if (in_x >= 0 && in_x < in_width) {
          for (int fy = 0; fy < filter_y; fy++) {
            int in_y = y + fy;
            if (in_y >= 0 && in_y < in_height) {
              //Filter is 20, so don't need a tail case
              __m256d sum_vec  = _mm256_setzero_pd();
              double* fil_vol  = &(filter_w[((filter_x * fy) + fx) * filter_z]);
              double* in_vol   = &(in_w[((in_width * in_y) + in_x) * in_depth]);

              __m256d curr_vec1 = _mm256_loadu_pd((fil_vol));
              __m256d curr_vec2 = _mm256_loadu_pd((in_vol));
              __m256d prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
              sum_vec           = _mm256_add_pd(sum_vec, prod_vec);

              curr_vec1 = _mm256_loadu_pd((fil_vol + 4));
              curr_vec2 = _mm256_loadu_pd((in_vol  + 4));
              prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
              sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

              curr_vec1 = _mm256_loadu_pd((fil_vol + 8));
              curr_vec2 = _mm256_loadu_pd((in_vol  + 8));
              prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
              sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

              curr_vec1 = _mm256_loadu_pd((fil_vol + 12));
              curr_vec2 = _mm256_loadu_pd((in_vol  + 12));
              prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
              sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

              curr_vec1 = _mm256_loadu_pd((fil_vol + 16));
              curr_vec2 = _mm256_loadu_pd((in_vol  + 16));
              prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
              sum_vec   = _mm256_add_pd(sum_vec, prod_vec);
              double adder[4];
              _mm256_storeu_pd((double *) adder, sum_vec);
              sum += adder[0] + adder[1] + adder[2] + adder[3];
            }
          }
          }
        }

        sum += bias_weights[f];
        out_w[((out_width * out_y) + out_x) * out_depth + f] = sum;
        }
      }
    }
}

void conv_load(conv_layer_t* l, const char* file_name) {
  int filter_width;
  int filter_height;
  int depth;
  int filters;

  FILE* fin = fopen(file_name, "r");

  fscanf(fin, "%d %d %d %d", &filter_width, &filter_height, &depth, &filters);
  assert(filter_width == l->filter_width);
  assert(filter_height == l->filter_height);
  assert(depth == l->input_depth);
  assert(filters == l->output_depth);
  volume_t** filtery = l->filters;
  int output_depth   = l->output_depth;

  #pragma omp for collapse(4)
  for (int f = 0; f < filters; f++) {
    for (int x = 0; x < filter_width; x++) {
      for (int y = 0; y < filter_height; y++) {
        for (int d = 0; d < depth; d++) {
          double val;
          fscanf(fin, "%lf", &val);
          filtery[f]->weights[((filtery[f]->width * y) + x) * filtery[f]->depth + d] = val;
        }
      }
    }
  }

  double* bias_weights = l->biases->weights;
  #pragma omp for
  for (int d = 0; d < output_depth; d++) {
    double val;
    fscanf(fin, "%lf", &val);
    bias_weights[d] = val;
  }

  fclose(fin);
}

relu_layer_t* make_relu_layer(int input_width, int input_height, int input_depth) {
  relu_layer_t* l = (relu_layer_t*)malloc(sizeof(relu_layer_t));

  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->output_width  = input_width;
  l->output_height = input_height;
  l->output_depth  = input_depth;

  return l;
}

// Applies the Rectifier Linear Unit (ReLU) function to the input, which sets
// output(x, y, d) to max(0.0, input(x, y, d)).
void relu_forward(relu_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  volume_t* curr_input  = inputs[0];
  volume_t* curr_out    = outputs[0];

  double* curr_weights  = curr_input->weights;
  int            width  = curr_input->width;
  int            depth  = curr_input->depth;
  double* curr_weights1  = curr_out->weights;
  int            width1  = curr_out->width;
  int            depth1  = curr_out->depth;
  #pragma omp parallel for
  for (int x = 0; x < l->input_width; x++) {
    for (int y = 0; y < l->input_height; y++) {
      for (int d = 0; d < l->input_depth; d+=4) {
        double volume = curr_weights[((width * y) + x) * depth + d];
        double value = (volume < 0.0) ? 0.0 : volume;
        curr_weights1[((width1 * y) + x) * depth1 + d] = value;

        volume = curr_weights[((width * y) + x) * depth + (d + 1)];
        value = (volume < 0.0) ? 0.0 : volume;
        curr_weights1[((width1 * y) + x) * depth1 + (d + 1)] = value;

        volume = curr_weights[((width * y) + x) * depth + (d + 2)];
        value = (volume < 0.0) ? 0.0 : volume;
        curr_weights1[((width1 * y) + x) * depth1 + (d + 2)] = value;

        volume = curr_weights[((width * y) + x) * depth + (d + 3)];
        value = (volume < 0.0) ? 0.0 : volume;
        curr_weights1[((width1 * y) + x) * depth1 + (d + 3)] = value;
      }
    }
  }
}

pool_layer_t* make_pool_layer(int input_width, int input_height, int input_depth, int pool_width, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));

  l->pool_width   = pool_width;
  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->pool_height = pool_width;
  l->stride      = stride;
  l->pad         = 0;

  l->output_depth  = input_depth;
  l->output_width  = floor((input_width + 0 * 2 - pool_width) / stride + 1);
  l->output_height = floor((input_height + 0 * 2 - pool_width) / stride + 1);

  return l;
}

// This is like the convolutional layer in that we are sliding across the input
// volume, but instead of having a filter that we use to find the sum of an
// elementwise product, we instead just output the max value of some part of
// the image. For instance, if we consider a 2D case where the following is the
// part of the input that we are considering:
//
//     1 3 5
//     4 2 1
//     2 2 2
//
// then the value of the corresponding element in the output is 5 (since that
// is the maximum element). This effectively compresses the input.
void pool_forward(pool_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  int xy = 0;
  //2 used to be pool_width
  //xy used to be -l->pad
  int output_x   = l->output_width;
  int output_y   = l->output_height;
  int output_z   = l->output_depth;
  volume_t* in   = inputs[0];
  volume_t* out  = outputs[0];
  int in_width   = in->width;
  int in_depth   = in->depth;
  int in_height  = in->height;
  int out_width = out->width;
  int out_depth = out->depth;
  double* in_weights  = in->weights;
  double* out_weights = out->weights;
  #pragma omp parallel for
  for (int d = 0; d < output_z; d++) {
    int x = xy;
    for (int out_x = 0; out_x < output_x; x += 2, out_x++) {
      int y = xy;
      for (int out_y = 0; out_y < output_y; y += 2, out_y++) {

        double max = -INFINITY;
        for (int fx = 0; fx < 2; fx++) {
          int in_x = x + fx;
          if (in_x >= 0 && in_x < in_width) {
          for (int fy = 0; fy < 2; fy++) {
            int in_y = y + fy;
            if (in_y >= 0 && in_y < in_height) {
              double v = in_weights[((in_width * in_y) + in_x) * in_depth + d];
              if (v > max) {
                max = v;
              }
            }
          }
          }
        }
        out_weights[((out_width * out_y) + out_x) * out_depth + d] = max;
      }
    }
  }
}

fc_layer_t* make_fc_layer(int input_width, int input_height, int input_depth, int num_neurons) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  l->output_depth = num_neurons;
  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->num_inputs    = input_width * input_height * input_depth;
  l->output_width  = 1;
  l->output_height = 1;

  l->filters = (volume_t**)malloc(sizeof(volume_t*) * num_neurons);
  for (int i = 0; i < num_neurons; i++) {
    l->filters[i] = make_volume(1, 1, l->num_inputs, 0.0);
  }

  l->bias   = 0.0;
  l->biases = make_volume(1, 1, num_neurons, 0.0);

  return l;
}

// Computes the dot product (i.e. the sum of the elementwise product) of the
// input's weights with each of the filters. Note that these filters are not
// the same as the filters for the convolutional layer.
void fc_forward(fc_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  volume_t** filters = l->filters;
  volume_t* in        = inputs[0];
  volume_t* out       = outputs[0];
  double* in_weights  = in->weights;
  double* out_weights = out->weights;
  #pragma omp parallel for
  for (int i = 0; i < 10; i++) {
    double* filter_weights = filters[i]->weights;
    __m256d sum_vec = _mm256_setzero_pd();
    for (int d = 0; d < 320; d += 16) {
      __m256d curr_vec1 = _mm256_loadu_pd((filter_weights + d));
      __m256d curr_vec2 = _mm256_loadu_pd((in_weights + d));
      __m256d prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
      sum_vec           = _mm256_add_pd(sum_vec, prod_vec);

      curr_vec1 = _mm256_loadu_pd((filter_weights + d + 4));
      curr_vec2 = _mm256_loadu_pd((in_weights + d + 4));
      prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
      sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

      curr_vec1 = _mm256_loadu_pd((filter_weights + d + 8));
      curr_vec2 = _mm256_loadu_pd((in_weights + d + 8));
      prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
      sum_vec   = _mm256_add_pd(sum_vec, prod_vec);

      curr_vec1 = _mm256_loadu_pd((filter_weights + d + 12));
      curr_vec2 = _mm256_loadu_pd((in_weights + d + 12));
      prod_vec  = _mm256_mul_pd(curr_vec1, curr_vec2);
      sum_vec   = _mm256_add_pd(sum_vec, prod_vec);
    }
    double adder[4];
    _mm256_storeu_pd(adder, sum_vec);
    double dot = adder[0] + adder[1] + adder[2] + adder[3] + l->biases->weights[i];
    out_weights[i] = dot;
  }
}

void fc_load(fc_layer_t* l, const char* filename) {
  FILE* fin = fopen(filename, "r");

  int num_inputs;
  int output_depth;
  fscanf(fin, "%d %d", &num_inputs, &output_depth);
  assert(output_depth == 10);
  assert(num_inputs == 320);
  volume_t** filters = l->filters;
    #pragma omp for collapse(2)
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 320; j+=8) {
        double* filter_weights = filters[i]->weights;
        fscanf(fin, "%lf", &(filter_weights[j]));
        fscanf(fin, "%lf", &(filter_weights[j+1]));
        fscanf(fin, "%lf", &(filter_weights[j+2]));
        fscanf(fin, "%lf", &(filter_weights[j+3]));
        fscanf(fin, "%lf", &(filter_weights[j+4]));
        fscanf(fin, "%lf", &(filter_weights[j+5]));
        fscanf(fin, "%lf", &(filter_weights[j+6]));
        fscanf(fin, "%lf", &(filter_weights[j+7]));
      }
    }
  double* bias_weights = l->biases->weights;

  #pragma omp for
  for (int i = 0; i < 10; i++) {
  fscanf(fin, "%lf", &(bias_weights[i]));
  }

  fclose(fin);
}

softmax_layer_t* make_softmax_layer(int input_width, int input_height, int input_depth) {
  softmax_layer_t* l = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));

  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->output_width  = 1;
  l->output_height = 1;
  l->output_depth  = input_width * input_height * input_depth;

  l->likelihoods = (double*)malloc(sizeof(double) * l->output_depth);

  return l;
}

// This function converts an input's weights array into a probability
// distribution by using the following formula:
//
// likelihood[i] = exp(in->weights[i]) / sum(exp(in->weights))
//
// To increase the numerical stability of taking the exponential of a value, we
// subtract the maximum input weights from each weight before taking the
// exponential. This yields exactly the same results as the expression above,
// but is more resilient to floating point errors.
void softmax_forward(softmax_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  //all values of 10 were l->output_depth
  double likelihoods[10];
  volume_t* in  = inputs[0];
  volume_t* out = outputs[0];

  double* in_w  = in->weights;
  double* out_w = out->weights;

  double amax = in_w[0];
  // Compute max activation (used to compute exponentials)
  #pragma omp parallel for reduction(max: amax)
  for (int i = 1; i < 10; i++) {
    if (in_w[i] > amax) {
      amax = in_w[i];
    }
  }

  // Compute exponentials in a numerically stable way
  likelihoods[0] = exp(in_w[0] - amax);
  likelihoods[1] = exp(in_w[1] - amax);
  likelihoods[2] = exp(in_w[2] - amax);
  likelihoods[3] = exp(in_w[3] - amax);
  likelihoods[4] = exp(in_w[4] - amax);
  likelihoods[5] = exp(in_w[5] - amax);
  likelihoods[6] = exp(in_w[6] - amax);
  likelihoods[7] = exp(in_w[7] - amax);
  likelihoods[8] = exp(in_w[8] - amax);
  likelihoods[9] = exp(in_w[9] - amax);
  double total = likelihoods[0] + likelihoods[1] + likelihoods[2] + likelihoods[3] + likelihoods[4] +
                 likelihoods[5] + likelihoods[6] + likelihoods[7] + likelihoods[8] + likelihoods[9];
   // Normalize and output to sum to one
  __m256d first_div  = _mm256_setzero_pd();
  __m256d second_div = _mm256_setzero_pd();
  __m256d total_vec  = _mm256_set1_pd(total);
  __m256d curr_vec   = _mm256_loadu_pd(likelihoods);
  __m256d curr_vec2  = _mm256_loadu_pd(likelihoods + 4);

  first_div  = _mm256_div_pd(curr_vec, total_vec);
  second_div = _mm256_div_pd(curr_vec2, total_vec);
  _mm256_storeu_pd(out_w, first_div);
  _mm256_storeu_pd(out_w + 4, second_div);
  out_w[8] = likelihoods[8] / total;
  out_w[9] = likelihoods[9] / total;
}
