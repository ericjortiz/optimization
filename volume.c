#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "volume.h"

inline double volume_get(volume_t* v, int x, int y, int d) {
  return v->weights[((v->width * y) + x) * v->depth + d];
}

inline void volume_set(volume_t* v, int x, int y, int d, double value) {
  v->weights[((v->width * y) + x) * v->depth + d] = value;
}

volume_t* make_volume(int width, int height, int depth, double value) {
  volume_t* new_vol = malloc(sizeof(struct volume));
  new_vol->weights = malloc(sizeof(double) * width * height * depth);

  new_vol->width  = width;
  new_vol->height = height;
  new_vol->depth  = depth;

  double* vol_weights = new_vol -> weights;
  #pragma omp for collapse(3)
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      for (int d = 0; d < depth / 4 * 4; d++) {
        vol_weights[((width * y) + x) * depth + d] = value;
      }
    }
  }

  return new_vol;
}

void copy_volume(volume_t* dest, volume_t* src) {
  assert(dest->width == src->width);
  assert(dest->height == src->height);
  assert(dest->depth == src->depth);

  int width     = dest->width;
  int height    = dest->height;
  int depth     = dest->depth;
  int src_depth = src->depth;
  int src_width = src->width;
  double* dst_weights = dest->weights;
  double* src_weights = src->weights;

  //#pragma omp for collapse(3)
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      for (int d = 0; d < depth; d++) {
        dst_weights[((width * y) + x) * depth + d] = src_weights[((src_width * y) + x) * src_depth + d];
      }
    }
  }
}

void free_volume(volume_t* v) {
  free(v->weights);
  free(v);
}
