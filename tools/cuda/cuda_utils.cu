//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#include "ripples/cuda/cuda_utils.h"

namespace ripples {

  //
  // debug utilities
  //
  void cuda_check(cudaError_t err, const char *fname, int line) {
    if (err != cudaSuccess) {
      fprintf(stderr, "> CUDA error @%s:%d: name=%s msg='%s'\n", fname, line,
                    cudaGetErrorName(err), cudaGetErrorString(err));
      fflush(stderr);
    }
  }
  
  void cuda_check(const char *fname, int line) {
    cuda_check(cudaGetLastError(), fname, line);
  }
  }  // namespace ripples