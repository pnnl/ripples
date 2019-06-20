//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_UTILS_H
#define IM_CUDA_CUDA_UTILS_H

#include "spdlog/spdlog.h"

#include "ripples/graph.h"

namespace ripples {

//
// debug utilities
//
void cuda_check(cudaError_t err, const char *fname, int line) {
  if (err != cudaSuccess) {
    spdlog::error("> CUDA error @%s:%d: name=%s msg='%s'\n", fname, line,
            cudaGetErrorName(err), cudaGetErrorString(err));
  }
}

void cuda_check(const char *fname, int line) {
  cuda_check(cudaGetLastError(), fname, line);
}
}  // namespace ripples

#endif  // IM_CUDA_CUDA_UTILS_H
