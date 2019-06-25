//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_CUDA_CUDA_UTILS_H
#define RIPPLES_CUDA_CUDA_UTILS_H

namespace ripples {
void cuda_check(cudaError_t err, const char *fname, int line);
void cuda_check(const char *fname, int line);
}  // namespace ripples

#endif  // IM_CUDA_CUDA_UTILS_H
