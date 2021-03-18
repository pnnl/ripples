//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
// 
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
// 
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_CUDA_CUDA_UTILS_H
#define RIPPLES_CUDA_CUDA_UTILS_H

#include <vector>
#include <utility>

#include "cuda_runtime.h"

namespace ripples {
void cuda_check(cudaError_t err, const char *fname, int line);
void cuda_check(const char *fname, int line);


//! \brief CUDA runtime wrap functions.
size_t cuda_max_blocks();
size_t cuda_num_devices();
void cuda_set_device(size_t);
void cuda_stream_create(cudaStream_t *);
void cuda_stream_destroy(cudaStream_t);

std::vector<std::pair<size_t, ssize_t>> cuda_get_reduction_tree();

bool cuda_malloc(void **dst, size_t size);
void cuda_free(void *ptr);
void cuda_d2h(void *dst, void *src, size_t size, cudaStream_t);
void cuda_d2h(void *dst, void *src, size_t size);
void cuda_h2d(void *dst, void *src, size_t size, cudaStream_t);
void cuda_h2d(void *dst, void *src, size_t size);
void cuda_memset(void *dst, int val, size_t size, cudaStream_t s);
void cuda_memset(void *dst, int val, size_t size);
void cuda_sync(cudaStream_t);

void cuda_enable_p2p(size_t dev_number);
void cuda_disable_p2p(size_t dev_number);

size_t cuda_available_memory();

}  // namespace ripples

#endif  // IM_CUDA_CUDA_UTILS_H
