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

#include "ripples/gpu/find_most_influential.h"

#include <algorithm>
#include <vector>

#include "ripples/gpu/gpu_runtime_trait.h"

#include "thrust/count.h"
#include "thrust/device_ptr.h"
#include "thrust/extrema.h"

namespace ripples {

__global__ void kernel_count(size_t batch_size, size_t num_nodes,
                             uint32_t *d_counters, uint32_t *d_rrr_sets) {
  int pos = threadIdx.x + blockDim.x * blockIdx.x;
  if (pos < batch_size) {
    uint32_t v = d_rrr_sets[pos];
    atomicAdd(d_counters + v, 1);
  }
}

std::pair<uint32_t, size_t> GPUMaxElement(uint32_t *b, size_t N) {
  thrust::device_ptr<uint32_t> dev_ptr(b);

  thrust::device_ptr<uint32_t> min_ptr =
      thrust::max_element(thrust::device, dev_ptr, dev_ptr + N);
  uint32_t v = thrust::distance(dev_ptr, min_ptr);
  return std::make_pair(v, size_t(dev_ptr[v]));
}

__global__ void count_uncovered_kernel(size_t batch_size, size_t num_nodes,
                                       uint32_t *d_rrr_index,
                                       uint32_t *d_rrr_sets, uint32_t *d_mask,
                                       uint32_t *d_counters) {
  int pos = threadIdx.x + blockDim.x * blockIdx.x;

  if (pos < batch_size) {
    uint32_t set = d_rrr_index[pos];
    if (d_mask[set] != 1) {
      uint32_t v = d_rrr_sets[pos];
      atomicAdd(d_counters + v, 1);
    }
  }
}

__global__ void update_mask_kernel(size_t batch_size, uint32_t *d_rrr_index,
                                   uint32_t *d_rrr_sets, uint32_t *d_mask,
                                   uint32_t last_seed) {
  size_t pos = threadIdx.x + blockDim.x * blockIdx.x;

  if (pos < batch_size && d_rrr_sets[pos] == last_seed) {
    uint32_t set = d_rrr_index[pos];
    d_mask[set] = 1;
  }
}

size_t CountZeros(char *d_rr_mask, size_t N) {
  thrust::device_ptr<char> dev_ptr = thrust::device_pointer_cast(d_rr_mask);
  char zero = 0;
  return thrust::count(dev_ptr, dev_ptr + N, zero);
}

size_t CountOnes(char *d_rr_mask, size_t N) {
  thrust::device_ptr<char> dev_ptr = thrust::device_pointer_cast(d_rr_mask);
  char one = 1;
  return thrust::count(dev_ptr, dev_ptr + N, one);
}

__global__ void sum_vectors(uint32_t *src, uint32_t *dst, size_t N) {
  size_t pos = threadIdx.x + blockDim.x * blockIdx.x;
  if (pos < N) {
    if (src[pos]) {
      atomicAdd(dst + pos, src[pos]);
    }
  }
}

}  // namespace ripples
