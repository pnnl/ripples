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

#include "ripples/cuda/find_most_influential.h"
#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_utils.h"

#include <algorithm>
#include <vector>

#include "thrust/extrema.h"
#include "thrust/device_ptr.h"


namespace ripples {

__global__ void kernel_count(size_t batch_size, size_t num_nodes,
                             uint32_t *d_counters, uint32_t *d_rrr_sets) {
  int pos = threadIdx.x + blockDim.x * blockIdx.x;
  if (pos < batch_size) {
    uint32_t v = d_rrr_sets[pos];
    atomicAdd(d_counters + v, 1);
  }
}


void cuda_count_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                       size_t num_nodes,
                       uint32_t *d_counters, uint32_t *d_rrr_sets,
                       cudaStream_t stream) {
  kernel_count<<<n_blocks, block_size, 0, stream>>>(
      batch_size, num_nodes, d_counters, d_rrr_sets);
  cuda_check(__FILE__, __LINE__);
}

void CudaCountOccurrencies(
    uint32_t * d_Counters, uint32_t * d_rrr_sets,
    size_t rrr_sets_size, size_t num_nodes, cudaStream_t S) {
  cuda_count_kernel(
      (rrr_sets_size + 255) / 256,
      256, rrr_sets_size, num_nodes, d_Counters, d_rrr_sets, S);
}

void CudaCountOccurrencies(
    uint32_t * d_Counters, uint32_t * d_rrr_sets,
    size_t rrr_sets_size, size_t num_nodes) {
  cudaStream_t compute_stream;
  cuda_stream_create(&compute_stream);

  CudaCountOccurrencies(d_Counters, d_rrr_sets, rrr_sets_size,
                        num_nodes, compute_stream);
  cuda_sync(compute_stream);
}

std::pair<uint32_t, size_t> CudaMaxElement(uint32_t * b, size_t N) {
  thrust::device_ptr<uint32_t> dev_ptr(b);

  thrust::device_ptr<uint32_t> min_ptr = thrust::max_element(thrust::device, dev_ptr, dev_ptr + N);
  uint32_t v = thrust::distance(dev_ptr, min_ptr);
  return std::make_pair(v, size_t(dev_ptr[v]));
}

__global__ void count_uncovered_kernel(
    size_t batch_size, size_t num_nodes,
    uint32_t *d_rrr_index, uint32_t * d_rrr_sets, uint32_t * d_mask,
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


void cuda_count_uncovered_kernel(size_t n_blocks, size_t block_size,
                                 size_t batch_size, size_t num_nodes,
                                 uint32_t *d_rr_vertices, uint32_t * d_rr_edges, uint32_t * d_mask,
                                 uint32_t *d_counters, cudaStream_t stream) {
  count_uncovered_kernel<<<n_blocks, block_size, 0, stream>>>(
      batch_size, num_nodes, d_rr_vertices, d_rr_edges, d_mask, d_counters);
}

__global__ void update_mask_kernel(
    size_t batch_size,
    uint32_t *d_rrr_index, uint32_t * d_rrr_sets,
    uint32_t *d_mask, uint32_t last_seed) {
  size_t pos = threadIdx.x + blockDim.x * blockIdx.x;

  if (pos < batch_size && d_rrr_sets[pos] == last_seed) {
    uint32_t set = d_rrr_index[pos];
    d_mask[set] = 1;
  }
}


void cuda_update_mask_kernel(size_t n_blocks, size_t block_size,
                             size_t batch_size, uint32_t *d_rrr_index,
                             uint32_t * d_rrr_sets, uint32_t * d_mask,
                             uint32_t last_seed, cudaStream_t stream) {
  update_mask_kernel<<<n_blocks, block_size, 0, stream>>>(
      batch_size, d_rrr_index, d_rrr_sets, d_mask, last_seed);
}


void CudaUpdateCounters(cudaStream_t compute_stream,
                        size_t batch_size, uint32_t *d_rr_vertices,
                        uint32_t * d_rr_edges, uint32_t * d_mask,
                        uint32_t * d_Counters, size_t num_nodes,
                        uint32_t last_seed) {
  cudaStream_t data_stream;
  cuda_stream_create(&data_stream);

  cuda_update_mask_kernel((batch_size + 255) / 256, 256, batch_size,
                          d_rr_vertices, d_rr_edges, d_mask, last_seed,
                          compute_stream);

  cuda_memset(reinterpret_cast<void *>(d_Counters), 0, num_nodes * sizeof(uint32_t), data_stream);
  cuda_sync(compute_stream);
  cuda_sync(data_stream);

  cuda_count_uncovered_kernel((batch_size + 255) / 256, 256, batch_size,
                              num_nodes, d_rr_vertices, d_rr_edges, d_mask,
                              d_Counters, compute_stream);

  cuda_sync(compute_stream);
}


void
CudaUpdateCounters(size_t batch_size, uint32_t *d_rr_vertices,
                   uint32_t * d_rr_edges, uint32_t * d_mask,
                   uint32_t * d_Counters, size_t num_nodes,
                   uint32_t last_seed) {
  cudaStream_t compute_stream;
  cuda_stream_create(&compute_stream);

  CudaUpdateCounters(compute_stream, batch_size, d_rr_vertices, d_rr_edges, d_mask,
                     d_Counters, num_nodes, last_seed);
}


size_t CountZeros(char * d_rr_mask, size_t N) {
  thrust::device_ptr<char> dev_ptr = thrust::device_pointer_cast(d_rr_mask);
  char zero = 0;
  return thrust::count(dev_ptr, dev_ptr + N, zero);
}

size_t CountOnes(char * d_rr_mask, size_t N) {
  thrust::device_ptr<char> dev_ptr = thrust::device_pointer_cast(d_rr_mask);
  char one = 1;
  return thrust::count(dev_ptr, dev_ptr + N, one);
}


__global__ void sum_vectors(uint32_t * src, uint32_t * dst, size_t N) {
  size_t pos = threadIdx.x + blockDim.x * blockIdx.x;
  if (pos < N) {
    if (src[pos]) {
      atomicAdd(dst + pos, src[pos]);
    }
  }
}


void cuda_sum_vectors(size_t n_blocks, size_t block_size,
                      size_t batch_size, uint32_t *src, uint32_t * dst,
                      cudaStream_t stream) {
  sum_vectors<<<n_blocks, block_size, 0, stream>>>(src, dst, batch_size);
}

void CudaReduceCounters(cudaStream_t S, uint32_t * src, uint32_t * dest, size_t N) {
  cuda_sum_vectors((N + 255) / 256, 256, N, src, dest, S);
  cuda_sync(S);
}

} // namespace ripples
