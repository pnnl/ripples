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

#ifndef RIPPLES_CUDA_FIND_MOST_INFLUENTIAL_H
#define RIPPLES_CUDA_FIND_MOST_INFLUENTIAL_H

#include <cstddef>
#include <cstdint>
#include <utility>

#include "ripples/gpu/gpu_runtime_trait.h"

namespace ripples {

std::pair<uint32_t, size_t> GPUMaxElement(uint32_t *b, size_t N);

__global__ void update_mask_kernel(size_t batch_size, uint32_t *d_rrr_index,
                                   uint32_t *d_rrr_sets, uint32_t *d_mask,
                                   uint32_t last_seed);
template <GPURuntime R>
void gpu_update_mask_kernel(size_t n_blocks, size_t block_size,
                            size_t batch_size, uint32_t *d_rrr_index,
                            uint32_t *d_rrr_sets, uint32_t *d_mask,
                            uint32_t last_seed,
                            typename GPU<R>::stream_type stream) {
#if defined(RIPPLES_ENABLE_CUDA)
  update_mask_kernel<<<n_blocks, block_size, 0, stream>>>(
      batch_size, d_rrr_index, d_rrr_sets, d_mask, last_seed);
#elif defined(RIPPLES_ENABLE_HIP)
  hipLaunchKernelGGL(update_mask_kernel, n_blocks, block_size, 0, stream,
                     batch_size, d_rrr_index, d_rrr_sets, d_mask, last_seed);
#else
#error "Unsupported GPU runtime"
#endif
}

__global__ void count_uncovered_kernel(size_t batch_size, size_t num_nodes,
                                       uint32_t *d_rrr_index,
                                       uint32_t *d_rrr_sets, uint32_t *d_mask,
                                       uint32_t *d_counters);
template <GPURuntime R>
void gpu_count_uncovered_kernel(size_t n_blocks, size_t block_size,
                                size_t batch_size, size_t num_nodes,
                                uint32_t *d_rr_vertices, uint32_t *d_rr_edges,
                                uint32_t *d_mask, uint32_t *d_counters,
                                typename GPU<R>::stream_type stream) {
#if defined(RIPPLES_ENABLE_CUDA)
  count_uncovered_kernel<<<n_blocks, block_size, 0, stream>>>(
      batch_size, num_nodes, d_rr_vertices, d_rr_edges, d_mask, d_counters);
#elif defined(RIPPLES_ENABLE_HIP)
  hipLaunchKernelGGL(count_uncovered_kernel, n_blocks, block_size, 0, stream,
                     batch_size, num_nodes, d_rr_vertices, d_rr_edges, d_mask,
                     d_counters);
#else
#error "Unsupported GPU runtime"
#endif
}

template <GPURuntime R>
void GPUUpdateCounters(typename GPU<R>::stream_type compute_stream,
                       size_t batch_size, uint32_t *d_rr_vertices,
                       uint32_t *d_rr_edges, uint32_t *d_mask,
                       uint32_t *d_Counters, size_t num_nodes,
                       uint32_t last_seed) {
  typename GPU<R>::stream_type data_stream = GPU<R>::create_stream();

  gpu_update_mask_kernel<R>((batch_size + 255) / 256, 256, batch_size,
                            d_rr_vertices, d_rr_edges, d_mask, last_seed,
                            compute_stream);

  GPU<R>::memset(reinterpret_cast<void *>(d_Counters), 0,
                 num_nodes * sizeof(uint32_t), data_stream);
  GPU<R>::stream_sync(compute_stream);
  GPU<R>::stream_sync(data_stream);

  gpu_count_uncovered_kernel<R>((batch_size + 255) / 256, 256, batch_size,
                                num_nodes, d_rr_vertices, d_rr_edges, d_mask,
                                d_Counters, compute_stream);

  GPU<R>::stream_sync(compute_stream);
}

template <GPURuntime R>
void GPUUpdateCounters(size_t batch_size, uint32_t *d_rr_vertices,
                       uint32_t *d_rr_edges, uint32_t *d_mask,
                       uint32_t *d_Counters, size_t num_nodes,
                       uint32_t last_seed) {
  typename GPU<R>::stream_type compute_stream = GPU<R>::create_stream();

  GPUUpdateCounters<R>(compute_stream, batch_size, d_rr_vertices, d_rr_edges,
                       d_mask, d_Counters, num_nodes, last_seed);
  GPU<R>::stream_sync(compute_stream);
}

__global__ void kernel_count(size_t batch_size, size_t num_nodes,
                             uint32_t *d_counters, uint32_t *d_rrr_sets);

template <GPURuntime R>
void gpu_count_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                      size_t num_nodes, uint32_t *d_counters,
                      uint32_t *d_rrr_sets,
                      typename GPU<R>::stream_type stream) {
#if defined(RIPPLES_ENABLE_CUDA)
  kernel_count<<<n_blocks, block_size, 0, stream>>>(batch_size, num_nodes,
                                                    d_counters, d_rrr_sets);
#elif defined(RIPPLES_ENABLE_HIP)
  hipLaunchKernelGGL(kernel_count, n_blocks, block_size, 0, stream, batch_size,
                     num_nodes, d_counters, d_rrr_sets);
#else
#error "Unsupported GPU runtime"
#endif
}

template <GPURuntime R>
void GPUCountOccurrencies(uint32_t *d_Counters, uint32_t *d_rrr_sets,
                          size_t rrr_sets_size, size_t num_nodes,
                          typename GPU<R>::stream_type S) {
  gpu_count_kernel<R>((rrr_sets_size + 255) / 256, 256, rrr_sets_size,
                      num_nodes, d_Counters, d_rrr_sets, S);
}

template <GPURuntime R>
void GPUCountOccurrencies(uint32_t *d_Counters, uint32_t *d_rrr_sets,
                          size_t rrr_sets_size, size_t num_nodes) {
  typename GPU<R>::stream_type compute_stream = GPU<R>::create_stream();

  GPUCountOccurrencies<R>(d_Counters, d_rrr_sets, rrr_sets_size, num_nodes,
                          compute_stream);
  GPU<R>::stream_sync(compute_stream);
}

__global__ void sum_vectors(uint32_t *src, uint32_t *dst, size_t N);

template <GPURuntime R>
void gpu_sum_vectors(size_t n_blocks, size_t block_size, size_t batch_size,
                     uint32_t *src, uint32_t *dst,
                     typename GPU<R>::stream_type stream) {
#if defined(RIPPLES_ENABLE_CUDA)
  sum_vectors<<<n_blocks, block_size, 0, stream>>>(src, dst, batch_size);
#elif defined(RIPPLES_ENABLE_HIP)
  hipLaunchKernelGGL(sum_vectors, n_blocks, block_size, 0, stream, src, dst,
                     batch_size);
#else
#error "Unsupported GPU runtime"
#endif
}

template <GPURuntime R>
void GPUReduceCounters(typename GPU<R>::stream_type S, uint32_t *src,
                       uint32_t *dest, size_t N) {
  gpu_sum_vectors<R>((N + 255) / 256, 256, N, src, dest, S);
  GPU<R>::stream_sync(S);
}

size_t CountZeros(char *d_rr_mask, size_t N);
size_t CountOnes(char *d_rr_mask, size_t N);

}  // namespace ripples

#endif /* RIPPLES_CUDA_FIND_MOST_INFLUENTIAL_H */
