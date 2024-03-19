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

#ifndef RIPPLES_GPU_LT_KERNEL_H
#define RIPPLES_GPU_LT_KERNEL_H

#include "ripples/gpu/generate_rrr_sets.h"
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#if defined(RIPPLES_ENABLE_CUDA)
#include "ripples/cuda/cuda_utils.h"
#endif

namespace ripples {

template <GPURuntime R, typename GraphTy, typename gpu_PRNGeneratorTy>
__global__ void kernel_lt_per_thread(
    size_t bs, typename gpu_graph<R, GraphTy>::index_t *index,
    typename gpu_graph<R, GraphTy>::vertex_t *edges,
    typename gpu_graph<R, GraphTy>::weight_t *weights, size_t num_nodes,
    gpu_PRNGeneratorTy *d_trng_states, mask_word_t *d_res_masks,
    size_t num_mask_words) {
  using vertex_type = typename gpu_graph<R, GraphTy>::vertex_t;
  using index_type = typename gpu_graph<R, GraphTy>::index_t;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < bs) {
    trng::uniform01_dist<float> u;
    trng::uniform_int_dist root_dist(0, num_nodes);

    // init res memory
    mask_word_t dr_res_mask[GPU_WALK_SIZE];
    size_t res_size = 0;

    // cache rng state
    auto &r(d_trng_states[tid]);

    // select source node
    vertex_type src = root_dist(r);
    dr_res_mask[res_size++] = src;

    float threshold;
    index_type first, last;
    vertex_type v;
    while (src != num_nodes) {
      // rng
      threshold = u(r);

      // scan neighbor list
      first = index[src];
      last = index[src + 1];
      src = num_nodes;
      for (; first != last; ++first) {
        threshold -= weights[first];
        if (threshold > 0) continue;

        // found candidate vertex
        v = edges[first];

        // insert if not visited
        size_t i = 0;
        while (i < res_size && dr_res_mask[i] != v) ++i;
        if (i == res_size) {
          // not visited
          if (res_size < num_mask_words) {
            // available result slot
            src = v;
            dr_res_mask[res_size++] = v;
          } else {
            // invalidate the walk
            dr_res_mask[1] = dr_res_mask[0];
            dr_res_mask[0] = num_nodes;
            res_size = num_mask_words;
          }
        }
        break;
      }
    }

    // mark end-of-set
    if (res_size < num_mask_words) dr_res_mask[res_size] = num_nodes;

    // write back to global memory
    auto d_res_mask = d_res_masks + tid * num_mask_words;
    memcpy(d_res_mask, dr_res_mask, GPU_WALK_SIZE * sizeof(mask_word_t));
  }  // end if active thread
}

template <GPURuntime R, typename GraphTy, typename gpu_PRNGeneratorTy>
void gpu_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                   size_t num_nodes, gpu_PRNGeneratorTy *d_trng_states,
                   mask_word_t *d_res_masks, size_t num_mask_words,
                   gpu_ctx<R, GraphTy> *ctx,
                   typename GPU<R>::stream_type stream) {
#if defined(RIPPLES_ENABLE_CUDA)
  kernel_lt_per_thread<R, GraphTy><<<n_blocks, block_size, 0, stream>>>(
      batch_size, ctx->d_graph->d_index_, ctx->d_graph->d_edges_,
      ctx->d_graph->d_weights_, num_nodes, d_trng_states, d_res_masks,
      num_mask_words);
#elif defined(RIPPLES_ENABLE_HIP)
  hipLaunchKernelGGL((kernel_lt_per_thread<R, GraphTy>), n_blocks, block_size,
                     0, stream, batch_size, ctx->d_graph->d_index_,
                     ctx->d_graph->d_edges_, ctx->d_graph->d_weights_,
                     num_nodes, d_trng_states, d_res_masks, num_mask_words);
#else
#error "Unsupported GPU runtime"
#endif
}
}  // namespace ripples
#endif
