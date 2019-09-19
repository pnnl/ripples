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

#include <unordered_map>

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_utils.h"

namespace ripples {

__global__ void kernel_lt_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                     cuda_PRNGeneratorTy r, size_t num_seqs,
                                     size_t first_seq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  d_trng_states[tid] = r;
  d_trng_states[tid].split(num_seqs, first_seq + tid);
}

__global__ void kernel_ic_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                     cuda_PRNGeneratorTy r, size_t num_seqs,
                                     size_t first_seq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  d_trng_states[tid] = r;
  d_trng_states[tid].split(num_seqs, first_seq + tid);
}

typename cuda_device_graph::vertex_t *cuda_graph_index(cuda_ctx *ctx) {
  return ctx->d_graph->d_index_;
}

typename cuda_device_graph::vertex_t *cuda_graph_edges(cuda_ctx *ctx) {
  return ctx->d_graph->d_edges_;
}

typename cuda_device_graph::weight_t *cuda_graph_weights(cuda_ctx *ctx) {
  return ctx->d_graph->d_weights_;
}

void cuda_lt_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size) {
  kernel_lt_trng_setup<<<n_blocks, block_size>>>(d_trng_state, r, num_seqs,
                                                 first_seq);
  cuda_check(__FILE__, __LINE__);
}

void cuda_ic_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size) {
  kernel_ic_trng_setup<<<n_blocks, block_size>>>(d_trng_state, r, num_seqs,
                                                 first_seq);
  cuda_check(__FILE__, __LINE__);
}

cuda_ctx *cuda_make_ctx(const cuda_GraphTy &G, size_t gpu_id) {
  auto res = new cuda_ctx();
  res->gpu_id = gpu_id;
  cuda_set_device(gpu_id);
  res->d_graph = make_cuda_graph(G);
  return res;
}

void cuda_destroy_ctx(cuda_ctx *ctx) {
  cuda_set_device(ctx->gpu_id);
  destroy_cuda_graph(ctx->d_graph);
}

__global__ void kernel_lt_per_thread(
    size_t bs, typename cuda_device_graph::vertex_t *index,
    typename cuda_device_graph::vertex_t *edges,
    typename cuda_device_graph::weight_t *weights, size_t num_nodes,
    cuda_PRNGeneratorTy *d_trng_states, mask_word_t *d_res_masks,
    size_t num_mask_words) {
  using vertex_type = typename cuda_device_graph::vertex_t;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < bs) {
    trng::uniform01_dist<float> u;
    trng::uniform_int_dist root_dist(0, num_nodes);

    // init res memory
    mask_word_t dr_res_mask[CUDA_WALK_SIZE];
    size_t res_size = 0;

    // cache rng state
    auto &r(d_trng_states[tid]);

    // select source node
    vertex_type src = root_dist(r);
    dr_res_mask[res_size++] = src;

    float threshold;
    vertex_type first, last;
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
    memcpy(d_res_mask, dr_res_mask, CUDA_WALK_SIZE * sizeof(mask_word_t));
  }  // end if active thread
}


void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                    size_t num_nodes, cuda_PRNGeneratorTy *d_trng_states,
                    mask_word_t *d_res_masks, size_t num_mask_words,
                    cuda_ctx *ctx, cudaStream_t stream) {
  kernel_lt_per_thread<<<n_blocks, block_size, 0, stream>>>(
      batch_size, ctx->d_graph->d_index_, ctx->d_graph->d_edges_,
      ctx->d_graph->d_weights_, num_nodes, d_trng_states, d_res_masks,
      num_mask_words);
  cuda_check(__FILE__, __LINE__);
}

}  // namespace ripples
