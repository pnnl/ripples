//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <unordered_map>

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_utils.h"

namespace ripples {

size_t cuda_max_blocks() {
  // TODO query CUDA runtime
  return 1 << 16;
}

size_t cuda_num_devices() {
  int res;
  auto e = cudaGetDeviceCount(&res);
  cuda_check(e, __FILE__, __LINE__);
  return res;
}

void cuda_set_device(size_t gpu_id) {
  auto e = cudaSetDevice(gpu_id);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_stream_create(cudaStream_t *sp) {
  auto e = cudaStreamCreate(sp);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_stream_destroy(cudaStream_t s) {
  auto e = cudaStreamDestroy(s);
  cuda_check(e, __FILE__, __LINE__);
}

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

void cuda_malloc(void **dst, size_t size) {
  cudaError_t e = cudaMalloc(dst, size);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_free(void *ptr) {
  cudaError_t e = cudaFree(ptr);
  cuda_check(e, __FILE__, __LINE__);
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

void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size,
              cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
  cuda_check(__FILE__, __LINE__);
}

void cuda_sync(cudaStream_t stream) { cudaStreamSynchronize(stream); }

}  // namespace ripples