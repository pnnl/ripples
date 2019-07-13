//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_utils.h"

namespace ripples {

struct cuda_ctx_t {
  cuda_device_graph *d_graph = nullptr;
} cuda_ctx;

size_t cuda_max_blocks() {
  // TODO query CUDA runtime
  return 1 << 16;
}

size_t cuda_warp_size() {
  cudaDeviceProp cuda_prop;
  cudaGetDeviceProperties(&cuda_prop, 0);
  return cuda_prop.warpSize;
}

__global__ void kernel_lt_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                     cuda_PRNGeneratorTy r, size_t num_seqs,
                                     size_t first_seq, size_t warp_step) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_step == 0) {
    int wid = tid / warp_step;

    d_trng_states[wid] = r;
    d_trng_states[wid].split(num_seqs, first_seq + wid);
  }
}

__global__ void kernel_ic_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                     cuda_PRNGeneratorTy r, size_t num_seqs,
                                     size_t first_seq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  d_trng_states[tid] = r;
  d_trng_states[tid].split(num_seqs, first_seq + tid);
}

void cuda_graph_init(const cuda_GraphTy &G) {
  cuda_ctx.d_graph = make_cuda_graph(G);
}

typename cuda_device_graph::vertex_t *cuda_graph_index() {
  return cuda_ctx.d_graph->d_index_;
}

typename cuda_device_graph::vertex_t *cuda_graph_edges() {
  return cuda_ctx.d_graph->d_edges_;
}

typename cuda_device_graph::weight_t *cuda_graph_weights() {
  return cuda_ctx.d_graph->d_weights_;
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
                       size_t first_seq, size_t n_blocks, size_t block_size,
                       size_t warp_step) {
  kernel_lt_trng_setup<<<n_blocks, block_size>>>(d_trng_state, r, num_seqs,
                                                 first_seq, warp_step);
  cuda_check(__FILE__, __LINE__);
}

void cuda_ic_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
  const cuda_PRNGeneratorTy &r, size_t num_seqs,
  size_t first_seq, size_t n_blocks, size_t block_size) {
  kernel_ic_trng_setup<<<n_blocks, block_size>>>(d_trng_state, r, num_seqs,
                                                 first_seq);
  cuda_check(__FILE__, __LINE__);
}

void cuda_graph_fini() {
  // cleanup
  destroy_cuda_graph(cuda_ctx.d_graph);
}

__global__ void kernel_lt_per_thread(size_t bs,
                                     typename cuda_device_graph::vertex_t *index,
                                     typename cuda_device_graph::vertex_t *edges,
                                     typename cuda_device_graph::weight_t *weights,
                                     size_t num_nodes, size_t warp_step,
                                     cuda_PRNGeneratorTy *d_trng_states,
                                     mask_word_t *d_res_masks,
                                     size_t num_mask_words) {
  using vertex_type = typename cuda_device_graph::vertex_t;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_step == 0) {
    int wid = tid / warp_step;
    if (wid < bs) {
      trng::uniform01_dist<float> u;
      trng::uniform_int_dist root_dist(0, num_nodes);

      // init res memory
      mask_word_t dr_res_mask[CUDA_WALK_SIZE];
      size_t res_size = 0;

      // cache rng state
      auto &r(d_trng_states[wid]);

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
      auto d_res_mask = d_res_masks + wid * num_mask_words;
      memcpy(d_res_mask, dr_res_mask, CUDA_WALK_SIZE * sizeof(mask_word_t));
    }  // end if active warp
  }    // end if active thread-in-warp
}

void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                    size_t num_nodes, size_t warp_step,
                    cuda_PRNGeneratorTy *d_trng_states,
                    mask_word_t *d_res_masks, size_t num_mask_words, cudaStream_t stream) {
  kernel_lt_per_thread<<<n_blocks, block_size, 0, stream>>>(
      batch_size, cuda_ctx.d_graph->d_index_, cuda_ctx.d_graph->d_edges_,
      cuda_ctx.d_graph->d_weights_, num_nodes, warp_step, d_trng_states,
      d_res_masks, num_mask_words);
  cuda_check(__FILE__, __LINE__);
}

void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size, cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
  cuda_check(__FILE__, __LINE__);
}

void cuda_sync(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
}

}  // namespace ripples
