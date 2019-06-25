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
  cuda_graph<cuda_GraphTy> *d_graph = nullptr;
} cuda_ctx;

size_t cuda_warp_size() {
  cudaDeviceProp cuda_prop;
  cudaGetDeviceProperties(&cuda_prop, 0);
  return cuda_prop.warpSize;
}

__global__ void kernel_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                  size_t rank, cuda_PRNGeneratorTy r,
                                  size_t warp_step, size_t num_threads,
                                  size_t num_total_threads, size_t rng_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_step == 0) {
    int wid = tid / warp_step;

    d_trng_states[wid] = r;
    d_trng_states[wid].split(num_total_threads,
                             rng_offset + rank * num_threads + wid);
  }
}

void cuda_graph_init(const cuda_GraphTy &G) {
  cuda_ctx.d_graph = make_cuda_graph(G);
}

void cuda_malloc(void **dst, size_t size) {
  cudaError_t e = cudaMalloc(dst, size);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_free(void *ptr) {
  cudaError_t e = cudaFree(ptr);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_rng_setup(size_t n_blocks, size_t block_size,
                    cuda_PRNGeneratorTy *d_trng_state, size_t rank,
                    const cuda_PRNGeneratorTy &r, size_t warp_step,
                    size_t max_batch_size, size_t num_total_threads,
                    size_t rng_offset) {
  kernel_trng_setup<<<n_blocks, block_size>>>(d_trng_state, rank, r, warp_step,
                                              max_batch_size, num_total_threads,
                                              rng_offset);
  cuda_check(__FILE__, __LINE__);
}

void cuda_graph_fini() {
  // cleanup
  destroy_cuda_graph(cuda_ctx.d_graph);
}

template <typename HostGraphTy>
__global__ void kernel_lt_per_thread(
    size_t bs, typename HostGraphTy::DestinationTy **index, size_t num_nodes,
    size_t warp_step, cuda_PRNGeneratorTy *d_trng_states,
    mask_word_t *d_res_masks, size_t num_mask_words) {
  using destination_type = typename HostGraphTy::DestinationTy;
  using vertex_type = typename HostGraphTy::vertex_type;

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
      destination_type *first, *last;
      vertex_type v;
      while (src != num_nodes) {
        // rng
        threshold = u(r);

        // scan neighbor list
        first = index[src];
        last = index[src + 1];
        src = num_nodes;
        for (; first != last; ++first) {
          threshold -= first->weight;
          if (threshold > 0) continue;

          // found candidate vertex
          v = first->vertex;

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

struct circular_buffer {
  __device__ circular_buffer(mask_word_t nil) : nil_(nil) {
    // printf("> [circular_buffer]:");
    // printf("\tsize=%d", size_);
    // printf("\tnil=%d\n", nil_);
    for (size_t i = 0; i < size_; ++i) {
      buf_[i] = nil_;
    }
    pread_ = pwrite_ = 0;
  }

  __device__ void invalidate() { buf_[pread_] = nil_; }

  __device__ bool push(mask_word_t x) {
    // printf("> [circular_buffer::push]:");
    // printf("\tpwrite=%d", pwrite_);
    // printf("\tx=%d", x);
    // printf("\tsize=%d", size_);
    // printf("\tnil=%d\n", nil_);
    if (buf_[pwrite_] == nil_) {
      buf_[pwrite_] = x;
      pwrite_ = (pwrite_ + 1) % size_;
      return true;
    }
    return false;
  }

  __device__ bool pop(mask_word_t &dst) {
    // printf("> [circular_buffer::pop]:");
    // printf("\tpread=%d\n", pread_);
    if (buf_[pread_] != nil_) {
      dst = buf_[pread_];
      buf_[pread_] = nil_;
      pread_ = (pread_ + 1) % size_;
      return true;
    }
    return false;
  }

  mask_word_t buf_[CUDA_WALK_SIZE];
  const size_t size_ = CUDA_WALK_SIZE;
  size_t pwrite_, pread_;
  const mask_word_t nil_;
};

template <typename HostGraphTy>
__global__ void kernel_ic_per_thread(
    size_t bs, typename HostGraphTy::DestinationTy **index, size_t num_nodes,
    size_t warp_step, cuda_PRNGeneratorTy *d_trng_states,
    mask_word_t *d_res_masks, size_t num_mask_words) {
  using destination_type = typename HostGraphTy::DestinationTy;
  using vertex_type = typename HostGraphTy::vertex_type;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_step == 0) {
    int wid = tid / warp_step;
    if (wid < bs) {
      trng::uniform01_dist<float> u;
      trng::uniform_int_dist root_dist(0, num_nodes);

      // init frontiers memory
      circular_buffer d_frontier(num_nodes);

      // init res memory
      mask_word_t dr_res_mask[CUDA_WALK_SIZE];
      size_t res_size = 0;

      // cache rng state
      auto &r(d_trng_states[wid]);

      // select source node
      vertex_type src = root_dist(r);
      d_frontier.push(src);
      dr_res_mask[res_size++] = src;

      while (d_frontier.pop(src)) {
        // scan neighbor list
        auto first = index[src];
        auto last = index[src + 1];
        for (; first != last; ++first) {
          if (u(r) <= first->weight) {
            // node selected
            size_t i = 0;
            auto v = first->vertex;
            while (i < res_size && dr_res_mask[i] != v) ++i;
            if (i == res_size) {
              // not visited yet
              if (d_frontier.push(v) && res_size < num_mask_words) {
                // add to result
                dr_res_mask[res_size++] = v;
              } else {
                // invalidate the walk
                dr_res_mask[1] = dr_res_mask[0];
                dr_res_mask[0] = num_nodes;
                res_size = num_mask_words;
                d_frontier.invalidate();
                break;
              }
            }
          }
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
                    mask_word_t *d_res_masks, size_t num_mask_words) {
  kernel_lt_per_thread<cuda_GraphTy><<<n_blocks, block_size>>>(
      batch_size, cuda_ctx.d_graph->d_index_, num_nodes, warp_step,
      d_trng_states, d_res_masks, num_mask_words);
  cuda_check(__FILE__, __LINE__);
}

void cuda_ic_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                    size_t num_nodes, size_t warp_step,
                    cuda_PRNGeneratorTy *d_trng_states,
                    mask_word_t *d_res_masks, size_t num_mask_words) {
  kernel_ic_per_thread<cuda_GraphTy><<<n_blocks, block_size>>>(
      batch_size, cuda_ctx.d_graph->d_index_, num_nodes, warp_step,
      d_trng_states, d_res_masks, num_mask_words);
  cuda_check(__FILE__, __LINE__);
}

void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  cuda_check(__FILE__, __LINE__);
}

}  // namespace ripples
