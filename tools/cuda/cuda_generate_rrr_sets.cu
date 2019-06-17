//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/cuda_graph.cuh"

namespace ripples {

// tested configurations:
// + 1 walk per thread:
// - warp_step = 1
//
// + 1 walk per warp:
// - warp_step = cuda_prop.warpSize
//
// + 1 walk per block:
// - warp step = 1
// - block_size = 1
struct cuda_ctx_t {
    cuda_graph<cuda_GraphTy> *d_graph = nullptr;
} cuda_ctx;

__global__ void kernel_trng_setup(cuda_PRNGeneratorTy *d_trng_states, size_t rank, cuda_PRNGeneratorTy r, size_t warp_size,
                                  size_t num_threads) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_size == 0) {
    int wid = tid / warp_size;

    d_trng_states[wid] = r;
    d_trng_states[wid].split(MAX_PARDEG * num_threads, rank * MAX_PARDEG + wid);
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
		const cuda_PRNGeneratorTy &r, size_t warp_step, size_t max_batch_size) {
	kernel_trng_setup<<<n_blocks, block_size>>>(
			d_trng_state, rank, r, warp_step, max_batch_size);
	cuda_check(__FILE__, __LINE__);
}

void cuda_graph_fini() {
  // cleanup
  destroy_cuda_graph(cuda_ctx.d_graph);
}

template <typename HostGraphTy>
__global__ void kernel_lt_per_thread(
    size_t bs, typename HostGraphTy::DestinationTy **index, size_t num_nodes,
    size_t warp_size, cuda_PRNGeneratorTy *d_trng_states, mask_word_t *d_res_masks) {
  using destination_type = typename HostGraphTy::DestinationTy;
  using vertex_type = typename HostGraphTy::vertex_type;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_size == 0) {
    int wid = tid / warp_size;
    if (wid < bs) {
      trng::uniform01_dist<float> u;
      trng::uniform_int_dist root_dist(0, num_nodes);
      size_t res_size = 0;

      // init res memory
      auto d_res_mask = d_res_masks + wid * MAX_SET_SIZE;
      memset(d_res_mask, 0, MAX_SET_SIZE * sizeof(mask_word_t));

      // cache rng state
      auto &r(d_trng_states[wid]);

      // select source node
      vertex_type src = root_dist(r);
      d_res_mask[res_size++] = src;

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
          while (i < res_size && d_res_mask[i] != v) ++i;
          if (i == res_size) {
              src = v;
              d_res_mask[res_size++] = v;
          }
          break;
        }
      }

      // mark end-of-set
      if (res_size < MAX_SET_SIZE) d_res_mask[res_size] = num_nodes;
    }  // end if active warp
  }    // end if active thread-in-warp
}  // namespace ripples

void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
		size_t num_nodes, size_t warp_step, cuda_PRNGeneratorTy *d_trng_states,
		mask_word_t *d_res_masks) {
	kernel_lt_per_thread<cuda_GraphTy> <<<n_blocks, block_size>>>(
			batch_size, cuda_ctx.d_graph->d_index_, num_nodes,
			warp_step, d_trng_states, d_res_masks);
	cuda_check(__FILE__, __LINE__);
}

void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  cuda_check(__FILE__, __LINE__);
}

}  // namespace ripples
