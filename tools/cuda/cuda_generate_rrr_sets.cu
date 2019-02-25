//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <curand_kernel.h>

#include "im/cuda/cuda_generate_rrr_sets.h"
#include "im/cuda/cuda_graph.cuh"
#include "im/cuda/cuda_utils.h"

#define CUDA_BATCHED 1

#define vertex_mask_set(m, v) \
  { (m)[(v) >> 3] |= (uint8_t)1 << ((v) & (vertex_type)7); }
#define vertex_mask_get(m, v) \
  ((m)[(v) >> 3] & (uint8_t)1 << ((v) & (vertex_type)7))

namespace im {

extern __shared__ uint8_t shmem_lt_per_thread[];
template <typename DeviceGraphTy>
__global__ void kernel_lt_per_thread(
    typename DeviceGraphTy::destination_type **index, size_t num_nodes,
    size_t mask_size, curandState *d_rng_states, uint8_t *d_res_masks) {
  using destination_type = typename DeviceGraphTy::destination_type;
  using vertex_type = typename DeviceGraphTy::vertex_type;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // init res memory
  uint8_t *shmem_res_mask = shmem_lt_per_thread + threadIdx.x * mask_size;
  memset(shmem_res_mask, 0, mask_size);

  // cache rng state
  auto rng_state = d_rng_states + tid;

  // select source node
  vertex_type src = curand(rng_state) % num_nodes;
  vertex_mask_set(shmem_res_mask, src);

  float threshold;
  destination_type *first, *last;
  vertex_type v;
  CUDA_LOG("> [kernel] START root=%d\n", src);
  while (src != num_nodes) {
    CUDA_LOG("> [kernel] insert %d\n", src);
    threshold = curand_uniform(rng_state);
    first = index[src];
    last = index[src + 1];
    src = num_nodes;
    for (; first != last; ++first) {
      threshold -= first->weight;
      CUDA_LOG("> [kernel] visiting: v=%d w=%f (threshold=%f)\n", first->vertex,
               first->weight, threshold);
      if (threshold <= 0) {
        v = first->vertex;
        if (!vertex_mask_get(shmem_res_mask, v)) {
          src = v;
          vertex_mask_set(shmem_res_mask, src);
        }
        break;
      }
    }
  }
  CUDA_LOG("> [kernel] END\n");

  // print mask for debug
  CUDA_LOG("> [kernel threadIdx.x=%d] shmem_res_mask first=%p last=%p\n",
           threadIdx.x, shmem_res_mask, shmem_res_mask + mask_size);
  for (int i = 0; i < mask_size; ++i) {
    CUDA_LOG("> [kernel threadIdx.x=%d] shmem_res_mask w=%d\t: %d\n",
             threadIdx.x, i, shmem_res_mask[i]);
  }

  // write back results to global memory
  uint8_t *d_res_mask = d_res_masks + tid * mask_size;
  memcpy(d_res_mask, shmem_res_mask, mask_size);
}

__global__ void kernel_rng_setup(curandState *d_rng_states,
                                 unsigned long long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, tid, 0, d_rng_states + tid);
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::linear_threshold_tag &&model_tag,
                               unsigned long long seed) {
  using vertex_type = typename cuda_GraphTy::vertex_type;

  cuda_res_t rrr_sets(theta);

  // copy graph to device
  cuda_graph<cuda_GraphTy> d_graph(G);

#if CUDA_BATCHED
  // sizing
  size_t block_size = 512;  // 512
  size_t n_blocks = 128;    // 128
  size_t grid_size = n_blocks * block_size;
  size_t mask_size = (G.num_nodes() + 7) / 8;

  // print sizing info
  CUDA_LOG("> *** CUDA_BATCHED sizing ***\n");
  CUDA_LOG("block-size =%d\n", block_size);
  CUDA_LOG("n. blocks  =%d\n", n_blocks);
  CUDA_LOG("g-mem size =%d\n", grid_size * mask_size);
  CUDA_LOG("shmem size =%d\n", block_size * mask_size);

  // allocate memory for result-masks
  uint8_t *res_masks, *d_res_masks;
  res_masks = (uint8_t *)malloc(grid_size * mask_size);
  cudaMalloc(&d_res_masks, grid_size * mask_size);

  // init rng
  curandState *d_rng_states;
  cudaMalloc(&d_rng_states, grid_size * sizeof(curandState));
  kernel_rng_setup<<<n_blocks, block_size>>>(d_rng_states, seed);

  for (size_t bf = 0; bf < rrr_sets.size(); bf += grid_size) {
    CUDA_LOG("instance %d/%d\n", bf, rrr_sets.size());
    fflush(stdout);
    // execute a batch
    kernel_lt_per_thread<cuda_graph<cuda_GraphTy>>
        <<<n_blocks, block_size, block_size * mask_size>>>(
            d_graph.d_index_, G.num_nodes(), mask_size, d_rng_states,
            d_res_masks);

    // copy masks back to host
    CUDA_LOG("> copying back masks batch-first=%d\n", bf);
    fflush(stdout);
    cudaMemcpy(res_masks, d_res_masks, grid_size * mask_size,
               cudaMemcpyDeviceToHost);

    // convert masks to results
    // TODO optimize bitwise operations
    CUDA_LOG("> converting sets\n");
    fflush(stdout);
    for (size_t i = 0; i < grid_size && (bf + i) < rrr_sets.size(); ++i) {
      auto &rrr_set = rrr_sets[bf + i];
      auto res_mask = res_masks + (i * mask_size);
      for (size_t j = 0; j < mask_size; ++j) {
        // scan a word from the res mask
        uint8_t w = res_mask[j];
        for (uint8_t bi = 0; bi < 8; ++bi) {
          if (w & (uint8_t)1) {
            // convert mask bit to vertex id
            auto v = vertex_type(8 * j + bi);
            rrr_set.push_back(v);
          }
          w >>= 1;
        }
      }

      check_lt(rrr_set, G, bf + i);
    }
  }

  // cleanup
  free(res_masks);
  cudaFree(d_res_masks);
  cudaFree(d_rng_states);
#endif

  return rrr_sets;
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::independent_cascade_tag &&model_tag,
                               unsigned long long seed) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace im
