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

struct cuda_conf_t {
  cuda_graph<cuda_GraphTy> *d_graph = nullptr;
  uint8_t *res_masks = nullptr, *d_res_masks = nullptr;
  curandState *d_rng_states = nullptr;
  size_t mask_size = 0, grid_size = 0, block_size = 0, n_blocks = 0;
} cuda_conf;

__global__ void kernel_rng_setup(curandState *d_rng_states,
                                 unsigned long long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, tid, 0, d_rng_states + tid);
}

void cuda_init(const cuda_GraphTy &G, unsigned long long seed,
               im::linear_threshold_tag &&model_tag) {
  cudaError_t e;

  // copy graph to device
  cuda_conf.d_graph = make_cuda_graph(G);

  // sizing
  cuda_conf.block_size = 128;  // 128
  cuda_conf.n_blocks = 512;    // 512
  cuda_conf.grid_size = cuda_conf.n_blocks * cuda_conf.block_size;
  cuda_conf.mask_size = (G.num_nodes() + 7) / 8;

  // print sizing info
  CUDA_LOG("> *** CUDA_BATCHED sizing ***\n");
  CUDA_LOG("block-size =%d\n", cuda_conf.block_size);
  CUDA_LOG("n. blocks  =%d\n", cuda_conf.n_blocks);
  CUDA_LOG("g-mem size =%d\n", cuda_conf.grid_size * cuda_conf.mask_size);
  CUDA_LOG("shmem size =%d\n", cuda_conf.block_size * cuda_conf.mask_size);

  // allocate memory for result-masks
  cuda_conf.res_masks =
      (uint8_t *)malloc(cuda_conf.grid_size * cuda_conf.mask_size);
  e = cudaMalloc(&cuda_conf.d_res_masks,
                 cuda_conf.grid_size * cuda_conf.mask_size);
  cuda_check(e, __FILE__, __LINE__);

  // init rng
  cudaMalloc(&cuda_conf.d_rng_states,
             cuda_conf.grid_size * sizeof(curandState));
  cuda_check(e, __FILE__, __LINE__);

  kernel_rng_setup<<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
      cuda_conf.d_rng_states, seed);
  cuda_check(__FILE__, __LINE__);
}

void cuda_init(const cuda_GraphTy &G, unsigned long long seed,
               im::independent_cascade_tag &&) {}

void cuda_fini(im::linear_threshold_tag &&) {
  // cleanup
  if (cuda_conf.res_masks) free(cuda_conf.res_masks);
  if (cuda_conf.d_res_masks) cudaFree(cuda_conf.d_res_masks);
  if (cuda_conf.d_rng_states) cudaFree(cuda_conf.d_rng_states);
  destroy_cuda_graph(cuda_conf.d_graph);
}

void cuda_fini(im::independent_cascade_tag &&) {}

template <typename HostGraphTy>
__global__ void kernel_lt_per_thread(
    typename HostGraphTy::DestinationTy **index, size_t num_nodes,
    size_t mask_size, curandState *d_rng_states, uint8_t *d_res_masks) {
  using destination_type = typename HostGraphTy::DestinationTy;
  using vertex_type = typename HostGraphTy::vertex_type;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // init res memory
  uint8_t *d_res_mask = d_res_masks + tid * mask_size;
  memset(d_res_mask, 0, mask_size);

  // cache rng state
  auto rng_state = d_rng_states + tid;

  // select source node
  vertex_type src = curand(rng_state) % num_nodes;
  vertex_mask_set(d_res_mask, src);

  float threshold;
  destination_type *first, *last;
  vertex_type v;
  while (src != num_nodes) {
    threshold = curand_uniform(rng_state);
    first = index[src];
    last = index[src + 1];
    src = num_nodes;
    for (; first != last; ++first) {
      threshold -= first->weight;
      if (threshold <= 0) {
        v = first->vertex;
        if (!vertex_mask_get(d_res_mask, v)) {
          src = v;
          vertex_mask_set(d_res_mask, src);
        }
        break;
      }
    }
  }
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::linear_threshold_tag &&model_tag) {
  using vertex_type = typename cuda_GraphTy::vertex_type;

  cuda_res_t rrr_sets(theta);

#if CUDA_BATCHED
  for (size_t bf = 0; bf < rrr_sets.size(); bf += cuda_conf.grid_size) {
    // execute a batch
    kernel_lt_per_thread<cuda_GraphTy>
        <<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
            cuda_conf.d_graph->d_index_, G.num_nodes(), cuda_conf.mask_size,
            cuda_conf.d_rng_states, cuda_conf.d_res_masks);
    cuda_check(__FILE__, __LINE__);

    // copy masks back to host
    cudaMemcpy(cuda_conf.res_masks, cuda_conf.d_res_masks,
               cuda_conf.grid_size * cuda_conf.mask_size,
               cudaMemcpyDeviceToHost);

    // convert masks to results
    // TODO optimize bitwise operations
    for (size_t i = 0; i < cuda_conf.grid_size && (bf + i) < rrr_sets.size();
         ++i) {
      auto &rrr_set = rrr_sets[bf + i];
      auto res_mask = cuda_conf.res_masks + (i * cuda_conf.mask_size);
      for (size_t j = 0; j < cuda_conf.mask_size; ++j) {
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
#endif

  return rrr_sets;
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::independent_cascade_tag &&model_tag) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace im
