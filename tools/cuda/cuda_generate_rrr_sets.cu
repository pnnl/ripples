//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <curand_kernel.h>

#include "im/cuda/cuda_generate_rrr_sets.h"
#include "im/cuda/cuda_graph.cuh"
#include "im/cuda/cuda_utils.h"

#define is_set_in_word(w, v) ((w) & ((uint8_t)1 << ((v) & (vertex_type)7)))

#define set_in_word(w, v) \
  { (w) |= ((uint8_t)1 << ((v) & (vertex_type)7)); }

#define read_mask_word(m, v) (m)[(v) >> 3]

#define write_mask_word(m, v, w) \
  { (m)[(v) >> 3] = (w); }

namespace im {

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
struct cuda_conf_t {
  cudaDeviceProp cuda_prop;
  cuda_graph<cuda_GraphTy> *d_graph = nullptr;
  uint8_t *res_masks = nullptr, *d_res_masks = nullptr;
  curandState *d_rng_states = nullptr;
  size_t grid_size = 0, block_size = 0, n_blocks = 0;
  size_t warp_step = 0;   // 1: per-thread, warp-size: per-warp
  size_t batch_size = 0;  // walks per batch
  size_t mask_size = 0;
} cuda_conf;

__global__ void kernel_rng_setup(curandState *d_rng_states,
                                 unsigned long long seed, int warp_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_size == 0) {
    int wid = tid / warp_size;
    curand_init(seed, wid, 0, d_rng_states + wid);
  }
}

void cuda_init(const cuda_GraphTy &G, unsigned long long seed,
               im::linear_threshold_tag &&model_tag) {
  cudaError_t e;

  cudaGetDeviceProperties(&cuda_conf.cuda_prop, 0);

  // copy graph to device
  cuda_conf.d_graph = make_cuda_graph(G);

  // sizing
  // cuda_conf.warp_step = 1;                             // per thread
  cuda_conf.warp_step = cuda_conf.cuda_prop.warpSize;  // per warp
  cuda_conf.block_size = 256;                          // 128
  cuda_conf.n_blocks = 512;                            // 512
  cuda_conf.grid_size = cuda_conf.n_blocks * cuda_conf.block_size;
  cuda_conf.batch_size = cuda_conf.grid_size / cuda_conf.warp_step;
  cuda_conf.mask_size = (G.num_nodes() + 7) / 8;

  // print sizing info
  CUDA_LOG("> *** CUDA_BATCHED sizing ***\n");
  CUDA_LOG("block-size = %d\n", cuda_conf.block_size);
  CUDA_LOG("n. blocks  = %d\n", cuda_conf.n_blocks);
  CUDA_LOG("warp size  = %d\n", cuda_conf.cuda_prop.warpSize);
  CUDA_LOG("grid size  = %d\n", cuda_conf.grid_size);
  CUDA_LOG("batch size = %d\n", cuda_conf.batch_size);
  CUDA_LOG("g-mem size = %d\n", cuda_conf.grid_size * cuda_conf.mask_size);
  CUDA_LOG("shmem size = %d\n", cuda_conf.block_size * cuda_conf.mask_size);

  // allocate memory for result-masks
  cuda_conf.res_masks =
      (uint8_t *)malloc(cuda_conf.batch_size * cuda_conf.mask_size);
  e = cudaMalloc(&cuda_conf.d_res_masks,
                 cuda_conf.batch_size * cuda_conf.mask_size);
  cuda_check(e, __FILE__, __LINE__);

  // init rng
  cudaMalloc(&cuda_conf.d_rng_states,
             cuda_conf.batch_size * sizeof(curandState));
  cuda_check(e, __FILE__, __LINE__);

#if CUDA_PER_WARP
  kernel_rng_setup<<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
      cuda_conf.d_rng_states, seed, cuda_conf.cuda_prop.warpSize);
#else
  kernel_rng_setup<<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
      cuda_conf.d_rng_states, seed, 1);
#endif
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
    size_t mask_size, size_t warp_size, curandState *d_rng_states,
    uint8_t *d_res_masks) {
  using destination_type = typename HostGraphTy::DestinationTy;
  using vertex_type = typename HostGraphTy::vertex_type;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_size == 0) {
    int wid = tid / warp_size;
    uint8_t mask_word = 0;

    // init res memory
    uint8_t *d_res_mask = d_res_masks + wid * mask_size;
    memset(d_res_mask, 0, mask_size);

    // cache rng state
    auto rng_state = d_rng_states + wid;

    // select source node
    vertex_type src = curand(rng_state) % num_nodes;
    set_in_word(mask_word, src);
    write_mask_word(d_res_mask, src, mask_word);

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
          mask_word = read_mask_word(d_res_mask, v);
          if (!is_set_in_word(mask_word, v)) {
            src = v;
            set_in_word(mask_word, src);
            write_mask_word(d_res_mask, v, mask_word);
          }
          break;
        }
      }
    }
  }
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::linear_threshold_tag &&model_tag) {
  using vertex_type = typename cuda_GraphTy::vertex_type;

  cuda_res_t rrr_sets(theta);

  for (size_t bf = 0; bf < rrr_sets.size(); bf += cuda_conf.batch_size) {
    // execute a batch
#if CUDA_PER_WARP
    kernel_lt_per_thread<cuda_GraphTy>
        <<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
            cuda_conf.d_graph->d_index_, G.num_nodes(), cuda_conf.mask_size,
            cuda_conf.cuda_prop.warpSize, cuda_conf.d_rng_states,
            cuda_conf.d_res_masks);
#else
    kernel_lt_per_thread<cuda_GraphTy>
        <<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
            cuda_conf.d_graph->d_index_, G.num_nodes(), cuda_conf.mask_size, 1,
            cuda_conf.d_rng_states, cuda_conf.d_res_masks);
#endif
    cuda_check(__FILE__, __LINE__);

    // copy masks back to host
    cudaMemcpy(cuda_conf.res_masks, cuda_conf.d_res_masks,
               cuda_conf.batch_size * cuda_conf.mask_size,
               cudaMemcpyDeviceToHost);

    // convert masks to results
    // TODO optimize bitwise operations
    for (size_t i = 0; i < cuda_conf.batch_size && (bf + i) < rrr_sets.size();
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

  return rrr_sets;
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::independent_cascade_tag &&model_tag) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace im
