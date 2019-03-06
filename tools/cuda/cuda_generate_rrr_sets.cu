//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <curand_kernel.h>
#include <chrono>
#include <cstring>
#include <iostream>

#include "im/cuda/cuda_generate_rrr_sets.h"
#include "im/cuda/cuda_graph.cuh"
#include "im/cuda/cuda_utils.h"

// macros for 32-bit mask encoding
#define is_set_in_word(w, v) ((w) & ((uint32_t)1 << ((v) & (vertex_type)31)))

#define set_in_word(w, v) \
  { (w) |= ((uint32_t)1 << ((v) & (vertex_type)31)); }

#define read_mask_word(m, v) (m)[(v) >> 5]

#define write_mask_word(m, v, w) \
  { (m)[(v) >> 5] = (w); }

namespace im {

using mask_word_t = unsigned long long;
using d_mask_word_t = uint32_t;

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
  mask_word_t *res_masks = nullptr;
  d_mask_word_t *d_res_masks = nullptr;
  size_t *res_sizes = nullptr, *d_res_sizes = nullptr;
  curandState *d_rng_states = nullptr;
  size_t grid_size = 0, block_size = 0, n_blocks = 0;
  size_t warp_step = 0;   // 1: per-thread, warp-size: per-warp
  size_t batch_size = 0;  // walks per batch
  size_t mask_words = 0;
} cuda_conf;

__global__ void kernel_rng_setup(curandState *d_rng_states,
                                 unsigned long long seed, int warp_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_size == 0) {
    int wid = tid / warp_size;
    curand_init(seed, wid, 0, d_rng_states + wid);
  }
}

#if CUDA_PROFILE
struct cuda_profile {
  clock_t rng, el;
} * profile, *d_profile;
#endif

void cuda_init(const cuda_GraphTy &G, unsigned long long seed,
               im::linear_threshold_tag &&model_tag) {
  static_assert(
      sizeof(mask_word_t) == sizeof(unsigned long long),
      "host-side mask-word must have the same size as unsigned long long");
  static_assert(sizeof(mask_word_t) % sizeof(d_mask_word_t) == 0,
                "host-side mask-word size must be multiple of device-side "
                "mask-word size");

  cudaError_t e;

  cudaGetDeviceProperties(&cuda_conf.cuda_prop, 0);

  // copy graph to device
  cuda_conf.d_graph = make_cuda_graph(G);

  // sizing
  cuda_conf.warp_step = 1;  // per thread
  // cuda_conf.warp_step = cuda_conf.cuda_prop.warpSize;  // per warp
  cuda_conf.block_size = cuda_conf.warp_step * (1 << 0);
  cuda_conf.n_blocks = 1 << 10;
  cuda_conf.grid_size = cuda_conf.n_blocks * cuda_conf.block_size;
  cuda_conf.batch_size = cuda_conf.grid_size / cuda_conf.warp_step;
  auto bits_per_word = 8 * sizeof(mask_word_t);
  cuda_conf.mask_words = (G.num_nodes() + bits_per_word - 1) / bits_per_word;

  // print sizing info
  CUDA_LOG("> *** CUDA_BATCHED sizing ***\n");
  CUDA_LOG("block-size = %d\n", cuda_conf.block_size);
  CUDA_LOG("n. blocks  = %d\n", cuda_conf.n_blocks);
  CUDA_LOG("warp size  = %d\n", cuda_conf.cuda_prop.warpSize);
  CUDA_LOG("grid size  = %d\n", cuda_conf.grid_size);
  CUDA_LOG("batch size = %d\n", cuda_conf.batch_size);
  CUDA_LOG("g-mem size = %d\n",
           cuda_conf.grid_size * cuda_conf.mask_words * sizeof(mask_word_t));

  // allocate memory for result masks and sizes
  auto batch_mask_size =
      cuda_conf.batch_size * cuda_conf.mask_words * sizeof(mask_word_t);
  cuda_conf.res_masks = (mask_word_t *)malloc(batch_mask_size);
  e = cudaMalloc(&cuda_conf.d_res_masks, batch_mask_size);
  cuda_check(e, __FILE__, __LINE__);
  cuda_conf.res_sizes = (size_t *)malloc(cuda_conf.batch_size * sizeof(size_t));
  e = cudaMalloc(&cuda_conf.d_res_sizes, cuda_conf.batch_size * sizeof(size_t));
  cuda_check(e, __FILE__, __LINE__);

  // init rng
  cudaMalloc(&cuda_conf.d_rng_states,
             cuda_conf.batch_size * sizeof(curandState));
  cuda_check(e, __FILE__, __LINE__);

  kernel_rng_setup<<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
      cuda_conf.d_rng_states, seed, cuda_conf.warp_step);
  cuda_check(__FILE__, __LINE__);

#if CUDA_PROFILE
  profile = (cuda_profile *)malloc(cuda_conf.batch_size * sizeof(cuda_profile));
  cudaMalloc(&d_profile, cuda_conf.batch_size * sizeof(cuda_profile));
#endif
}

void cuda_init(const cuda_GraphTy &G, unsigned long long seed,
               im::independent_cascade_tag &&) {}

void cuda_fini(im::linear_threshold_tag &&) {
  // cleanup
  if (cuda_conf.res_masks) free(cuda_conf.res_masks);
  if (cuda_conf.d_res_masks) cudaFree(cuda_conf.d_res_masks);
  if (cuda_conf.res_sizes) free(cuda_conf.res_sizes);
  if (cuda_conf.d_res_sizes) cudaFree(cuda_conf.d_res_sizes);
  if (cuda_conf.d_rng_states) cudaFree(cuda_conf.d_rng_states);
  destroy_cuda_graph(cuda_conf.d_graph);
#if CUDA_PROFILE
  if (profile) free(profile);
  if (d_profile) cudaFree(d_profile);
#endif
}

void cuda_fini(im::independent_cascade_tag &&) {}

template <typename HostGraphTy>
__global__ void kernel_lt_per_thread(
    typename HostGraphTy::DestinationTy **index, size_t num_nodes,
    size_t d_mask_words, size_t warp_size, curandState *d_rng_states,
    d_mask_word_t *d_res_masks, size_t *d_res_sizes
#if CUDA_PROFILE
    ,
    cuda_profile *d_profile
#endif
) {
  using destination_type = typename HostGraphTy::DestinationTy;
  using vertex_type = typename HostGraphTy::vertex_type;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % warp_size == 0) {
    int wid = tid / warp_size;
    d_mask_word_t mask_word = 0;
    size_t res_size = 0;

    // init res memory
    auto d_res_mask = d_res_masks + wid * d_mask_words;
    memset(d_res_mask, 0, d_mask_words * sizeof(d_mask_word_t));

    // cache rng state
    auto rng_state = d_rng_states + wid;

    // select source node
    vertex_type src = curand(rng_state) % num_nodes;
    CUDA_LOG("> [kernel] root=%d\n", src);
    set_in_word(mask_word, src);
    write_mask_word(d_res_mask, src, mask_word);
    CUDA_LOG("> [kernel] word=%d @mask-pos=%d\n", mask_word, src >> 5);
    ++res_size;

    float threshold;
    destination_type *first, *last;
    vertex_type v;
#if CUDA_PROFILE
    clock_t rng_time = 0, el_time = 0, timer;
#endif
    while (src != num_nodes) {
      // rng
#if CUDA_PROFILE
      timer = clock();
#endif
      threshold = curand_uniform(rng_state);
#if CUDA_PROFILE
      rng_time += clock() - timer;
#endif

      // scan neighbor list
#if CUDA_PROFILE
      timer = clock();
#endif
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
            CUDA_LOG("> [kernel] marking v=%d\n", v);
            set_in_word(mask_word, src);
            write_mask_word(d_res_mask, v, mask_word);
            ++res_size;
          }
          break;
        }
      }
#if CUDA_PROFILE
      el_time += clock() - timer;
#endif
    }

    d_res_sizes[wid] = res_size;

#if CUDA_PROFILE
    d_profile[wid].rng = rng_time;
    d_profile[wid].el = el_time;
#endif
  }  // end if active warp
}

void batch_kernel(const cuda_GraphTy &G) {
  kernel_lt_per_thread<cuda_GraphTy>
      <<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
          cuda_conf.d_graph->d_index_, G.num_nodes(),
          cuda_conf.mask_words * (sizeof(mask_word_t) / sizeof(d_mask_word_t)),
          cuda_conf.warp_step, cuda_conf.d_rng_states, cuda_conf.d_res_masks,
          cuda_conf.d_res_sizes
#if CUDA_PROFILE
          ,
          d_profile
#endif
      );
  cuda_check(__FILE__, __LINE__);
}

void batch_kernel_profile() {
#if CUDA_PROFILE
  cudaMemcpy(profile, d_profile, cuda_conf.batch_size * sizeof(cuda_profile),
             cudaMemcpyDeviceToHost);
  std::vector<clock_t> rng_prof, el_prof;
  for (size_t i = 0; i < cuda_conf.batch_size; ++i) {
    if (profile[i].rng) rng_prof.push_back(profile[i].rng);
    if (profile[i].el) el_prof.push_back(profile[i].el);
  }
  std::sort(rng_prof.begin(), rng_prof.end());
  std::sort(el_prof.begin(), el_prof.end());
  printf("*** rng:\tmin =%10d\tmed =%10d\tmax =%10d\t(#zeros=%d)\n",
         rng_prof[0], rng_prof[rng_prof.size() / 2], rng_prof.back(),
         cuda_conf.batch_size - rng_prof.size());
  printf("*** el :\tmin =%10d\tmed =%10d\tmax =%10d\t(#zeros=%d)\n", el_prof[0],
         el_prof[el_prof.size() / 2], el_prof.back(),
         cuda_conf.batch_size - el_prof.size());
#endif
}

void batch_d2h() {
  cudaMemcpy(cuda_conf.res_masks, cuda_conf.d_res_masks,
             cuda_conf.batch_size * cuda_conf.mask_words * sizeof(mask_word_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cuda_conf.res_sizes, cuda_conf.d_res_sizes,
             cuda_conf.batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::linear_threshold_tag &&model_tag) {
  CUDA_LOG("> *** CudaGenerateRRRSets theta=%d ***\n", theta);

  using vertex_type = typename cuda_GraphTy::vertex_type;
#if CUDA_PROFILE
  std::chrono::nanoseconds kernel_time, copyback_time, postproc_time;
#endif

  cuda_res_t rrr_sets(theta);

  for (size_t bf = 0; bf < rrr_sets.size(); bf += cuda_conf.batch_size) {
    // execute a batch
#if CUDA_PROFILE
    printf("*** [kernel_lt_per_thread] profile #%d ***\n",
           bf / cuda_conf.batch_size);
    auto start = std::chrono::high_resolution_clock::now();
#endif
    batch_kernel(G);
#if CUDA_PROFILE
    // TODO wait
    kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
#endif

    batch_kernel_profile();

    // copy results back to host
#if CUDA_PROFILE
    start = std::chrono::high_resolution_clock::now();
#endif
    batch_d2h();
#if CUDA_PROFILE
    copyback_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
#endif

    // convert masks to results
#if CUDA_PROFILE
    start = std::chrono::high_resolution_clock::now();
#endif
    for (size_t i = 0; i < cuda_conf.batch_size && (bf + i) < rrr_sets.size();
         ++i) {
      auto &rrr_set = rrr_sets[bf + i];
      rrr_set.reserve(cuda_conf.res_sizes[i]);
      auto res_mask = cuda_conf.res_masks + (i * cuda_conf.mask_words);
      for (size_t j = 0; j < cuda_conf.mask_words; ++j) {
        vertex_type offset = sizeof(mask_word_t) * 8 * j;
        // scan a word from the res mask
        auto w = res_mask[j];
        while (w != 0) {
          auto delta = __builtin_ffsll(w);
          w >>= delta - 1;
          offset += delta - 1;
          auto v = vertex_type(offset);
          rrr_set.push_back(v);
          w >>= 1;
          ++offset;
        }
      }
#if CUDA_CHECK
      assert(rrr_set.size() == cuda_conf.res_sizes[i]);
#endif
    }
#if CUDA_PROFILE
    postproc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
#endif

#if CUDA_CHECK
    for (size_t i = 0; i < cuda_conf.batch_size && (bf + i) < rrr_sets.size();
         ++i)
      check_lt(rrr_sets[bf + i], G, bf + i);
#endif

      // print profiling
#if CUDA_PROFILE
    printf("*** [host] breakdown #%d ***\n", bf / cuda_conf.batch_size);
    std::cout << "*** kernel          : " << kernel_time.count() << " ns\n";
    std::cout << "*** copy-back       : " << copyback_time.count() << " ns\n";
    std::cout << "*** post-processing : " << postproc_time.count() << " ns\n";
#endif
  }

  // TODO check
  // check_lt(rrr_set, G, bf + i);

  return rrr_sets;
}  // namespace im

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::independent_cascade_tag &&model_tag) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace im
