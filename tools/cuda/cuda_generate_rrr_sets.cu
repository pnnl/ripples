//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <curand_kernel.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "im/cuda/cuda_generate_rrr_sets.h"
#include "im/cuda/cuda_graph.cuh"
#include "im/cuda/cuda_utils.h"

namespace im {

using mask_word_t = typename cuda_GraphTy::vertex_type;
constexpr size_t MAX_SET_SIZE = 32;

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
  mask_word_t *res_masks = nullptr, *d_res_masks = nullptr;
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
enum breakdown_tag { KERNEL, COPY, POSTPROC };
std::unordered_map<breakdown_tag, std::vector<std::chrono::nanoseconds>>
    profile_breakdown;

void print_profile(breakdown_tag tag, const std::string &label) {
  std::sort(profile_breakdown[tag].begin(), profile_breakdown[tag].end());
  auto &sample(profile_breakdown[tag]);
  std::chrono::microseconds tot{0};
  for (auto &x : sample)
    tot += std::chrono::duration_cast<std::chrono::microseconds>(x);
  std::cout << "*** tag: " << label << "\n*** "
            << "cnt=" << sample.size() << "\tmin=" << sample[0].count()
            << "\tmed=" << sample[sample.size() / 2].count()
            << "\tmax=" << sample.back().count() << "\ttot=" << tot.count()
            << "us\n";
}
#endif

void cuda_init(const cuda_GraphTy &G, unsigned long long seed,
               im::linear_threshold_tag &&model_tag) {
  cudaError_t e;
  cudaGetDeviceProperties(&cuda_conf.cuda_prop, 0);

  // copy graph to device
  cuda_conf.d_graph = make_cuda_graph(G);

  // sizing
  // cuda_conf.warp_step = 1;  // per thread
  cuda_conf.warp_step = cuda_conf.cuda_prop.warpSize;  // per warp
  cuda_conf.block_size = cuda_conf.warp_step * (1 << 0);
  cuda_conf.n_blocks = 1 << 13;
  cuda_conf.grid_size = cuda_conf.n_blocks * cuda_conf.block_size;
  cuda_conf.batch_size = cuda_conf.grid_size / cuda_conf.warp_step;
  cuda_conf.mask_words = MAX_SET_SIZE;

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
// print profiling
#if CUDA_PROFILE
  printf("*** profiling summary (time unit: ns) ***\n");

  // print sizing info
  printf("> *** CUDA_BATCHED sizing ***\n");
  printf("block-size = %d\n", cuda_conf.block_size);
  printf("n. blocks  = %d\n", cuda_conf.n_blocks);
  printf("warp size  = %d\n", cuda_conf.cuda_prop.warpSize);
  printf("grid size  = %d\n", cuda_conf.grid_size);
  printf("batch size = %d\n", cuda_conf.batch_size);
  printf("g-mem size = %d\n",
         cuda_conf.grid_size * cuda_conf.mask_words * sizeof(mask_word_t));

  print_profile(breakdown_tag::KERNEL, "kernel");
  print_profile(breakdown_tag::COPY, "device-to-host copy");
  print_profile(breakdown_tag::POSTPROC, "post-processing");
#endif

  // cleanup
  if (cuda_conf.res_masks) free(cuda_conf.res_masks);
  if (cuda_conf.d_res_masks) cudaFree(cuda_conf.d_res_masks);
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
    size_t warp_size, curandState *d_rng_states, mask_word_t *d_res_masks
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
    size_t res_size = 0;

    // init res memory
    auto d_res_mask = d_res_masks + wid * MAX_SET_SIZE;
    memset(d_res_mask, 0, MAX_SET_SIZE * sizeof(mask_word_t));

    // cache rng state
    auto rng_state = d_rng_states + wid;

    // select source node
    vertex_type src = curand(rng_state) % num_nodes;
    CUDA_LOG("> [kernel] root=%d\n", src);
    d_res_mask[res_size++] = src;

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
          // found candidate vertex
          v = first->vertex;

          // insert if not visited
          size_t i = 0;
          while (i < res_size && d_res_mask[i] != v) ++i;
          if (i == res_size) {
            CUDA_LOG("> [kernel] marking v=%d\n", v);
            src = v;
            d_res_mask[res_size++] = v;
          }
          break;
        }
      }
#if CUDA_PROFILE
      el_time += clock() - timer;
#endif
    }

    // mark end-of-set
    if (res_size < MAX_SET_SIZE) d_res_mask[res_size] = num_nodes;

#if CUDA_PROFILE
    d_profile[wid].rng = rng_time;
    d_profile[wid].el = el_time;
#endif
  }  // end if active warp
}  // namespace im

void batch_kernel(const cuda_GraphTy &G) {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  kernel_lt_per_thread<cuda_GraphTy>
      <<<cuda_conf.n_blocks, cuda_conf.block_size>>>(
          cuda_conf.d_graph->d_index_, G.num_nodes(), cuda_conf.warp_step,
          cuda_conf.d_rng_states, cuda_conf.d_res_masks
#if CUDA_PROFILE
          ,
          d_profile
#endif
      );
  cuda_check(__FILE__, __LINE__);

#if CUDA_PROFILE
  cudaDeviceSynchronize();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  profile_breakdown[breakdown_tag::KERNEL].push_back(elapsed);
  // batch_kernel_profile();
#endif
}

#if CUDA_PROFILE
void batch_kernel_profile() {
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
}
#endif

void batch_d2h() {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif
  cudaMemcpy(cuda_conf.res_masks, cuda_conf.d_res_masks,
             cuda_conf.batch_size * cuda_conf.mask_words * sizeof(mask_word_t),
             cudaMemcpyDeviceToHost);
#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  profile_breakdown[breakdown_tag::COPY].push_back(elapsed);
#endif
}

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::linear_threshold_tag &&model_tag) {
  CUDA_LOG("> *** CudaGenerateRRRSets theta=%d ***\n", theta);

  using vertex_type = typename cuda_GraphTy::vertex_type;
  cuda_res_t rrr_sets(theta);

  for (size_t bf = 0; bf < rrr_sets.size(); bf += cuda_conf.batch_size) {
    // execute a batch
    batch_kernel(G);

    // copy results back to host
    batch_d2h();

    // convert masks to results
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    for (size_t i = 0; i < cuda_conf.batch_size && (bf + i) < rrr_sets.size();
         ++i) {
      auto &rrr_set = rrr_sets[bf + i];
      rrr_set.reserve(MAX_SET_SIZE);
      auto res_mask = cuda_conf.res_masks + (i * cuda_conf.mask_words);
      for (size_t j = 0;
           j < cuda_conf.mask_words && res_mask[j] != G.num_nodes(); ++j) {
        rrr_set.push_back(res_mask[j]);
      }

      if (rrr_set.size() == MAX_SET_SIZE) {
        fprintf(stderr, "> an RRR set hit the maximum size %d\n", MAX_SET_SIZE);
        exit(1);
      }
    }
#if CUDA_PROFILE
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    profile_breakdown[breakdown_tag::POSTPROC].push_back(elapsed);
#endif

#if CUDA_CHECK
    for (size_t i = 0; i < cuda_conf.batch_size && (bf + i) < rrr_sets.size();
         ++i)
      check_lt(rrr_sets[bf + i], G, bf + i);
#endif
  }

  return rrr_sets;
}  // namespace im

cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::independent_cascade_tag &&model_tag) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace im
