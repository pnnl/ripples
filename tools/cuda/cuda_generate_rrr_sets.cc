//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <iostream>
#include <unordered_map>

#include <omp.h>

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/generate_rrr_sets.h"
#include "ripples/cuda/cuda_generate_rrr_sets.h"

#if CUDA_PROFILE
#include <atomic>
#include <chrono>
#endif

namespace ripples {

#if CUDA_PROFILE
enum breakdown_tag { OVERALL, KERNEL, COPY, TRANSLATE_ALLOC, TRANSLATE_BUILD };
#endif

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
struct ctx_t {
  // TODO
  // cudaDeviceProp cuda_prop;

  const cuda_GraphTy *graph = nullptr;

  // host-side buffers
  mask_word_t **res_masks = nullptr;

  // device-side buffers
  mask_word_t **d_res_masks = nullptr;
  cuda_PRNGeneratorTy **d_trng_states;

  // sizing
  size_t grid_size = 0, block_size = 0, n_blocks = 0;
  size_t warp_step = 0;       // 1: per-thread, warp-size: per-warp
  size_t max_batch_size = 0;  // walks per batch
  size_t mask_words = 0;
  size_t cpu_threads = 0;

#if CUDA_PROFILE
  size_t num_sets{0};
  std::atomic<size_t> num_exceedings{0};
  std::unordered_map<breakdown_tag, std::vector<std::chrono::nanoseconds>>
      profile_breakdown;
#endif
} ctx;

#if CUDA_PROFILE
void print_profile(breakdown_tag tag, const std::string &label) {
  if (!ctx.profile_breakdown[tag].empty()) {
    std::sort(ctx.profile_breakdown[tag].begin(),
              ctx.profile_breakdown[tag].end());
    auto &sample(ctx.profile_breakdown[tag]);
    std::chrono::microseconds tot{0};
    for (auto &x : sample)
      tot += std::chrono::duration_cast<std::chrono::microseconds>(x);
    std::cout << "*** tag: " << label << "\n*** "
              << "cnt=" << sample.size() << "\tmin=" << sample[0].count()
              << "\tmed=" << sample[sample.size() / 2].count()
              << "\tmax=" << sample.back().count()
              << "\ttot(us)=" << tot.count() << std::endl;
  } else
    std::cout << "*** tag: " << label << " N/A\n";
}
#endif

size_t cuda_num_total_threads() {
  return ctx.cpu_threads + ctx.cpu_threads * ctx.max_batch_size;
}

size_t cuda_num_cpu_threads() { return ctx.cpu_threads; }

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::linear_threshold_tag &&model_tag) {
  // TODO
  // cudaGetDeviceProperties(&ctx.cuda_prop, 0);

  // copy graph to device
  ctx.graph = &G;

  // sizing
  ctx.warp_step = 1;  // per block
  // ctx.warp_step = ctx.cuda_prop.warpSize;  // per warp
  ctx.block_size = ctx.warp_step * (1 << 0);
  ctx.n_blocks = 1 << 13;
  ctx.grid_size = ctx.n_blocks * ctx.block_size;
  ctx.max_batch_size = ctx.grid_size / ctx.warp_step;
  ctx.mask_words = MAX_SET_SIZE;
  auto mask_size = ctx.mask_words * sizeof(mask_word_t);

#pragma omp single
  ctx.cpu_threads = omp_get_max_threads();

  // print sizing info
  CUDA_LOG("> *** CUDA_BATCHED sizing ***\n");
  CUDA_LOG("block-size = %d\n", ctx.block_size);
  CUDA_LOG("n. blocks  = %d\n", ctx.n_blocks);

  // TODO
  // CUDA_LOG("warp size  = %d\n", ctx.cuda_prop.warpSize);

  CUDA_LOG("grid size  = %d\n", ctx.grid_size);
  CUDA_LOG("batch size = %d\n", ctx.max_batch_size);

  // TODO move
  CUDA_LOG("g-mem size = %d\n", ctx.cpu_threads * ctx.grid_size *
                                    ctx.mask_words * sizeof(mask_word_t));

  // set up device-side graph
  cuda_graph_init(G);

  // allocate host-side memory for result masks
  ctx.res_masks =
      (mask_word_t **)malloc(ctx.cpu_threads * sizeof(mask_word_t *));
  for (size_t i = 0; i < ctx.cpu_threads; ++i)
    ctx.res_masks[i] = (mask_word_t *)malloc(ctx.max_batch_size * mask_size);

  // allocate device-memory for resulst masks
  ctx.d_res_masks =
      (mask_word_t **)malloc(ctx.cpu_threads * sizeof(mask_word_t *));
  for (size_t i = 0; i < ctx.cpu_threads; ++i) {
    cuda_malloc((void **)&ctx.d_res_masks[i], ctx.max_batch_size * mask_size);
  }

  // set up device-side RNGs
  ctx.d_trng_states = (cuda_PRNGeneratorTy **)malloc(
      ctx.cpu_threads * sizeof(cuda_PRNGeneratorTy *));
  for (size_t i = 0; i < ctx.cpu_threads; ++i) {
    cuda_malloc((void **)&ctx.d_trng_states[i],
                ctx.max_batch_size * sizeof(cuda_PRNGeneratorTy));
    cuda_rng_setup(ctx.n_blocks, ctx.block_size, ctx.d_trng_states[i], i, r,
                   ctx.warp_step, ctx.max_batch_size, cuda_num_total_threads(),
                   ctx.cpu_threads);
  }
}

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::independent_cascade_tag &&) {}

void cuda_fini(ripples::linear_threshold_tag &&) {
// print profiling
#if CUDA_PROFILE
  printf("*** profiling summary (time unit: ns) ***\n");

  // print sizing info
  printf("> *** CUDA_BATCHED sizing ***\n");
  printf("block-size     = %d\n", ctx.block_size);
  printf("n. blocks      = %d\n", ctx.n_blocks);
  // TODO
  // printf("warp size    = %d\n", ctx.cuda_prop.warpSize);
  printf("grid size      = %d\n", ctx.grid_size);
  printf("batch size     = %d\n", ctx.max_batch_size);
  printf("n. cpu threads = %d\n", ctx.cpu_threads);
  printf("g-mem size     = %d\n", ctx.cpu_threads * ctx.grid_size *
                                      ctx.mask_words * sizeof(mask_word_t));

  print_profile(breakdown_tag::OVERALL, "overall");
  print_profile(breakdown_tag::KERNEL, "kernel");
  print_profile(breakdown_tag::COPY, "device-to-host copy");
  print_profile(breakdown_tag::TRANSLATE_BUILD, "translate > build");
  print_profile(breakdown_tag::TRANSLATE_ALLOC, "translate > build > alloc");

  auto ne = ctx.num_exceedings.load();
  printf("exceeding sets = %d/%d (%f)\n", ne, ctx.num_sets,
         (float)ne / ctx.num_sets);
#endif

  // finalize streams and free memory
  assert(ctx.res_masks);
  for (size_t i = 0; i < ctx.cpu_threads; ++i) free(ctx.res_masks[i]);
  free(ctx.res_masks);
  assert(ctx.d_res_masks);
  for (size_t i = 0; i < ctx.cpu_threads; ++i) cuda_free(ctx.d_res_masks[i]);
  free(ctx.d_res_masks);
  assert(ctx.d_trng_states);
  for (size_t i = 0; i < ctx.cpu_threads; ++i) cuda_free(ctx.d_trng_states[i]);
  free(ctx.d_trng_states);

  // cleanup
  cuda_graph_fini();
}

void cuda_fini(ripples::independent_cascade_tag &&) {}

void batch_kernel(size_t rank, size_t batch_size) {
  CUDA_LOG("> [batch_kernel] size=%d\n", batch_size);

#if CUDA_PROFILE
  // TODO
//  auto start = std::chrono::high_resolution_clock::now();
#endif

  cuda_lt_kernel(ctx.n_blocks, ctx.block_size, batch_size,
                 ctx.graph->num_nodes(), ctx.warp_step, ctx.d_trng_states[rank],
                 ctx.d_res_masks[rank]);

#if CUDA_PROFILE
  // un-comment the following line to measure effective kernel run-time (rather
  // than launch-time)
  // cudaDeviceSynchronize();
  // TODO
//  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
//      std::chrono::high_resolution_clock::now() - start);
//  ctx.profile_breakdown[breakdown_tag::KERNEL].push_back(elapsed);
#endif
}

void batch_d2h(size_t rank, size_t batch_size) {
  CUDA_LOG("> [batch_d2h] size=%d\n", batch_size);

#if CUDA_PROFILE
  // TODO
  // auto start = std::chrono::high_resolution_clock::now();
#endif
  cuda_d2h(ctx.res_masks[rank], ctx.d_res_masks[rank],
           batch_size * ctx.mask_words * sizeof(mask_word_t));

#if CUDA_PROFILE
  // TODO
//  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
//      std::chrono::high_resolution_clock::now() - start);
//  ctx.profile_breakdown[breakdown_tag::COPY].push_back(elapsed);
#endif
}

template<typename diff_model_tag>
void batch_build(size_t rank, cuda_res_t &rrr_sets,
                 cuda_PRNGeneratorsTy &generators, size_t bf,
                 size_t batch_size, diff_model_tag &&model_tag) {
  // translate
  CUDA_LOG("> [batch_build] size=%d first=%d\n", batch_size, bf);
#if CUDA_PROFILE
  // TODO
//  auto start = std::chrono::high_resolution_clock::now();
//  std::chrono::nanoseconds m_elapsed{0};
#endif

  for (size_t i = 0; i < batch_size; ++i) {
    auto &rrr_set = rrr_sets[bf + i];
#if CUDA_PROFILE
    // TODO
    // auto m_start = std::chrono::high_resolution_clock::now();
#endif
    rrr_set.reserve(MAX_SET_SIZE);
#if CUDA_PROFILE
    // TODO
    // m_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
    // std::chrono::high_resolution_clock::now() - m_start);
#endif
    auto res_mask = ctx.res_masks[rank] + (i * ctx.mask_words);
    for (size_t j = 0;
         j < ctx.mask_words && res_mask[j] != ctx.graph->num_nodes(); ++j) {
      rrr_set.push_back(res_mask[j]);
    }

#if CUDA_CHECK
    check_lt(rrr_set, *ctx.graph, bf + i);
#endif

    if (rrr_set.size() == MAX_SET_SIZE) {
      ++ctx.num_exceedings;
      auto root = rrr_set[0];
      rrr_set.clear();
      AddRRRSet(*ctx.graph, root, generators[rank], rrr_set,
                std::forward<diff_model_tag>(model_tag));
    }

    std::stable_sort(rrr_set.begin(), rrr_set.end());
  }

#if CUDA_PROFILE
  // TODO
//  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
//      std::chrono::high_resolution_clock::now() - start);
//  ctx.profile_breakdown[breakdown_tag::TRANSLATE_BUILD].push_back(elapsed);
//  ctx.profile_breakdown[breakdown_tag::TRANSLATE_ALLOC].push_back(m_elapsed);
#endif
}

cuda_res_t CudaGenerateRRRSets(size_t theta, cuda_PRNGeneratorsTy &generators,
                               ripples::linear_threshold_tag &&model_tag) {
  CUDA_LOG("> *** CudaGenerateRRRSets theta=%d ***\n", theta);

#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  cuda_res_t rrr_sets(theta);

  auto num_batches =
      (rrr_sets.size() + ctx.max_batch_size - 1) / ctx.max_batch_size;
  printf("> [CudaGenerateRRRSets] BEGIN-phase batches=%d\n", num_batches);

#pragma omp parallel for schedule(guided)
  for (size_t bi = 0; bi < num_batches; ++bi) {
    auto batch_first = bi * ctx.max_batch_size;
    auto batch_size =
        std::min(rrr_sets.size() - batch_first, ctx.max_batch_size);
    auto rank = omp_get_thread_num();

    batch_kernel(rank, batch_size);
    batch_d2h(rank, batch_size);
    batch_build(rank, rrr_sets, generators, batch_first, batch_size,
    		    ripples::linear_threshold_tag{});
  }

#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::OVERALL].push_back(elapsed);
  ctx.num_sets += theta;
#endif

  return rrr_sets;
}  // namespace ripples

cuda_res_t CudaGenerateRRRSets(size_t theta, cuda_PRNGeneratorsTy &generators,
                               ripples::independent_cascade_tag &&model_tag) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace ripples
