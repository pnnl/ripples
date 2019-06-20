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

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/generate_rrr_sets.h"

#if CUDA_PROFILE
#include <atomic>
#include <chrono>
#endif

#define CUDA_PARFOR

namespace ripples {

#if CUDA_PROFILE
enum breakdown_tag { OVERALL, KERNEL, COPY, BUILD };
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
  size_t cpu_threads = 0, gpu_streams = 0;

#if CUDA_PROFILE
  size_t num_sets{0};
  std::atomic<size_t> num_exceedings{0};
  using mt_sample_t = std::vector<std::vector<std::chrono::nanoseconds>>;
  using breakdown_t = std::unordered_map<breakdown_tag, mt_sample_t>;
  breakdown_t profile_breakdown;
#endif
} ctx;

size_t cuda_num_total_threads() {
  return ctx.cpu_threads + ctx.gpu_streams * ctx.max_batch_size;
}

size_t cuda_num_cpu_threads() { return ctx.cpu_threads; }

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::linear_threshold_tag &&model_tag) {
  // TODO
  // cudaGetDeviceProperties(&ctx.cuda_prop, 0);

  // copy graph to device
  ctx.graph = &G;

  // sizing
  ctx.warp_step = 1;  // distance between active threads in a warp
  // ctx.warp_step = ctx.cuda_prop.warpSize;
  ctx.block_size = ctx.warp_step * (1 << 0);
  ctx.n_blocks = 1 << 13;
  ctx.grid_size = ctx.n_blocks * ctx.block_size;
  ctx.max_batch_size = ctx.grid_size / ctx.warp_step;
  ctx.mask_words = 8;
  auto mask_size = ctx.mask_words * sizeof(mask_word_t);

#pragma omp single
  ctx.cpu_threads = omp_get_max_threads();

#ifdef CUDA_PARFOR
  ctx.gpu_streams = ctx.cpu_threads;
#else
  ctx.gpu_streams = 1;
#endif
  // set up device-side graph
  cuda_graph_init(G);

  // allocate host-side memory for result masks
  ctx.res_masks =
      (mask_word_t **)malloc(ctx.gpu_streams * sizeof(mask_word_t *));
  for (size_t i = 0; i < ctx.gpu_streams; ++i)
    ctx.res_masks[i] = (mask_word_t *)malloc(ctx.max_batch_size * mask_size);

  // allocate device-memory for resulst masks
  ctx.d_res_masks =
      (mask_word_t **)malloc(ctx.gpu_streams * sizeof(mask_word_t *));
  for (size_t i = 0; i < ctx.gpu_streams; ++i) {
    cuda_malloc((void **)&ctx.d_res_masks[i], ctx.max_batch_size * mask_size);
  }

  // set up device-side RNGs
  ctx.d_trng_states = (cuda_PRNGeneratorTy **)malloc(
      ctx.gpu_streams * sizeof(cuda_PRNGeneratorTy *));
  for (size_t i = 0; i < ctx.gpu_streams; ++i) {
    cuda_malloc((void **)&ctx.d_trng_states[i],
                ctx.max_batch_size * sizeof(cuda_PRNGeneratorTy));
    cuda_rng_setup(ctx.n_blocks, ctx.block_size, ctx.d_trng_states[i], i, r,
                   ctx.warp_step, ctx.max_batch_size, cuda_num_total_threads(),
                   ctx.cpu_threads);
  }

#if CUDA_PROFILE
  ctx.profile_breakdown[breakdown_tag::OVERALL] =
      ctx_t::mt_sample_t(ctx.cpu_threads);
  ctx.profile_breakdown[breakdown_tag::KERNEL] =
      ctx_t::mt_sample_t(ctx.cpu_threads);
  ctx.profile_breakdown[breakdown_tag::COPY] =
      ctx_t::mt_sample_t(ctx.cpu_threads);
  ctx.profile_breakdown[breakdown_tag::BUILD] =
      ctx_t::mt_sample_t(ctx.cpu_threads);
#endif
}

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::independent_cascade_tag &&) {}

void cuda_fini(ripples::linear_threshold_tag &&) {
// print profiling
#if CUDA_PROFILE
  auto logst = spdlog::stdout_color_st("CUDA-profile");

  // print sizing info
  logst->info("block-size     = {}", ctx.block_size);
  logst->info("n. blocks      = {}", ctx.n_blocks);
  // TODO
  // logst->info("warp size    = {}", ctx.cuda_prop.warpSize);
  logst->info("grid size      = {}", ctx.grid_size);
  logst->info("batch size     = {}", ctx.max_batch_size);
  logst->info("n. cpu threads = {}", ctx.cpu_threads);
  logst->info("n. gpu streams = {}", ctx.gpu_streams);

  // mem sizing
  logst->info("g-mem size     = {}", ctx.gpu_streams * ctx.grid_size *
                                         ctx.mask_words * sizeof(mask_word_t));

  print_profile_breakdown(logst, ctx.profile_breakdown[breakdown_tag::OVERALL],
                          "overall");
  print_profile_breakdown(logst, ctx.profile_breakdown[breakdown_tag::KERNEL],
                          "kernel");
  print_profile_breakdown(logst, ctx.profile_breakdown[breakdown_tag::COPY],
                          "device-to-host copy");
  print_profile_breakdown(logst, ctx.profile_breakdown[breakdown_tag::BUILD],
                          "build");

  auto ne = ctx.num_exceedings.load();
  logst->info("exceeding sets = {}/{} ({})", ne, ctx.num_sets,
              (float)ne / ctx.num_sets);
#endif

  // finalize streams and free memory
  assert(ctx.res_masks);
  for (size_t i = 0; i < ctx.gpu_streams; ++i) free(ctx.res_masks[i]);
  free(ctx.res_masks);
  assert(ctx.d_res_masks);
  for (size_t i = 0; i < ctx.gpu_streams; ++i) cuda_free(ctx.d_res_masks[i]);
  free(ctx.d_res_masks);
  assert(ctx.d_trng_states);
  for (size_t i = 0; i < ctx.gpu_streams; ++i) cuda_free(ctx.d_trng_states[i]);
  free(ctx.d_trng_states);

  // cleanup
  cuda_graph_fini();
}

void cuda_fini(ripples::independent_cascade_tag &&) {}

void batch_kernel(size_t rank, size_t batch_size) {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  cuda_lt_kernel(ctx.n_blocks, ctx.block_size, batch_size,
                 ctx.graph->num_nodes(), ctx.warp_step, ctx.d_trng_states[rank],
                 ctx.d_res_masks[rank], ctx.mask_words);

#if CUDA_PROFILE
  // un-comment the following line to measure effective kernel run-time (rather
  // than launch-time)
  // cudaDeviceSynchronize();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::KERNEL][rank].push_back(elapsed);
#endif
}

void batch_d2h(size_t rank, size_t batch_size) {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif
  cuda_d2h(ctx.res_masks[rank], ctx.d_res_masks[rank],
           batch_size * ctx.mask_words * sizeof(mask_word_t));

#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::COPY][rank].push_back(elapsed);
#endif
}

template <typename diff_model_tag>
void batch_build(size_t rank, cuda_res_t &rrr_sets,
                 cuda_PRNGeneratorsTy &generators, size_t bf, size_t batch_size,
                 diff_model_tag &&model_tag) {
  // translate

#ifdef CUDA_PARFOR

#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  for (size_t i = 0; i < batch_size; ++i) {
    auto &rrr_set = rrr_sets[bf + i];
    rrr_set.reserve(ctx.mask_words);
    auto res_mask = ctx.res_masks[rank] + (i * ctx.mask_words);
    for (size_t j = 0;
         j < ctx.mask_words && res_mask[j] != ctx.graph->num_nodes(); ++j) {
      rrr_set.push_back(res_mask[j]);
    }

    if (rrr_set.size() == ctx.mask_words) {
      ++ctx.num_exceedings;
      auto root = rrr_set[0];
      rrr_set.clear();
      AddRRRSet(*ctx.graph, root, generators[rank], rrr_set,
                std::forward<diff_model_tag>(model_tag));
    }

    std::stable_sort(rrr_set.begin(), rrr_set.end());
  }

#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::BUILD][rank].push_back(elapsed);
#endif

#else  // CUDA_PARFOR
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
  // std::chrono::nanoseconds m_elapsed{0};
#endif

#pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < batch_size; ++i) {
    auto &rrr_set = rrr_sets[bf + i];
    rrr_set.reserve(ctx.mask_words);
    auto res_mask = ctx.res_masks[rank] + (i * ctx.mask_words);
    for (size_t j = 0;
         j < ctx.mask_words && res_mask[j] != ctx.graph->num_nodes(); ++j) {
      rrr_set.push_back(res_mask[j]);
    }

    if (rrr_set.size() == ctx.mask_words) {
      ++ctx.num_exceedings;
      auto root = rrr_set[0];
      rrr_set.clear();
      AddRRRSet(*ctx.graph, root, generators[omp_get_thread_num()], rrr_set,
                std::forward<diff_model_tag>(model_tag));
    }

    std::stable_sort(rrr_set.begin(), rrr_set.end());
  }

#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::BUILD][rank].push_back(elapsed);
#endif
#endif  // CUDA_PARFOR

#if CUDA_CHECK
  size_t i = 0;
  for (auto &rrr_set : rrr_sets) check_lt(rrr_set, *ctx.graph, bf + i++);
#endif
}

cuda_res_t CudaGenerateRRRSets(size_t theta, cuda_PRNGeneratorsTy &generators,
                               ripples::linear_threshold_tag &&model_tag) {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  cuda_res_t rrr_sets(theta);

  auto num_batches = (theta + ctx.max_batch_size - 1) / ctx.max_batch_size;

#ifdef CUDA_PARFOR
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

#else
  auto batch_first = 0;
  auto batch_size = std::min(rrr_sets.size(), ctx.max_batch_size);
  batch_kernel(0, batch_size);

  for (size_t bi = 1; bi < num_batches; ++bi) {
    batch_d2h(0, batch_size);

    auto next_batch_first = bi * ctx.max_batch_size;
    auto next_batch_size =
        std::min(rrr_sets.size() - next_batch_first, ctx.max_batch_size);

    batch_kernel(0, next_batch_size);

    batch_build(0, rrr_sets, generators, batch_first, batch_size,
                ripples::linear_threshold_tag{});

    batch_first = next_batch_first;
    batch_size = next_batch_size;
  }

  batch_d2h(0, batch_size);
  batch_build(0, rrr_sets, generators, batch_first, batch_size,
              ripples::linear_threshold_tag{});
#endif  // CUDA_PARFOR

#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::OVERALL][0].push_back(elapsed);
  ctx.num_sets += theta;
#endif

  return rrr_sets;
}

cuda_res_t CudaGenerateRRRSets(size_t theta, cuda_PRNGeneratorsTy &generators,
                               ripples::independent_cascade_tag &&model_tag) {
  assert(false);
  return cuda_res_t{};
}

}  // namespace ripples
