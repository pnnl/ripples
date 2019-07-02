//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <iostream>
#include <unordered_map>

#include <omp.h>

#include "spdlog/spdlog.h"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/imm.h"
#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/streaming_rrr_generator_omp.h"

#if CUDA_PROFILE
#include <atomic>
#include <chrono>
#endif

namespace ripples {

#if 0
#if CUDA_PROFILE
enum breakdown_tag { OVERALL, KERNEL, COPY, BUILD };
#endif

struct ctx_t {
#if CUDA_PROFILE
  size_t num_sets{0};
  std::atomic<size_t> num_exceedings{0};
  using mt_sample_t = std::vector<std::vector<std::chrono::nanoseconds>>;
  using breakdown_t = std::unordered_map<breakdown_tag, mt_sample_t>;
  breakdown_t profile_breakdown;
#endif
} ctx;

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
}  // namespace ripples

void cuda_fini_common() {
// print profiling
#if CUDA_PROFILE
  auto logst = spdlog::stdout_color_st("CUDA-profile");

  // print sizing info
  logst->info("block-size     = {}", ctx.block_size);
  logst->info("n. blocks      = {}", ctx.n_blocks);
  // TODO
  logst->info("warp size      = {}", cuda_warp_size());
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
}

template <typename diff_model_tag>
void batch_kernel(size_t rank, size_t batch_size, diff_model_tag &&) {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

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

#define CUDA_PARFOR

#ifdef CUDA_PARFOR

#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  for (size_t i = 0; i < batch_size; ++i) {
    auto &rrr_set = rrr_sets[bf + i];
    rrr_set.reserve(ctx.mask_words);
    auto res_mask = ctx.res_masks[rank] + (i * ctx.mask_words);
    if (res_mask[0] != ctx.graph->num_nodes()) {
      // valid walk
      for (size_t j = 0;
           j < ctx.mask_words && res_mask[j] != ctx.graph->num_nodes(); ++j) {
        rrr_set.push_back(res_mask[j]);
      }
    } else {
      // invalid walk
      ++ctx.num_exceedings;
      auto root = res_mask[1];
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

template <typename diff_model_t>
cuda_res_t CudaGenerateRRRSets_common(size_t theta,
                                      cuda_PRNGeneratorsTy &generators,
                                      diff_model_t &&model_tag) {
#if CUDA_PROFILE
  auto start = std::chrono::high_resolution_clock::now();
#endif

#if CUDA_PROFILE
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ctx.profile_breakdown[breakdown_tag::OVERALL][0].push_back(elapsed);
  ctx.num_sets += theta;
#endif

  return rrr_sets;
}
#endif

//
//
//
//
//
//

using generator_t = StreamingRRRGenerator<cuda_GraphTy, cuda_PRNGeneratorTy>;

struct {
  generator_t *rrr_generator;
} ctx;

void cuda_init_common(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r) {
  auto CFG = configuration();
  auto workers = CFG.streaming_workers;
  auto gpu_workers = CFG.streaming_gpu_workers;
  assert(gpu_workers <= workers);
  ctx.rrr_generator = new generator_t(G, r, workers - gpu_workers, gpu_workers);
}

void cuda_fini_common() {
  delete reinterpret_cast<generator_t *>(ctx.rrr_generator);
}

template<typename diff_model_tag>
cuda_res_t CudaGenerateRRRSets_common(size_t theta, diff_model_tag &&m) {
  return reinterpret_cast<generator_t *>(ctx.rrr_generator)
      ->generate(theta, std::forward<diff_model_tag>(m));
}

// dispatchers
void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::linear_threshold_tag &&model_tag) {
  cuda_init_common(G, r);
}

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::independent_cascade_tag &&model_tag) {
  cuda_init_common(G, r);
}

void cuda_fini(ripples::linear_threshold_tag &&model_tag) {
  cuda_fini_common();
}

void cuda_fini(ripples::independent_cascade_tag &&model_tag) {
  cuda_fini_common();
}

cuda_res_t CudaGenerateRRRSets(size_t theta,
                               ripples::linear_threshold_tag &&m) {
  return CudaGenerateRRRSets_common(
      theta, std::forward<ripples::linear_threshold_tag>(m));
}

cuda_res_t CudaGenerateRRRSets(size_t theta,
                               ripples::independent_cascade_tag &&m) {
  return CudaGenerateRRRSets_common(
      theta, std::forward<ripples::independent_cascade_tag>(m));
}

}  // namespace ripples
