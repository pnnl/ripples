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

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/streaming_rrr_generator_omp.h"
#include "ripples/imm.h"

#if CUDA_PROFILE
#include <atomic>
#include <chrono>
#endif

namespace ripples {

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

template <typename diff_model_tag>
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
