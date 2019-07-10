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

using lt_generator_t = StreamingRRRGenerator<cuda_GraphTy, cuda_PRNGeneratorTy,
                                             ripples::linear_threshold_tag>;
using ic_generator_t = StreamingRRRGenerator<cuda_GraphTy, cuda_PRNGeneratorTy,
                                             ripples::independent_cascade_tag>;

struct ctx_t {
  ctx_t(lt_generator_t *ltg) : lt_generator(ltg), ic_generator(nullptr) {}
  ctx_t(ic_generator_t *icg) : lt_generator(nullptr), ic_generator(icg) {}

  lt_generator_t *lt_generator;
  ic_generator_t *ic_generator;
} *ctx{nullptr};

template <typename diff_model_tag>
void cuda_init_common(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r) {
  using generator_t =
      StreamingRRRGenerator<cuda_GraphTy, cuda_PRNGeneratorTy, diff_model_tag>;
  auto CFG = configuration();
  auto workers = CFG.streaming_workers;
  auto gpu_workers = CFG.streaming_gpu_workers;
  assert(gpu_workers <= workers);
  ctx = new ctx_t(new generator_t(G, r, workers - gpu_workers, gpu_workers));
}

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::independent_cascade_tag &&) {
  cuda_init_common<ripples::independent_cascade_tag>(G, r);
}

void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &r,
               ripples::linear_threshold_tag &&) {
  cuda_init_common<ripples::linear_threshold_tag>(G, r);
}

void cuda_fini() {
  if(ctx->ic_generator) delete ctx->ic_generator;
  if(ctx->lt_generator) delete ctx->lt_generator;
}

cuda_res_t CudaGenerateRRRSets(size_t theta) {
  cuda_res_t res;
  if (ctx->ic_generator) {
    assert(!ctx->lt_generator);
    res = ctx->ic_generator->generate(theta);
  } else {
    assert(!ctx->ic_generator);
    res = ctx->lt_generator->generate(theta);
  }
  return res;
}

}  // namespace ripples
