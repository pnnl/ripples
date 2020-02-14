//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_CUDA_CUDA_GENERATE_RRR_SETS_H
#define RIPPLES_CUDA_CUDA_GENERATE_RRR_SETS_H

#include <cstddef>
#include <vector>
#include <utility>

#include <cuda_runtime.h>

#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/diffusion_simulation.h"
#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/graph.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#define CUDA_CHECK 0
#define CUDA_PROFILE 0

namespace ripples {

//
// host-host API
//

// forward declarations to enable separate compilation
template <typename GraphTy>
using cuda_res_t = std::vector<std::vector<typename GraphTy::vertex_type>>;
using cuda_PRNGeneratorTy = trng::lcg64;
using cuda_PRNGeneratorsTy = std::vector<cuda_PRNGeneratorTy>;

constexpr size_t CUDA_WALK_SIZE = 8;

//
// host-device API
//
using mask_word_t = int; // TODO: vertex type hard-coded in nvgraph

void cuda_lt_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size);

void cuda_ic_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size);

template <typename GraphTy, typename cuda_PRNGeneratorTy>
extern void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                           size_t num_nodes, cuda_PRNGeneratorTy *d_trng_states,
                           mask_word_t *d_res_masks, size_t num_mask_words,
                           cuda_ctx<GraphTy> *ctx, cudaStream_t stream);

#if CUDA_PROFILE
template <typename logst_t, typename sample_t>
void print_profile_counter(logst_t &logst, sample_t &sample,
                           const std::string &label) {
  if (!sample.empty()) {
    auto n = sample.size();
    std::sort(sample.begin(), sample.end());
    auto tot = std::accumulate(sample.begin(), sample.end(), size_t{0});
    logst->info("cnt={}\tmin={}\tmed={}\tmax={}\tavg={}", sample.size(),
                sample[0], sample[sample.size() / 2], sample.back(),
                (float)tot / sample.size());
    auto max_qi = 100, qi_step = 1;
    for (size_t qi = qi_step; qi < max_qi; qi += qi_step) {
      auto qp = (float)qi / max_qi * 100;
      auto si = qi * sample.size() / max_qi;
      logst->info("size {}%-percentile:\t{}", qp, sample[si]);
    }
  } else
    logst->info("*** tag: {} N/A", label);
}
#endif
}  // namespace ripples

#endif  // IM_CUDA_CUDA_GENERATE_RRR_SETS_H
