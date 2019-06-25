//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_CUDA_CUDA_GENERATE_RRR_SETS_H
#define RIPPLES_CUDA_CUDA_GENERATE_RRR_SETS_H

#include <cstddef>
#include <vector>

#include "trng/lcg64.hpp"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace ripples {

//
// host-host API
//

// TODO why template+specialization doesn't work?
// template<typename res_t, typename GraphTy, typename PRNGeneratorTy,
//		typename diff_model_tag>
// void CudaGenerateRRRSets(res_t &rrr_sets, const GraphTy &G, size_t theta,
//		PRNGeneratorTy &generator, diff_model_tag &&model_tag);

// forward declarations to enable separate compilation
using cuda_GraphTy =
    ripples::Graph<uint32_t, float, ripples::BackwardDirection<uint32_t>>;
using cuda_res_t = std::vector<std::vector<typename cuda_GraphTy::vertex_type>>;
using cuda_PRNGeneratorTy = trng::lcg64;
using cuda_PRNGeneratorsTy = std::vector<cuda_PRNGeneratorTy>;

constexpr size_t CUDA_WALK_SIZE = 8;

//! \brief Initialize CUDA execution context for LT model.
//!
//! \param G The input host-side graph.
//! \param model_tag The diffusion model tag.
void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &,
               ripples::linear_threshold_tag &&model_tag);

//! \brief Initialize CUDA execution context for IC model.
//!
//! \param G The input host-side graph.
//! \param model_tag The diffusion model tag.
void cuda_init(const cuda_GraphTy &G, const cuda_PRNGeneratorTy &,
               ripples::independent_cascade_tag &&model_tag);

//! \brief Returns the total number of threads (i.e., CPU+GPU) performing walks.
//!
//! \return The total number of threads performing walks.
size_t cuda_num_total_threads();

//! \brief Returns the number of CPU threads performing walks.
//!
//! \return The number of CPU threads performing walks.
size_t cuda_num_cpu_threads();

//! \brief Returns the GPU warp size.
//!
//! \return The GPU warp size.
size_t cuda_warp_size();

//! \brief Finalize CUDA execution context for LT model.
//!
//! \param model_tag The diffusion model tag.
void cuda_fini(ripples::linear_threshold_tag &&model_tag);

//! \brief Finalize CUDA execution context for IC model.
//!
//! \param model_tag The diffusion model tag.
void cuda_fini(ripples::independent_cascade_tag &&model_tag);

//! \brief Generate Random Reverse Reachability Sets according to Linear
//! Threshold model - CUDA.
//!
//! \param theta The number of RRR sets to be generated.
//! \param model_tag The diffusion model tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
cuda_res_t CudaGenerateRRRSets(size_t theta, cuda_PRNGeneratorsTy &generators,
                               ripples::linear_threshold_tag &&model_tag);

//! \brief Generate Random Reverse Reachability Sets according to Independent
//! Cascade model - CUDA.
//!
//! \param theta The number of RRR sets to be generated.
//! \param model_tag The diffusion model tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
cuda_res_t CudaGenerateRRRSets(size_t theta, cuda_PRNGeneratorsTy &generators,
                               ripples::independent_cascade_tag &&model_tag);

//
// host-device API
//
using mask_word_t = typename cuda_GraphTy::vertex_type;

#define CUDA_CHECK 0
#define CUDA_PROFILE 1

//
// check utilities
//
template <typename graph_t>
bool reaches(const graph_t &g, typename graph_t::vertex_type src,
             typename graph_t::vertex_type dst) {
  assert(src != dst);
  for (auto &n : g.neighbors(src))
    if (n.vertex == dst) return true;
  return false;
}

template <typename rrr_t, typename graph_t>
bool check_lt_from(const rrr_t &r,
                   const typename rrr_t::const_iterator &root_it,
                   const graph_t &g) {
  if (r.size() == 1) return root_it == r.begin();
  auto wr = r;
  auto root = *root_it;
  wr.erase(wr.begin() + std::distance(r.begin(), root_it));
  for (auto it = wr.begin(); it != wr.end(); ++it)
    if (reaches(g, root, *it) && check_lt_from(wr, it, g)) return true;
  return false;
}

#if CUDA_CHECK
template <typename rrr_t, typename graph_t>
void check_lt(const rrr_t &r, const graph_t &g, size_t id) {
  bool res = false;
  for (auto it = r.begin(); it != r.end(); ++it) {
    if (check_lt_from(r, it, g)) {
      res = true;
      break;
    }
  }

  if (!res) {
    spdlog::error("check_lt FAILED\n");
    exit(1);
  }
}
#else
template <typename... Args>
void check_lt(Args &&...) {}
#endif

void cuda_graph_init(const cuda_GraphTy &G);
void cuda_malloc(void **dst, size_t size);
void cuda_free(void *ptr);
void cuda_rng_setup(size_t n_blocks, size_t block_size,
                    cuda_PRNGeneratorTy *d_trng_state, size_t rank,
                    const cuda_PRNGeneratorTy &r, size_t warp_step,
                    size_t max_batch_size, size_t num_total_threads,
                    size_t rng_offset);
void cuda_graph_fini();
void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                    size_t num_nodes, size_t warp_step,
                    cuda_PRNGeneratorTy *d_trng_states,
                    mask_word_t *d_res_masks, size_t num_mask_words);
void cuda_ic_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                    size_t num_nodes, size_t warp_step,
                    cuda_PRNGeneratorTy *d_trng_states,
                    mask_word_t *d_res_masks, size_t num_mask_words);
void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size);

#if CUDA_PROFILE
template <typename logst_t, typename mt_sample_t>
void print_profile_breakdown(logst_t &logst, mt_sample_t &mt_sample,
                             const std::string &label) {
  if (!mt_sample.empty()) {
    logst->info("*** tag: {}", label);
    for (size_t tid = 0; tid < mt_sample.size(); ++tid) {
      auto &sample(mt_sample[tid]);
      if (!sample.empty()) {
        std::sort(sample.begin(), sample.end());
        auto tot = std::accumulate(
            sample.begin(), sample.end(), std::chrono::nanoseconds{0},
            [](std::chrono::nanoseconds acc, std::chrono::nanoseconds x) {
              return acc + x;
            });
        logst->info("[tid={}]\tcnt={}\tmin={}\tmed={}\tmax={}\tavg={}\ttot={}",
                    tid, sample.size(), sample[0].count(),
                    sample[sample.size() / 2].count(), sample.back().count(),
                    (float)tot.count() / sample.size(), tot.count());
      } else {
        logst->info("[tid={}] N/A", tid);
      }
    }
  } else
    logst->info("*** tag: {} N/A", label);
}

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
