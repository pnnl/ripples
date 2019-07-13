//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_CUDA_CUDA_GENERATE_RRR_SETS_H
#define RIPPLES_CUDA_CUDA_GENERATE_RRR_SETS_H

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "trng/lcg64.hpp"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#define CUDA_CHECK 1
#define CUDA_PROFILE 1

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

struct cuda_device_graph {
  using vertex_t = int; // TODO vertex type hard-coded in nvgraph
  using weight_t = typename cuda_GraphTy::edge_weight_type;
  vertex_t *d_index_ = nullptr, *d_edges_ = nullptr;
  weight_t *d_weights_ = nullptr;
};

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


typename cuda_device_graph::vertex_t *cuda_graph_index();
typename cuda_device_graph::vertex_t *cuda_graph_edges();
typename cuda_device_graph::weight_t *cuda_graph_weights();

//! \brief Returns the maximum number of CUDA blocks.
//!
//! \return The maximum number of CUDA blocks.
size_t cuda_max_blocks();

//! \brief Finalize CUDA execution context.
void cuda_fini();

//! \brief Generate Random Reverse Reachability Sets with CUDA.
//!
//! \param theta The number of RRR sets to be generated.
//!
//! \return A list of theta Random Reverse Rachability Sets.
cuda_res_t CudaGenerateRRRSets(size_t theta);

//
// host-device API
//
using mask_word_t = typename cuda_device_graph::vertex_t;

void cuda_graph_init(const cuda_GraphTy &G);
void cuda_graph_fini();

void cuda_malloc(void **dst, size_t size);
void cuda_free(void *ptr);
void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size, cudaStream_t);
void cuda_sync(cudaStream_t);

void cuda_lt_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size);

void cuda_ic_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size);

void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                    size_t num_nodes, cuda_PRNGeneratorTy *d_trng_states,
                    mask_word_t *d_res_masks, size_t num_mask_words,
                    cudaStream_t);

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
