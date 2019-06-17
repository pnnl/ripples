//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_GENERATE_RRR_SETS_H
#define IM_CUDA_CUDA_GENERATE_RRR_SETS_H

#include <cstddef>
#include <vector>

#include "trng/lcg64.hpp"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"

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

//! \brief Initialize CUDA execution context for LT model.
//!
//! \param G The input host-side graph.
//! \param model_tag The diffusion model tag.
void cuda_init(const cuda_GraphTy &G,
		       const cuda_PRNGeneratorTy &,
               ripples::linear_threshold_tag &&model_tag);

//! \brief Initialize CUDA execution context for IC model.
//!
//! \param G The input host-side graph.
//! \param model_tag The diffusion model tag.
void cuda_init(const cuda_GraphTy &G,
		       const cuda_PRNGeneratorTy &,
               ripples::independent_cascade_tag &&model_tag);

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
cuda_res_t CudaGenerateRRRSets(size_t theta,
                               ripples::linear_threshold_tag &&model_tag);

//! \brief Generate Random Reverse Reachability Sets according to Independent
//! Cascade model - CUDA.
//!
//! \param theta The number of RRR sets to be generated.
//! \param model_tag The diffusion model tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
cuda_res_t CudaGenerateRRRSets(size_t theta,
                               ripples::independent_cascade_tag &&model_tag);

//
// host-device API
//
using mask_word_t = typename cuda_GraphTy::vertex_type;
constexpr size_t MAX_SET_SIZE = 256;
constexpr size_t MAX_PARDEG = 10;

#define CUDA_DBG 0
#define CUDA_CHECK 0
#define CUDA_PROFILE 1

#if CUDA_DBG
#define CUDA_LOG(...) printf(__VA_ARGS__)
#else
#define CUDA_LOG(...)
#endif

constexpr size_t debug_walk_id = 1;

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
  CUDA_LOG("> checking set %d: ", id);
  for (auto &v : r) CUDA_LOG("%d ", v);
  CUDA_LOG("\n");

  bool res = false;
  for (auto it = r.begin(); it != r.end(); ++it) {
    if (check_lt_from(r, it, g)) {
      res = true;
      break;
    }
  }

  if (!res) {
    printf("> check FAILED\n");
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
		const cuda_PRNGeneratorTy &r, size_t warp_step, size_t max_batch_size);
void cuda_graph_fini();
void cuda_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
		size_t num_nodes,
		size_t warp_step, cuda_PRNGeneratorTy *d_trng_states,
		mask_word_t *d_res_masks);
void cuda_d2h(mask_word_t *dst, mask_word_t *src, size_t size);
}  // namespace ripples

#endif  // IM_CUDA_CUDA_GENERATE_RRR_SETS_H
