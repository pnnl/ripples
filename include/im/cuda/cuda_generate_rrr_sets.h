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

#include "im/diffusion_simulation.h"
#include "im/graph.h"

namespace im {

// TODO why template+specialization doesn't work?
// template<typename res_t, typename GraphTy, typename PRNGeneratorTy,
//		typename diff_model_tag>
// void CudaGenerateRRRSets(res_t &rrr_sets, const GraphTy &G, size_t theta,
//		PRNGeneratorTy &generator, diff_model_tag &&model_tag);

// forward declarations to enable separate compilation
using cuda_GraphTy =
    im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>>;
using cuda_res_t = std::vector<std::vector<typename cuda_GraphTy::vertex_type>>;
using cuda_PRNGeneratorTy = std::vector<trng::lcg64>;

//! \brief Generate Random Reverse Reachability Sets according to Linear
//! Threshold model - CUDA.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param model_tag The diffusion model tag.
//! \param seed The seed for the device-side generator.
//!
//! \return A list of theta Random Reverse Rachability Sets.
cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::linear_threshold_tag &&model_tag,
                               unsigned long long seed);

//! \brief Generate Random Reverse Reachability Sets according to Independent
//! Cascade model - CUDA.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param model_tag The diffusion model tag.
//! \param seed The seed for the device-side generator.
//!
//! \return A list of theta Random Reverse Rachability Sets.
cuda_res_t CudaGenerateRRRSets(const cuda_GraphTy &G, size_t theta,
                               im::independent_cascade_tag &&model_tag,
                               unsigned long long seed);

}  // namespace im

#endif  // IM_CUDA_CUDA_GENERATE_RRR_SETS_H
