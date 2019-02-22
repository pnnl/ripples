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

//template<typename res_t, typename GraphTy, typename PRNGeneratorTy,
//		typename diff_model_tag>
//void CudaGenerateRRRSets(res_t &rrr_sets, const GraphTy &G, size_t theta,
//		PRNGeneratorTy &generator, diff_model_tag &&model_tag);

// forward declarations to enable separate compilation
using cuda_GraphTy = im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>>;
using cuda_res_t = std::vector<std::vector<typename cuda_GraphTy::vertex_type>>;
using cuda_PRNGeneratorTy = std::vector<trng::lcg64>;

void CudaGenerateRRRSets(cuda_res_t &, const cuda_GraphTy &, size_t,
		cuda_PRNGeneratorTy&, im::linear_threshold_tag&&);

void CudaGenerateRRRSets(cuda_res_t &, const cuda_GraphTy &, size_t,
		cuda_PRNGeneratorTy&, im::independent_cascade_tag&&);

} // namespace im

#endif  // IM_CUDA_CUDA_GENERATE_RRR_SETS_H
