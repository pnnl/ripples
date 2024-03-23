//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_OPIMC_H
#define RIPPLES_OPIMC_H

#include <algorithm>

#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/opimc_execution_record.h"
#include "ripples/streaming_rrr_generator.h"
#include "ripples/utility.h"

namespace ripples {

inline float ThetaHelper(size_t n, float epsilon, size_t k, float delta) {
  constexpr float A = 0.6321203113733684;  // 1 - 1/e

  float B = A * sqrt(log(6.0 / delta));
  float C = sqrt(A * (logBinomial(n, k) + log(6.0 / delta)));
  float D = (B + C) * (B + C);
  return (2 * D);
}

//! Equation (16) in the original paper.
inline size_t ThetaMax(size_t n, float epsilon, size_t k, float delta) {
  return ceil(n * ThetaHelper(n, epsilon, k, delta) / (epsilon * epsilon * k));
}

//! Equation (17) in the original paper.
inline size_t ThetaZero(size_t n, float epsilon, size_t k, float delta) {
  return ceil(ThetaHelper(n, epsilon, k, delta));
}

//! Equation (5) in the original paper.
inline float LowerBound(size_t coverage, float delta2, size_t n, size_t theta) {
  float A = log(1 / delta2);
  float B = sqrt(coverage + 2 * A / 9);
  float C = sqrt(A / 2);
  float D = (B - C) * (B - C);
  float E = A / 18;

  return (float(n) / theta) * (D - E);
}

//! Equation (6) in the original paper.
inline float UpperBound(size_t coverage) {
  constexpr float A = 0.6321203113733684;  // 1 - 1/e

  return A * coverage;
}

template <typename RRRSetsTy, typename SeedSetTy>
size_t FindCoverage(const RRRSetsTy &RR, const SeedSetTy &S) {
  size_t result = 0;

#pragma omp parallel for reduction(+ : result)
  for (size_t i = 0; i < RR.size(); ++i) {
    auto itr = S.end();
    for (size_t j = 0; itr == S.end() && j < RR[i].size(); ++j) {
      itr = std::find(S.begin(), S.end(), RR[i][j]);
    }

    if (itr != S.end()) {
      result += 1;
    }
  }
  return result;
}

//! The OPIM-C algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ConfTy The configuration type
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param CFG The configuration.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
std::vector<typename GraphTy::vertex_type> OPIMC(const GraphTy &G,
                                                 const ConfTy &CFG, double l,
                                                 GeneratorTy &gen,
                                                 OPIMCExecutionRecord &record,
                                                 diff_model_tag &&model_tag,
                                                 omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  constexpr float A = 0.6321203113733684;  // 1 - 1/e

  auto console = spdlog::get("console");

  size_t thetaZero = ThetaZero(G.num_nodes(), CFG.epsilon, CFG.k, CFG.delta);
  size_t thetaMax = ThetaMax(G.num_nodes(), CFG.epsilon, CFG.k, CFG.delta);

  RRRsetAllocator<GraphTy> allocator;
  RRRsets<GraphTy> R1(thetaZero, RRRset<GraphTy>(allocator));
  RRRsets<GraphTy> R2(thetaZero, RRRset<GraphTy>(allocator));

  GenerateRRRSets(G, gen, R1.begin(), R1.end(), record,
                  std::forward<diff_model_tag>(model_tag),
                  std::forward<omp_parallel_tag>(ex_tag));
  GenerateRRRSets(G, gen, R2.begin(), R2.end(), record,
                  std::forward<diff_model_tag>(model_tag),
                  std::forward<omp_parallel_tag>(ex_tag));

  size_t iMax = ceil(log2(thetaMax / thetaZero));
  float delta = CFG.delta / (3 * iMax);
  std::vector<vertex_type> results;
  for (size_t i = 0;; ++i) {
    const auto &[coverage1, seeds] = FindMostInfluentialSet(
        G, CFG, R1.begin(), R1.end(), record, gen.isGpuEnabled(),
        std::forward<omp_parallel_tag>(ex_tag));

    size_t coverage2 = FindCoverage(R2, seeds);

    float upperBound = UpperBound(coverage1 * R1.size());
    float lowerBound = LowerBound(coverage2, delta, G.num_nodes(), R2.size());

    float alpha = upperBound / lowerBound;

    results = std::move(seeds);
    if (alpha >= (A - CFG.epsilon)) break;

    thetaZero *= 2;

    size_t delta = thetaZero - R1.size();

    R1.insert(R1.end(), delta, RRRset<GraphTy>(allocator));
    R2.insert(R2.end(), delta, RRRset<GraphTy>(allocator));

    auto begin1 = R1.end() - delta;
    auto begin2 = R2.end() - delta;

    GenerateRRRSets(G, gen, begin1, R1.end(), record,
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<omp_parallel_tag>(ex_tag));
    GenerateRRRSets(G, gen, begin2, R2.end(), record,
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<omp_parallel_tag>(ex_tag));
  }

  return results;
}

}  // namespace ripples

#endif
