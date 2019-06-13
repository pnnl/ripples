//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_IMM_H
#define RIPPLES_IMM_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "nlohmann/json.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/tim.h"
#include "ripples/utility.h"

namespace ripples {

//! The IMM algorithm configuration descriptor.
struct IMMConfiguration : public TIMConfiguration {};

//! IMM execution record.
struct IMMExecutionRecord {
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  //! Number of threads used during the execution.
  size_t NumThreads;
  //! Number of RRR sets generated.
  size_t Theta;
  //! The list of how many RRR sets are produced during the estimation phase.
  std::vector<size_t> ThetaPrimeDeltas;
  //! Execution time of the Theta estimation phase.
  ex_time_ms ThetaEstimationTotal;
  //! Execution times of the GenerateRRRSets steps in Theta estimation.
  std::vector<ex_time_ms> ThetaEstimationGenerateRRR;
  //! Execution times of the FindMostInfluentialSet steps in Theta estimation.
  std::vector<ex_time_ms> ThetaEstimationMostInfluential;
  //! Execution time of the RRR sets generation phase.
  ex_time_ms GenerateRRRSets;
  //! Execution time of the maximum coverage phase.
  ex_time_ms FindMostInfluentialSet;
  //! Total execution time.
  ex_time_ms Total;
};

//! Approximate logarithm of n chose k.
//! \param n
//! \param k
//! \return an approximation of log(n choose k).
inline double logBinomial(size_t n, size_t k) {
  return n * log(n) - k * log(k) - (n - k) * log(n - k);
}

//! Compute ThetaPrime.
//!
//! \tparam execution_tag The execution policy
//!
//! \param x The index of the current iteration.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
template <typename execution_tag>
ssize_t ThetaPrime(ssize_t x, double epsilonPrime, double l, size_t k,
                   size_t num_nodes, execution_tag &&) {
  return (2 + 2. / 3. * epsilonPrime) *
         (l * std::log(num_nodes) + logBinomial(num_nodes, k) +
          std::log(std::log2(num_nodes))) *
         std::pow(2.0, x) / (epsilonPrime * epsilonPrime);
}

//! Compute Theta.
//!
//! \param epsilon Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param LB The estimate of the lower bound.
//! \param num_nodes The number of nodes in the input graph.
inline size_t Theta(double epsilon, double l, size_t k, double LB,
                    size_t num_nodes) {
  double term1 = 0.6321205588285577;  // 1 - 1/e
  double alpha = sqrt(l * std::log(num_nodes) + std::log(2));
  double beta = sqrt(term1 * (logBinomial(num_nodes, k) +
                              l * std::log(num_nodes) + std::log(2)));
  double lamdaStar = 2 * num_nodes * (term1 * alpha + beta) *
                     (term1 * alpha + beta) * pow(epsilon, -2);

  return lamdaStar / LB;
}

//! Collect a set of Random Reverse Reachable set.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam PRNGeneratorTy The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param generator The parallel random number generator.
//! \param record Data structure storing timing and event counts.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag,
          typename execution_tag>
auto Sampling(const GraphTy &G, std::size_t k, double epsilon, double l,
              PRNGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  std::vector<RRRset<GraphTy>> RR;

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));

    record.ThetaPrimeDeltas.push_back(thetaPrime - RR.size());

    auto timeRRRSets = measure<>::exec_time([&]() {
      auto deltaRR = GenerateRRRSets(G, thetaPrime - RR.size(), generator,
                                     std::forward<diff_model_tag>(model_tag),
                                     std::forward<execution_tag>(ex_tag));

      RR.insert(RR.end(), std::make_move_iterator(deltaRR.begin()),
                std::make_move_iterator(deltaRR.end()));
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    double f;

    auto timeMostInfluential = measure<>::exec_time([&]() {
      const auto &S =
          FindMostInfluentialSet(G, k, RR, std::forward<execution_tag>(ex_tag));

      f = S.first;
    });

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      auto deltaRR = GenerateRRRSets(G, theta - RR.size(), generator,
                                     std::forward<diff_model_tag>(model_tag),
                                     std::forward<execution_tag>(ex_tag));

      RR.insert(RR.end(), std::make_move_iterator(deltaRR.begin()),
                std::make_move_iterator(deltaRR.end()));
    }
  });

  return RR;
}

//! The IMM algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNG, typename diff_model_tag,
          typename execution_tag>
auto IMM(const GraphTy &G, std::size_t k, double epsilon, double l, PRNG &gen,
         diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  IMMExecutionRecord record;

  size_t max_num_threads(1);

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp single
    max_num_threads = omp_get_max_threads();
  }

  std::vector<trng::lcg64> generator(max_num_threads, gen);

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp parallel
    {
      generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                            omp_get_thread_num());
    }
  }

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  auto R = Sampling(G, k, epsilon, l, generator, record,
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<execution_tag>(ex_tag));

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S =
      FindMostInfluentialSet(G, k, R, std::forward<execution_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  return std::make_pair(S.second, record);
}

}  // namespace ripples

#endif  // RIPPLES_IMM_H
