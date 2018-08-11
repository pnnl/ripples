//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_IMM_H
#define IM_IMM_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"
#include "trng/lcg64.hpp"

#include "im/utility.h"
#include "im/tim.h"

namespace im {

template <typename GraphTy, typename PRNGeneratorTy, typename execution_tag>
auto Sampling(const GraphTy &G, std::size_t k, double epsilon, double l, PRNGeneratorTy generator, execution_tag&& tag) {
  using vertex_type = typename GraphTy::vertex_type;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  auto logBinomial = [](size_t n, size_t k) -> double {
    return n * log(n) - k * log(k) - (n - k) * log(n - k);
  };

  double LB = 0;
  std::vector<std::vector<vertex_type>> deltaRR;
  std::vector<std::vector<vertex_type>> RR;

  std::vector<std::deque<size_t>> deltaHyperG;
  std::vector<std::deque<size_t>> HyperG;

  for (size_t i = 1; i  < std::log2(G.num_nodes()); ++i) {
    size_t x = G.num_nodes() >> i;
    // Equation 9
    float lambdaPrime = std::pow(epsilonPrime, -2) * (2 + 2./3. * epsilonPrime) *
                        (logBinomial(G.num_nodes(), k) + l * std::log(G.num_nodes()) + std::log(std::log2(G.num_nodes()))) * G.num_nodes();
    size_t thetaPrime = lambdaPrime / x;

    std::tie(deltaRR, deltaHyperG) = std::move(GenerateRRRSets(G, thetaPrime - RR.size(), generator, std::forward<execution_tag>(tag)));
    std::move(deltaRR.begin(), deltaRR.end(), std::back_inserter(RR));
    mergeHG(HyperG, deltaHyperG);

    auto S = std::move(FindMostInfluentialSet(G, k, RR, HyperG));
    double f = double(seeds.first) / RR.size();

    if (G.num_nodes() >= (1 + epsilonPrime) * x) {
      LB = (n * f) / (1 + epsilonPrime);
      break;
    }
  }

  double lamdaStar = 1;
  size_t delta = lamdaStar / LB;

  std::tie(deltaRR, deltaHyperG) = std::move(GenerateRRRSets(G, thetaPrime - RR.size(), generator, std::forward<execution_tag>(tag)));
  std::move(deltaRR.begin(), deltaRR.end(), std::back_inserter(RR));
  mergeHG(HyperG, deltaHyperG);

  return std::make_pair(std::move(RR), std::move(HyperG));
}


template <typename GraphTy, typename execution_tag>
auto IMM(const GraphTy &G, std::size_t k, double epsilon, double l, execution_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;
  ExecutionRecord record;

  std::vector<trng::lcg64> generator(max_num_threads);

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  auto R = std::move(Sampling(G, k, epsilon, epsilon, l, generator));

  auto S = std::move(FindMostInfluentialSet(G, k, R.first, R.second));

  return S.first;
}

}  // namespace im

#endif /* IM_IMM_H */
