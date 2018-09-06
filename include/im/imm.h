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

struct IMMExecutionRecord {
  size_t NumThreads;
  std::chrono::duration<double, std::milli> ThetaEstimation;
  std::chrono::duration<double, std::milli> GenerateRRRSets;
  std::chrono::duration<double, std::milli> FindMostInfluentialSet;
  std::chrono::duration<double, std::milli> Total;

  template <typename Ostream>
  friend Ostream & operator<<(Ostream &O, const IMMExecutionRecord &R) {
    O << "{ "
      << "\"NumThreads\" : " << R.NumThreads << ", "
      << "\"ThetaEstimation\" : " << R.ThetaEstimation.count() << ", "
      << "\"GenerateRRRSets\" : " << R.GenerateRRRSets.count() << ", "
      << "\"FindMostInfluentialSet\" : " << R.FindMostInfluentialSet.count() << ", "
      << "\"Total\" : " << R.Total.count()
      << " }";
    return O;
  }
};

void FuseHG(std::vector<std::deque<size_t>> &out, std::vector<std::deque<size_t>> &in, size_t firstID) {
  for (auto & e : in)
    for (auto & v : e)
      v += firstID;

  for (size_t i = 0; i < in.size(); ++i)
    out[i].insert(out[i].end(), in[i].begin(), in[i].end());
}

template <typename GraphTy, typename PRNGeneratorTy, typename execution_tag>
auto Sampling(const GraphTy &G, std::size_t k, double epsilon, double l, PRNGeneratorTy &generator, IMMExecutionRecord & record, execution_tag&& tag) {
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
  std::vector<std::deque<size_t>> HyperG(G.num_nodes());

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    float thetaPrime = (2 + 2./3. * epsilonPrime) *
                       (l * std::log(G.num_nodes()) + logBinomial(G.num_nodes(), k) + std::log(std::log2(G.num_nodes()))) * std::pow(2.0, x) / (epsilonPrime * epsilonPrime);

    std::tie(deltaRR, deltaHyperG) = std::move(GenerateRRRSets(G, thetaPrime - RR.size(), generator, std::forward<execution_tag>(tag)));

    size_t firstID = RR.size();
    std::move(deltaRR.begin(), deltaRR.end(), std::back_inserter(RR));
    FuseHG(HyperG, deltaHyperG, firstID);

    auto S = std::move(FindMostInfluentialSet(G, k, RR, HyperG));
    double f = double(S.first) / RR.size();

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  double term1 = 0.6321205588285577;    // 1 - 1/e
  double alpha = sqrt(l * std::log(G.num_nodes()) + std::log(2));
  double beta = sqrt(term1 * (logBinomial(G.num_nodes(), k) + l * std::log(G.num_nodes()) + std::log(2)));
  double lamdaStar = 2 * G.num_nodes() * (term1 * alpha + beta) * (term1 * alpha + beta) * pow(epsilon, -2);
  size_t delta = lamdaStar / LB;
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimation = end - start;

  start = std::chrono::high_resolution_clock::now();
  if (delta > RR.size()) {
    std::tie(deltaRR, deltaHyperG) = std::move(GenerateRRRSets(G, delta - RR.size(), generator, std::forward<execution_tag>(tag)));
    size_t firstID = RR.size();
    std::move(deltaRR.begin(), deltaRR.end(), std::back_inserter(RR));
    FuseHG(HyperG, deltaHyperG, firstID);
  }
  end = std::chrono::high_resolution_clock::now();

  record.GenerateRRRSets = end - start;

  return std::make_pair(std::move(RR), std::move(HyperG));
}


template <typename GraphTy, typename execution_tag>
auto IMM(const GraphTy &G, std::size_t k, double epsilon, double l, execution_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;
  IMMExecutionRecord record;

  size_t max_num_threads(1);

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
    #pragma omp single
    max_num_threads = omp_get_max_threads();
  }

  std::vector<trng::lcg64> generator(max_num_threads);

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
    #pragma omp parallel
    {
      generator[omp_get_thread_num()].seed(0UL);
      generator[omp_get_thread_num()].split(omp_get_num_threads(), omp_get_thread_num());
    }
  }

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  auto R = std::move(Sampling(G, k, epsilon, l, generator, record, std::forward<execution_tag>(tag)));

  auto start = std::chrono::high_resolution_clock::now();
  auto S = std::move(FindMostInfluentialSet(G, k, R.first, R.second));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  return std::make_pair(S.second, record);
}

}  // namespace im

#endif /* IM_IMM_H */
