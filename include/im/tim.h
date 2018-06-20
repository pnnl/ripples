//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_TIM_H
#define IM_TIM_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <random>
#include <iterator>
#include <queue>

#include <omp.h>

#include "im/bfs.h"

#include "boost/math/special_functions/binomial.hpp"

namespace im {

struct tim_tag {};

#if 0
template <typename GraphTy, typename RRRSetList>
RRRSetList ReduceRandomRRSetList(typename GraphTy::vertex_type v,
                                 RRRSetList &R);

//! \brief TIM theta estimation function.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \param k The size of the seed set
template <typename GraphTy>
size_t thetaEstimation(GraphTy &G, size_t k, double epsilon) {
  // Compute KPT* according to Algorithm 2
  size_t KPTStar = 1;

  size_t start = 0;
  double sum = 0;

  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result;

  for (size_t i = 1; i < G.scale(); i <<= 1) {
    double c_i = 6 * log10(G.scale()) + 6 * log10(log2(G.scale())) * i;

#pragma omp parallel reduction(+:sum)
    {
      size_t size = omp_get_num_threads();
      size_t rank = omp_get_thread_num();

      size_t end = std::ceil(c_i);
      size_t chunk = end - start;

      std::default_random_engine generator;
      generator.discard(start + rank * (chunk * G.size() / size));

      std::vector<std::unordered_set<typename GraphTy::vertex_type>> intermediate_result;
      for (size_t j = rank * chunk/size; j < (rank + 1) * chunk/size; ++j) {
        auto RRset = BFSOnRandomGraph(G, generator);

        assert(!RRset.empty());

        double WR = 0;
        for (auto vertex : RRset) {
          WR += G.in_degree(vertex);
        }
        // Equation (8) of the paper.
        double KR = 1 - pow(1.0 - WR / G.size(), k);
        sum += KR;

        intermediate_result.emplace_back(std::move(RRset));
      }

#pragma omp critical
      std::move(intermediate_result.begin(), intermediate_result.end(), std::back_inserter(result));
    }

    start = std::ceil(c_i);

    if ((sum / c_i) > (1.0 / i)) {
      KPTStar = G.scale() * sum / (c_i * 2);
      break;
    }
  }

  size_t l = 1;

  // Try to refine the bound computing KPT' with Algorithm 3
  std::unordered_set<typename GraphTy::vertex_type> seedSet;
  while (seedSet.size() < k && !result.empty()) {
    // 1 - Find the most influential vertex v
    typename GraphTy::vertex_type v = GetMostInfluential(G, result);

    // 2 - Add v to seedSet
    seedSet.insert(v);

    // 3 - Remove all the RRRSet that includes v
    result = std::move(ReduceRandomRRSetList<GraphTy>(v, result));
  }
  double epsilonPrime = 5 * cbrt(l * pow(epsilon, 2) / (k + l));
  double lambdaPrime = (2 + epsilonPrime) * l * G.scale() * log10(G.scale()) * pow(epsilonPrime, -2);
  size_t thetaPrime = lambdaPrime / KPTStar;

  auto RRRsecond = generateRandomRRSet(G, thetaPrime, tim_tag());

  double f = 0;
  for (auto & set : RRRsecond) {
    auto itr = std::find_first_of(set.begin(), set.end(), seedSet.begin(), seedSet.end());
    if (itr != set.end()) ++f;
  }
  f /= RRRsecond.size();
  
  size_t KPTPrime = std::ceil(f * G.scale()/(1 + epsilonPrime));

  // KPT+ = max{KPT*, KPT'}
  size_t KPTPlus = std::max(KPTStar, KPTPrime);

  // Compute lambda from equation (4)
  double lambda =
      (8 + 2 * epsilon) * G.scale() *
      (log10(l * G.scale()) +
       log10(boost::math::binomial_coefficient<double>(G.scale(), k)) +
       log10(2.0)) *
      pow(epsilon, -2);

  // Compute theta
  size_t theta = lambda / KPTPlus;
  return theta;
}

//! \brief Generate Random RR sets.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \param theta The number of random RR set to be generated
//! \return A set of theta random RR set.
template <typename GraphTy>
std::vector<std::unordered_set<typename GraphTy::vertex_type>> generateRandomRRSet(
    GraphTy &G, size_t theta, const tim_tag &) {
  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result;

#pragma omp parallel
  {
    size_t size = omp_get_num_threads();
    size_t rank = omp_get_thread_num();

    std::default_random_engine generator;
    generator.discard(rank * (theta * G.size() / size));

    std::vector<std::unordered_set<typename GraphTy::vertex_type>> intermediate_result;
    for (size_t i = rank * theta/size; i < (rank + 1) * theta/size; ++i) {
      auto influenced_set = BFSOnRandomGraph(G, generator);
      intermediate_result.emplace_back(std::move(influenced_set));
    }
#pragma omp critical
    std::move(intermediate_result.begin(), intermediate_result.end(), std::back_inserter(result));
  }

  return result;
}

template <typename GraphTy>
std::vector<size_t> generateRandomRRSet(
    GraphTy &G, size_t theta, const rr_size_measure_tag &) {
  std::vector<size_t> result;

#pragma omp parallel
  {
    size_t size = omp_get_num_threads();
    size_t rank = omp_get_thread_num();

    std::default_random_engine generator;
    generator.discard(rank * (theta * G.size() / size));

    std::vector<size_t> intermediate_result;
    for (size_t i = rank * theta/size; i < (rank + 1) * theta/size; ++i) {
      auto influenced_set = BFSOnRandomGraph(G, generator);
      intermediate_result.emplace_back(influenced_set.size());
    }
#pragma omp critical
    result.insert(result.end(), intermediate_result.begin(), intermediate_result.end());
  }

  return result;
}

//! \brief Find the most influential node
//!
//! \tparam GraphTy The type of the graph.
//! \tparam RRRSetlist The type used to store the Random RR set.
//! \param G The graph instance.
//! \param R A collection of random RR sets.
//! \return The vertex appearing the most in R.
template <typename GraphTy, typename RRRSetList>
typename GraphTy::vertex_type GetMostInfluential(GraphTy &G, RRRSetList &R) {
  std::vector<size_t> counters(G.scale(), 0ul);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < G.scale(); ++i) {
    for (auto & r : R) {
      if (r.find(i) != r.end())
        ++counters[i];
    }
  }

  struct alignas(64) maxVertex {
    typename GraphTy::vertex_type vertex;
    size_t value{0};
  };


  alignas(64) maxVertex result[omp_get_max_threads()];
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < counters.size(); ++i) {
    if (result[omp_get_thread_num()].value < counters[i]) {
      result[omp_get_thread_num()].vertex = i;
      result[omp_get_thread_num()].value = counters[i];
    }
  }

  for (size_t i = 1; i < omp_get_max_threads(); ++i)
    if (result[0].value < result[i].value)
      result[0] = result[i];

  return result[0].vertex;
}

//! \brief Remove all the RR set containing a given vertex.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam RRRSetlist The type used to store the Random RR set.
//! \param v The vertex.
//! \param R A collection of random RR sets.
//! \return The updated list of RR sets.
template <typename GraphTy, typename RRRSetList>
RRRSetList ReduceRandomRRSetList(typename GraphTy::vertex_type v,
                                 RRRSetList &R) {
  RRRSetList result;

#pragma omp parallel
  {
    RRRSetList intermediate_result;

#pragma omp for schedule(static)
    for (auto itr = R.begin(); itr < R.end(); ++itr) {
      if (itr->find(v) == itr->end()) {
        intermediate_result.emplace_back(std::move(*itr));
      }
    }
#pragma omp critical
    std::move(intermediate_result.begin(), intermediate_result.end(), std::back_inserter(result));
  }

  return result;
}

//! \brief The TIM influence maximization algorithm.
//!
//! \tparm GraphTy The type of the graph.
//!
//! \param G The instance of the graph.
//! \param k The size of the seed set.
template <typename GraphTy>
std::unordered_set<typename GraphTy::vertex_type> influence_maximization(
    GraphTy &G, size_t k, double epsilon, const tim_tag & tag) {
  // Estimate the number of Random Reverse Reacheable Sets needed
  // Algorithm 2 in Tang Y. et all
  size_t theta = thetaEstimation(G, k, epsilon);

  // - Random Reverse Reacheable Set initialize to the empty set
  using RRRSet = std::unordered_set<typename GraphTy::vertex_type>;
  std::vector<RRRSet> R = std::move(generateRandomRRSet(G, theta, tag));

  assert(R.size() == theta);

  // - Initialize the seed set to the empty set
  std::unordered_set<typename GraphTy::vertex_type> seedSet;
  while (seedSet.size() < k && !R.empty()) {
    // 1 - Find the most influential vertex v
    typename GraphTy::vertex_type v = GetMostInfluential(G, R);

    // 2 - Add v to seedSet
    seedSet.insert(v);

    // 3 - Remove all the RRRSet that includes v
    R = std::move(ReduceRandomRRSetList<GraphTy>(v, R));
  }
  return seedSet;
}

#endif

template <typename GraphTy, typename PRNG>
size_t WR(GraphTy &G, typename GraphTy::vertex_type r, PRNG &generator) {
  using vertex_type = typename GraphTy::vertex_type;

  std::uniform_real_distribution<double> value(0.0, 1.0);

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.nodes(), false);

  queue.push(r);
  visited[r] = true;
  size_t wr = 1;

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    wr += G.in_degree(v);

    for (auto u : G.in_neighbors(v)) {
      if (!visited[u.v] &&
          value(generator) > u.attribute) {
        visited[u.v] = true;
        queue.push(u.v);
      }
    }
  }

  return wr;
}

template <typename GraphTy>
size_t KptEstimation(GraphTy &G, size_t k, double epsilon, double l) {
  // Compute KPT* according to Algorithm 2
  size_t KPTStar = 1;

  std::default_random_engine generator;

  for (size_t i = 0; i < log2(G.nodes()); ++i) {
    size_t c_i = (6/l * log10(G.nodes()) + 6 * log10(log2(G.nodes()))) * (1ul << i);
    size_t sum = 0;
    for (size_t j = 0; j < c_i; ++j) {
      // Pick a random vertex
      typename GraphTy::vertex_type v = generator() % G.nodes();

      size_t wr = WR(G, v, generator);

      // Equation (8) of the paper.
      double KR = 1 - pow(1.0 - wr / G.edges(), k);
      sum += KR;
    }

    sum /= c_i;
    if (sum > (1.0 / (1ul << i))) {
      KPTStar = G.nodes() * sum / (c_i * 2);
      break;
    }
  }

  return KPTStar;
#if 0
  size_t start = 0;
  double sum = 0;

  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result;

  for (size_t i = 1; i < G.scale(); i <<= 1) {
    double c_i = 6 * log10(G.scale()) + 6 * log10(log2(G.scale())) * i;

#pragma omp parallel reduction(+:sum)
    {
      size_t size = omp_get_num_threads();
      size_t rank = omp_get_thread_num();

      size_t end = std::ceil(c_i);
      size_t chunk = end - start;

      std::default_random_engine generator;
      generator.discard(start + rank * (chunk * G.size() / size));

      std::vector<std::unordered_set<typename GraphTy::vertex_type>> intermediate_result;
      for (size_t j = rank * chunk/size; j < (rank + 1) * chunk/size; ++j) {
        auto RRset = BFSOnRandomGraph(G, generator);

        assert(!RRset.empty());

        double WR = 0;
        for (auto vertex : RRset) {
          WR += G.in_degree(vertex);
        }
        // Equation (8) of the paper.
        double KR = 1 - pow(1.0 - WR / G.size(), k);
        sum += KR;

        intermediate_result.emplace_back(std::move(RRset));
      }

#pragma omp critical
      std::move(intermediate_result.begin(), intermediate_result.end(), std::back_inserter(result));
    }

    start = std::ceil(c_i);

    if ((sum / c_i) > (1.0 / i)) {
      KPTStar = G.scale() * sum / (c_i * 2);
      break;
    }
#endif
}

template <typename GraphTy>
std::unordered_set<typename GraphTy::vertex_type>
TIM(const GraphTy &G, size_t k, double epsilon) {
  using vertex_type = typename GraphTy::vertex_type;
  std::unordered_set<vertex_type> seedSet;

  return seedSet;
}

}  // namespace im

#endif /* IM_TIM_H */
