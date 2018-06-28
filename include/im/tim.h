//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_TIM_H
#define IM_TIM_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>

#include <omp.h>

#include "im/bfs.h"
#include "im/utility.h"

#include "spdlog/spdlog.h"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"
#include "trng/yarn2.hpp"

namespace im {

//! \brief Compute the number of elements in the RRR set starting at r.
//!
//! \tparam GraphTy The type of the Graph.
//! \tparam PNRG The type of the random number generator.
//!
//! \param G The original graph.
//! \param r The start vertex.
//! \param generator The random number generator used to sample G.
//!
//! \return The number of elements in the RRR set computed starting from r.
template <typename GraphTy, typename PRNG>
size_t WR(GraphTy &G, typename GraphTy::vertex_type r, PRNG &generator) {
  using vertex_type = typename GraphTy::vertex_type;

  trng::uniform01_dist<float> value;

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);

  queue.push(r);
  visited[r] = true;
  size_t wr = 1;

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    wr += G.in_degree(v);

    for (auto u : G.in_neighbors(v)) {
      if (!visited[u.vertex] && value(generator) < u.weight) {
        visited[u.vertex] = true;
        queue.push(u.vertex);
      }
    }
  }

  return wr;
}

//! \brief Estimate KPT.  Algorithm 2 in the original paper.
//!
//! \tparam GraphTy The type of the graph.
//!
//! \param G The original graph.
//! \param k The size of the desired seed set.
//! \param tag The execution policy tag.
//!
//! \return a lower bond of OPT computed with Algoirthm 2 of the original paper.
template <typename GraphTy>
double KptEstimation(GraphTy &G, size_t k, sequential_tag &&tag) {
  // Compute KPT* according to Algorithm 2
  double KPTStar = 1;

  trng::yarn2 generator;

  trng::uniform_int_dist root(0, G.num_nodes());

  for (size_t i = 1; i < log2(G.num_nodes()); ++i) {
    double sum = 0;
    size_t c_i =
        (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * (1ul << i);

    for (size_t j = 0; j < c_i; ++j) {
      // Pick a random vertex
      typename GraphTy::vertex_type v = root(generator);

      double wr = WR(G, v, generator);
      wr /= G.num_edges();
      // Equation (8) of the paper.
      sum += 1 - pow(1.0 - wr, k);
    }

    sum /= c_i;

    if (sum > (1.0 / (1ul << i))) {
      KPTStar = G.num_nodes() * sum / 2;
      spdlog::get("perf")->trace("i = {}, KPTStar = {}, sum = {}, c_i = {}", i, KPTStar,
                                 sum, c_i);
      break;
    }
  }

  return KPTStar;
}

//! \brief Estimate KPT.  Parallelization of Algorithm 2 in the original paper.
//!
//! \tparam GraphTy The type of the graph.
//!
//! \param G The original graph.
//! \param k The size of the desired seed set.
//! \param tag The execution policy tag.
//!
//! \return a lower bond of OPT computed with Algoirthm 2 of the original paper.
template <typename GraphTy>
double KptEstimation(GraphTy &G, size_t k, omp_parallel_tag &&tag) {
  double KPTStar = 1.0;

  for (size_t i = 2; i < G.num_nodes(); i <<= 1) {
    size_t c_i = (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * i;
    double sum = 0;

#pragma omp parallel reduction(+ : sum)
    {
      size_t size = omp_get_num_threads();
      size_t rank = omp_get_thread_num();

      trng::yarn2 generator;
      generator.split(size, rank);

      trng::uniform_int_dist root(0, G.num_nodes());

      size_t chunk = c_i;

      for (size_t j = rank * chunk / size; j < (rank + 1) * chunk / size; ++j) {
        // Pick a random vertex
        typename GraphTy::vertex_type v = root(generator);

        double wr = WR(G, v, generator);
        wr /= G.num_edges();
        // Equation (8) of the paper.
        sum += 1 - pow(1.0 - wr, k);
      }
    }

    sum /= c_i;

    if (sum > (1.0 / i)) {
      KPTStar = G.num_nodes() * sum / 2;
      spdlog::get("perf")->trace("i = {}, KPTStar = {}, sum = {}, c_i = {}", i, KPTStar,
                                 sum, c_i);
      break;
    }
  }

  return KPTStar;
}

//! \brief Generate Random Reverse Reachability Sets.
//!
//! \tparam GraphTy The type of the garph.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy>
std::vector<std::unordered_set<typename GraphTy::vertex_type>> GenerateRRRSets(
    GraphTy &G, size_t theta, sequential_tag &&tag) {
  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result(theta);

  trng::yarn2 generator;
  trng::uniform_int_dist start(0, G.num_nodes());

  for (size_t i = 0; i < theta; ++i) {
    typename GraphTy::vertex_type r = start(generator);
    result[i] = std::move(BFSOnRandomGraph(G, r, generator));
  }
  return result;
}

//! \brief Generate Random Reverse Reachability Sets.
//!
//! \tparam GraphTy The type of the garph.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy>
std::vector<std::unordered_set<typename GraphTy::vertex_type>> GenerateRRRSets(
    GraphTy &G, size_t theta, omp_parallel_tag &&tag) {
  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result(theta);

#pragma omp parallel
  {
    size_t size = omp_get_num_threads();
    size_t rank = omp_get_thread_num();

    trng::yarn2 generator;
    generator.split(size, rank);

    trng::uniform_int_dist start(0, G.num_nodes());

    for (size_t i = rank * theta / size; i < (rank + 1) * theta / size; ++i) {
      typename GraphTy::vertex_type r = start(generator);
      result[i] = std::move(BFSOnRandomGraph(G, r, generator));
    }
  }
  return result;
}

//! \brief Select k seeds starting from the a list of Random Reverse
//! Reachability Sets.
//!
//! \tparam GraphTy The graph type.
//!
//! \param G The original graph.
//! \param k The size of the seed set.
//! \param RRRsets A vector storing the Random Reverse Reachability sets.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.
template <typename GraphTy>
std::pair<size_t, std::unordered_set<typename GraphTy::vertex_type>>
FindMostInfluentialSet(
    GraphTy &G, size_t k,
    std::vector<std::unordered_set<typename GraphTy::vertex_type>> &RRRsets) {
  using vertex_type = typename GraphTy::vertex_type;
  using RRRMap = std::vector<std::unordered_set<vertex_type> *>;
  std::vector<RRRMap> hyperGraph(G.num_nodes());
  std::vector<size_t> vertexCoverage(G.num_nodes());

  for (auto &r : RRRsets) {
    for (auto v : r) {
      hyperGraph[v].push_back(&r);
    }
  }

  auto cmp = [](std::pair<vertex_type, size_t> &a,
                std::pair<vertex_type, size_t> &b) {
    return a.second < b.second;
  };
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, size_t>,
                          std::vector<std::pair<vertex_type, size_t>>,
                          decltype(cmp)>;

  priorityQueue queue(
      cmp, std::vector<std::pair<vertex_type, size_t>>(G.num_nodes()));
  for (vertex_type i = 0; i < G.num_nodes(); ++i) {
    vertexCoverage[i] = hyperGraph[i].size();
    queue.push(std::make_pair(i, vertexCoverage[i]));
  }

  std::unordered_set<typename GraphTy::vertex_type> result;
  size_t uncovered = RRRsets.size();

  while (result.size() < k && uncovered > 0) {
    auto element = queue.top();
    queue.pop();

    if (element.second > vertexCoverage[element.first]) {
      element.second = vertexCoverage[element.first];
      queue.push(element);
      continue;
    }

    uncovered -= vertexCoverage[element.first];

    for (auto rrrSetPtr : hyperGraph[element.first]) {
      for (auto v : *rrrSetPtr) {
        vertexCoverage[v] -= 1;
      }
    }

    result.insert(element.first);
  }

  return std::make_pair(RRRsets.size() - uncovered, result);
}

//! \brief Estimate the number of Random Reverse Reachability Sets to be
//! computed.
//!
//! \tparam GraphTy The graph type.
//! \tparam execution_tag Type tag to selecte the execution policy.
//!
//! \param G The original graph.
//! \param k The size of the seed set to be selected.
//! \param epsilon The approximation factor.
//! \param tag The execution policy tag.
//!
//! \return The number of Random Reverse Reachability sets to be computed.
template <typename GraphTy, typename execution_tag>
size_t ThetaEstimation(GraphTy &G, size_t k, double epsilon,
                       execution_tag &&tag) {
  auto start = std::chrono::high_resolution_clock::now();
  double kpt = KptEstimation(G, k, std::forward<execution_tag>(tag));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> exTime = end - start;
  spdlog::get("perf")->info("KptEstimation : {}ms", exTime.count());

  start = std::chrono::high_resolution_clock::now();
  // epsPrime is set according to the equation at the bottom of Section 4.1
  double epsPrime = 5 * cbrt((epsilon * epsilon) / (k + 1));

  // The following block implements the refinement algorithm (Algorithm 3)
  size_t thetaPrime = (2 + epsPrime) * G.num_nodes() * log(G.num_nodes()) /
                      (epsPrime * epsPrime * kpt);

  auto RR = GenerateRRRSets(G, thetaPrime, std::forward<execution_tag>(tag));

  auto seeds = FindMostInfluentialSet(G, k, RR);
  double f = double(seeds.first) / RR.size();
  double kptPrime = (f * G.num_nodes()) / (1 + epsPrime);

  // kpt now contains the best bound we were able to find after refinment.
  kpt = std::max(kpt, kptPrime);
  end = std::chrono::high_resolution_clock::now();
  exTime = end - start;
  spdlog::get("perf")->info("KptRefinement : {}ms", exTime.count());

  auto logBinomial = [](size_t n, size_t k) -> double {
    return n * log(n) - k * log(k) - (n - k) * log(n - k);
  };

  // Compute lambda from equation (4)
  double lambda = ((8 + 2 * epsilon) * G.num_nodes() *
                       (log(G.num_nodes()) + logBinomial(G.num_nodes(), k)) +
                   log(2.0)) /
                  (epsilon * epsilon);

  spdlog::get("perf")->trace("Kpt = {}, lambda = {}", kpt, lambda);
  // return theta according to equation (5)
  return ceil(lambda / kpt);
}

//! \brief The TIM+ algorithm for Influence Maximization.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam execution_tag The execution policy tag.
//!
//! \param G The original graph.
//! \param k The size of the seed set.
//! \param epsilon The approximation factor.
//! \param tag The execution policy tag.
//!
//! \return A set of vertices in the graph.
template <typename GraphTy, typename execution_tag>
std::unordered_set<typename GraphTy::vertex_type> TIM(const GraphTy &G,
                                                      size_t k, double epsilon,
                                                      execution_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;
  std::unordered_set<vertex_type> seedSet;

  auto theta = ThetaEstimation(G, k, epsilon, std::forward<execution_tag>(tag));
  spdlog::get("perf")->trace("theta = {}", theta);

  auto start = std::chrono::high_resolution_clock::now();
  auto RR = GenerateRRRSets(G, theta, std::forward<execution_tag>(tag));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> exTime = end - start;
  spdlog::get("perf")->info("Generate RRR : {}ms", exTime.count());

  start = std::chrono::high_resolution_clock::now();
  auto seeds = FindMostInfluentialSet(G, k, RR);
  end = std::chrono::high_resolution_clock::now();
  exTime = end - start;
  spdlog::get("perf")->info("FindMostInfluentialSet : {}ms", exTime.count());

  return seeds.second;
}

}  // namespace im

#endif /* IM_TIM_H */
