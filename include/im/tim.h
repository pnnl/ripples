//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_TIM_H
#define IM_TIM_H

#include <cassert>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iterator>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>

#include <omp.h>

#include "im/bfs.h"
#include "im/utility.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"
#include "trng/lcg64.hpp"

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
//! \tparam PRNGeneratorty The type of the random number generator.
//!
//! \param G The original graph.
//! \param k The size of the desired seed set.
//! \param generator The random numeber generator.
//! \param tag The execution policy tag.
//!
//! \return a lower bond of OPT computed with Algoirthm 2 of the original paper.
template <typename GraphTy, typename PRNGeneratorTy>
double KptEstimation(GraphTy &G, size_t k, PRNGeneratorTy &generator, sequential_tag &&tag) {
  // Compute KPT* according to Algorithm 2
  double KPTStar = 1;

  trng::uniform_int_dist root(0, G.num_nodes());

  for (size_t i = 1; i < log2(G.num_nodes()); ++i) {
    double sum = 0;
    size_t c_i =
        (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * (1ul << i);

    for (size_t j = 0; j < c_i; ++j) {
      // Pick a random vertex
      typename GraphTy::vertex_type v = root(generator[0]);

      double wr = WR(G, v, generator[0]);
      wr /= G.num_edges();
      // Equation (8) of the paper.
      sum += 1 - pow(1.0 - wr, k);
    }

    sum /= c_i;

    if (sum > (1.0 / (1ul << i))) {
      KPTStar = G.num_nodes() * sum / 2;
      break;
    }
  }

  return KPTStar;
}

//! \brief Estimate KPT.  Parallelization of Algorithm 2 in the original paper.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam PRNGeneratorty The type of the random number generator.
//!
//! \param G The original graph.
//! \param k The size of the desired seed set.
//! \param generator The random numeber generator.
//! \param tag The execution policy tag.
//!
//! \return a lower bond of OPT computed with Algoirthm 2 of the original paper.
template <typename GraphTy, typename PRNGeneratorTy>
double KptEstimation(GraphTy &G, size_t k, PRNGeneratorTy &generator, omp_parallel_tag &&tag) {
  double KPTStar = 1.0;

  for (size_t i = 2; i < G.num_nodes(); i <<= 1) {
    size_t c_i = (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * i;
    double sum = 0;

#pragma omp parallel reduction(+ : sum)
    {
      size_t size = omp_get_num_threads();
      size_t rank = omp_get_thread_num();

      trng::uniform_int_dist root(0, G.num_nodes());

      size_t chunk = c_i;

      for (size_t j = rank * chunk / size; j < (rank + 1) * chunk / size; ++j) {
        // Pick a random vertex
        typename GraphTy::vertex_type v = root(generator[rank]);

        double wr = WR(G, v, generator[rank]);
        wr /= G.num_edges();
        // Equation (8) of the paper.
        sum += 1 - pow(1.0 - wr, k);
      }
    }

    sum /= c_i;

    if (sum > (1.0 / i)) {
      KPTStar = G.num_nodes() * sum / 2;
      break;
    }
  }

  return KPTStar;
}

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam PRNGGeneratorTy The type of pseudo the random number generator.
//! \tparam execution_tag The execution tag.
//!
//! \param G The graph instance.
//! \param r The starting point for the exploration.
//! \param generator The pseudo random number generator.
//! \param result The RRR set
//! \param i The simulation number
//! \param HG The Hyper-Graph to build
template <typename GraphTy, typename PRNGeneratorTy, typename execution_tag>
void AddRRRSet(
    GraphTy &G, typename GraphTy::vertex_type r, PRNGeneratorTy &generator,
    std::vector<typename GraphTy::vertex_type> &result, size_t i,
    std::vector<std::deque<size_t>>& HG,
    execution_tag && tag) {
  using vertex_type = typename GraphTy::vertex_type;

  trng::uniform01_dist<float> value;

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);

  queue.push(r);
  visited[r] = true;
  result.push_back(r);

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    HG[v].push_back(i);
    for (auto u : G.in_neighbors(v)) {
      if (!visited[u.vertex] && value(generator) < u.weight) {
        queue.push(u.vertex);
        visited[u.vertex] = true;
        result.push_back(u.vertex);
      }
    }
  }
}

//! \brief Generate Random Reverse Reachability Sets.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param generator The random numeber generator.
//! \param tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy, typename PRNGeneratorTy>
std::pair<
  std::vector<std::vector<typename GraphTy::vertex_type>>,
  std::vector<std::deque<size_t>>>
GenerateRRRSets(
    GraphTy &G, size_t theta, PRNGeneratorTy &generator, sequential_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<std::vector<vertex_type>> rrrSets(theta);
  std::vector<std::deque<size_t>> HyperG(G.num_nodes());

  trng::uniform_int_dist start(0, G.num_nodes());

  for (size_t i = 0; i < theta; ++i) {
    typename GraphTy::vertex_type r = start(generator[0]);
    AddRRRSet(G, r, generator[0], rrrSets[i], i, HyperG, std::forward<sequential_tag>(tag));
  }
  return std::make_pair(std::forward<decltype(rrrSets)>(rrrSets),
                        std::forward<decltype(HyperG)>(HyperG));
}

void mergeHG(std::vector<std::deque<size_t>> &out, std::vector<std::deque<size_t>> &in) {
for (size_t i = 0; i < in.size(); ++i)
  out[i].insert(out[i].end(), in[i].begin(), in[i].end());
}

//! \brief Generate Random Reverse Reachability Sets.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param generator The random numeber generator.
//! \param tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy, typename PRNGeneratorTy>
std::pair<
  std::vector<std::vector<typename GraphTy::vertex_type>>,
  std::vector<std::deque<size_t>>>
GenerateRRRSets(
    GraphTy &G, size_t theta, PRNGeneratorTy &generator, omp_parallel_tag &&tag) {
  std::vector<std::vector<typename GraphTy::vertex_type>> rrrSets(theta);
  std::vector<std::deque<size_t>> HyperG(G.num_nodes());

#pragma omp declare reduction(MergeHyperGraph : std::vector<std::deque<size_t>> : mergeHG(omp_out, omp_in)) initializer(omp_priv = std::vector<std::deque<size_t>>(omp_orig.size()))
  
#pragma omp parallel reduction(MergeHyperGraph:HyperG)
  {
    size_t size = omp_get_num_threads();
    size_t rank = omp_get_thread_num();

    trng::uniform_int_dist start(0, G.num_nodes());

    for (size_t i = rank * theta / size; i < (rank + 1) * theta / size; ++i) {
      typename GraphTy::vertex_type r = start(generator[rank]);
      AddRRRSet(G, r, generator[rank], rrrSets[i], i, HyperG, std::forward<omp_parallel_tag>(tag));
    }
  }
  return std::make_pair(std::forward<decltype(rrrSets)>(rrrSets),
                        std::forward<decltype(HyperG)>(HyperG));
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
auto
FindMostInfluentialSet(
    GraphTy &G, size_t k,
    std::vector<std::vector<typename GraphTy::vertex_type>> &RRRsets,
    std::vector<std::deque<size_t>> &hyperGraph) {
  using vertex_type = typename GraphTy::vertex_type;

  std::vector<size_t> vertexCoverage(G.num_nodes());

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

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  std::vector<bool> removed(RRRsets.size(), false);
  size_t uncovered = RRRsets.size();

  while (result.size() < k && uncovered != 0) {
    auto element = queue.top();
    queue.pop();

    if (element.second > vertexCoverage[element.first]) {
      element.second = vertexCoverage[element.first];
      queue.push(element);
      continue;
    }

    uncovered -= element.second;

    for (auto  rrrSetId : hyperGraph[element.first]) {
      if (removed[rrrSetId]) continue;

      for (auto v : RRRsets[rrrSetId]) {
        vertexCoverage[v] -= 1;
      }

      removed[rrrSetId] = true;
    }

    result.push_back(element.first);
  }

  return std::make_pair(RRRsets.size() - uncovered, result);
}

//! \brief Estimate the number of Random Reverse Reachability Sets to be
//! computed.
//!
//! \tparam GraphTy The graph type.
//! \tparam PRNGeneratorty The type of the Random Number Generator.
//! \tparam execution_tag Type tag to selecte the execution policy.
//!
//! \param G The original graph.
//! \param k The size of the seed set to be selected.
//! \param epsilon The approximation factor.
//! \param generator The random number generator.
//! \param tag The execution policy tag.
//!
//! \return The number of Random Reverse Reachability sets to be computed.
template <typename GraphTy, typename PRNGeneratorTy, typename execution_tag>
size_t ThetaEstimation(GraphTy &G, size_t k, double epsilon, PRNGeneratorTy &generator,
                       ExecutionRecord &R, execution_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;

  auto start = std::chrono::high_resolution_clock::now();
  double kpt = KptEstimation(G, k, generator, std::forward<execution_tag>(tag));
  auto end = std::chrono::high_resolution_clock::now();
  R.KptEstimation = end - start;

  start = std::chrono::high_resolution_clock::now();
  // epsPrime is set according to the equation at the bottom of Section 4.1
  double epsPrime = 5 * cbrt((epsilon * epsilon) / (k + 1));

  // The following block implements the refinement algorithm (Algorithm 3)
  size_t thetaPrime = (2 + epsPrime) * G.num_nodes() * log(G.num_nodes()) /
                      (epsPrime * epsPrime * kpt);

  std::vector<std::vector<vertex_type>> RR;
  std::vector<std::deque<size_t>> HyperG;
  std::tie(RR, HyperG) = std::move(GenerateRRRSets(G, thetaPrime, generator, std::forward<execution_tag>(tag)));

  auto seeds = FindMostInfluentialSet(G, k, RR, HyperG);
  double f = double(seeds.first) / RR.size();
  double kptPrime = (f * G.num_nodes()) / (1 + epsPrime);

  // kpt now contains the best bound we were able to find after refinment.
  kpt = std::max(kpt, kptPrime);
  end = std::chrono::high_resolution_clock::now();
  R.KptRefinement = end - start;

  auto logBinomial = [](size_t n, size_t k) -> double {
    return n * log(n) - k * log(k) - (n - k) * log(n - k);
  };

  // Compute lambda from equation (4)
  double lambda = ((8 + 2 * epsilon) * G.num_nodes() *
                       (log(G.num_nodes()) + logBinomial(G.num_nodes(), k)) +
                   log(2.0)) /
                  (epsilon * epsilon);

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
auto TIM(const GraphTy &G, size_t k, double epsilon, execution_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;
  ExecutionRecord Record;

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


  auto theta = ThetaEstimation(G, k, epsilon, generator, Record, std::forward<execution_tag>(tag));

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<vertex_type>> RR;
  std::vector<std::deque<size_t>> HyperG;
  std::tie(RR, HyperG) = std::move(GenerateRRRSets(G, theta, generator, std::forward<execution_tag>(tag)));
  auto end = std::chrono::high_resolution_clock::now();
  Record.GenerateRRRSets = end - start;

  start = std::chrono::high_resolution_clock::now();
  auto seeds = FindMostInfluentialSet(G, k, RR, HyperG);
  end = std::chrono::high_resolution_clock::now();
  Record.FindMostInfluentialSet = end - start;

  return std::make_pair(seeds.second, Record);
}

}  // namespace im

#endif /* IM_TIM_H */
