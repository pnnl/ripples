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
#include <utility>

#include <omp.h>

#include "im/bfs.h"
#include "im/utility.h"

#include "spdlog/spdlog.h"
#include "trng/yarn2.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"


namespace im {

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
      if (!visited[u.vertex] &&
          value(generator) < u.weight) {
        visited[u.vertex] = true;
        queue.push(u.vertex);
      }
    }
  }

  return wr;
}

template <typename GraphTy>
double KptEstimation(GraphTy &G, size_t k, double epsilon, sequential_tag&&) {
  // Compute KPT* according to Algorithm 2
  double KPTStar = 1;

  trng::yarn2 generator;

  trng::uniform_int_dist root(0, G.num_nodes());

  for (size_t i = 1; i < log2(G.num_nodes()); ++i) {
    double sum = 0;
    size_t c_i = (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * (1ul << i);

    for (size_t j = 0; j < c_i; ++j) {
      // Pick a random vertex
      typename GraphTy::vertex_type v = root(generator);

      double wr = WR(G, v, generator);
      wr /= G.num_edges();
      // Equation (8) of the paper.
      sum += 1 - pow(1.0 - wr, k);
    }

    sum /= c_i;

    spdlog::get("perf")->debug("c_i = {}, sum = {} < {}", c_i, sum, (1.0 / (1ul << i)));
    if (sum > (1.0 / (1ul << i))) {
      KPTStar = G.num_nodes() * sum / 2;
      spdlog::get("perf")->debug("KPTStar = {}, sum = {}, c_i = {}", KPTStar, sum, c_i);
      break;
    }
  }

  return KPTStar;
}

template <typename GraphTy>
double KptEstimation(GraphTy &G, size_t k, double epsilon, omp_parallel_tag &&) {
  double KPTStar = 1.0;

  for (size_t i = 2; i < G.num_nodes(); i <<= 1) {
    size_t c_i = (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * i;
    double sum = 0;

#pragma omp parallel reduction(+:sum)
    {
      size_t size = omp_get_num_threads();
      size_t rank = omp_get_thread_num();

      spdlog::get("console")->info("num threads = {}, rank = {}", size, rank);

      trng::yarn2 generator;
      generator.split(size, rank);

      trng::uniform_int_dist root(0, G.num_nodes());

      size_t chunk = c_i;

      for (size_t j = rank * chunk/size; j < (rank + 1) * chunk/size; ++j) {
        // Pick a random vertex
        typename GraphTy::vertex_type v = root(generator);

        double wr = WR(G, v, generator);
        wr /= G.num_edges();
        // Equation (8) of the paper.
        sum += 1 - pow(1.0 - wr, k);
      }
    }

    sum /= c_i;

    spdlog::get("perf")->debug("c_i = {}, sum = {} < {}", c_i, sum, (1.0 / i));
    if (sum > (1.0 / i)) {
      KPTStar = G.num_nodes() * sum / 2;
      spdlog::get("perf")->debug("KPTStar = {}, sum = {}, c_i = {}", KPTStar, sum, c_i);
      break;
    }
  }

  return KPTStar;
}


template <typename GraphTy>
std::vector<std::unordered_set<typename GraphTy::vertex_type>>
GenerateRRRSets(GraphTy &G, size_t theta, sequential_tag &&tag) {
  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result (theta);

  trng::yarn2 generator;
  trng::uniform_int_dist start(0, G.num_nodes());

  for (size_t i = 0; i < theta; ++i) {
    typename GraphTy::vertex_type r = start(generator);
    result[i] = std::move(BFSOnRandomGraph(G, r, generator));
  }
  return result;
}


template <typename GraphTy>
std::vector<std::unordered_set<typename GraphTy::vertex_type>>
GenerateRRRSets(GraphTy &G, size_t theta, omp_parallel_tag &&tag) {
  std::vector<std::unordered_set<typename GraphTy::vertex_type>> result (theta);

#pragma omp parallel
  {
      size_t size = omp_get_num_threads();
      size_t rank = omp_get_thread_num();

      trng::yarn2 generator;
      generator.split(size, rank);

      trng::uniform_int_dist start(0, G.num_nodes());

      for (size_t i = rank * theta/size; i < (rank + 1) * theta / size; ++i) {
        typename GraphTy::vertex_type r = start(generator);
        result[i] = std::move(BFSOnRandomGraph(G, r, generator));
      }
  }
  return result;
}


template <typename GraphTy>
std::pair<size_t, std::unordered_set<typename GraphTy::vertex_type>>
FindMostInfluentialSet(
    GraphTy &G, size_t k,
    std::vector<std::unordered_set<typename GraphTy::vertex_type>> &RRRsets) {
  using vertex_type = typename GraphTy::vertex_type;
  using RRRMap = std::vector<std::unordered_set<vertex_type> *>;
  std::vector<RRRMap> hyperGraph(G.num_nodes());
  std::vector<size_t> vertexCoverage(G.num_nodes());

  for (auto & r : RRRsets) {
    for (auto v : r) {
      hyperGraph[v].push_back(&r);
    }
  }

#if 1
  auto cmp = [](std::pair<vertex_type, size_t> &a, std::pair<vertex_type, size_t>& b) {
               return a.second < b.second;
             };
  using priorityQueue = std::priority_queue<
    std::pair<vertex_type, size_t>,
    std::vector<std::pair<vertex_type, size_t>>,
    decltype(cmp)>;

  priorityQueue queue(cmp, std::vector<std::pair<vertex_type, size_t>>(G.num_nodes()));
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
#else
  std::unordered_set<typename GraphTy::vertex_type> result;
  size_t uncovered = RRRsets.size();

  while (result.size() < k && uncovered > 0) {
    typename GraphTy::vertex_type maxVertex = 0;
    for (typename GraphTy::vertex_type v = 0;
         v < G.num_nodes() && result.find(v) == result.end(); ++v) {
      if (vertexCoverage[maxVertex] < vertexCoverage[v]) {
        maxVertex = v;
      }
    }

    for (auto rrrSetPtr : hyperGraph[maxVertex]) {
      for (auto v : *rrrSetPtr) {
        vertexCoverage[v] -= 1;
      }
    }

    uncovered -= vertexCoverage[maxVertex];
    vertexCoverage[maxVertex] = 0;
    result.insert(maxVertex);
  }
#endif

  return std::make_pair(RRRsets.size() - uncovered, result);
}

template <typename GraphTy, typename execution_tag>
size_t ThetaEstimation(GraphTy &G, size_t k, double epsilon, execution_tag &&tag) {
  double kpt = KptEstimation(G, k, epsilon, std::forward<execution_tag>(tag));

  double epsPrime = 5 * cbrt((epsilon * epsilon) / (k + 1));
  size_t thetaPrime = (2 + epsPrime) * G.num_nodes() * log(G.num_nodes()) / (epsPrime * epsPrime * kpt);

  auto RR = GenerateRRRSets(G, thetaPrime, std::forward<execution_tag>(tag));

  auto seeds = FindMostInfluentialSet(G, k, RR);
  double f = double(seeds.first) / RR.size();
  double kptPrime = (f * G.num_nodes()) / (1 + epsPrime);

  kpt = std::max(kpt, kptPrime);
  spdlog::get("console")->info("kpt = {}", kpt);

  // Compute lambda from equation (4)
  auto logBinomial = [](size_t n, size_t k) -> double {
    return n * log(n) - k * log(k) - (n - k) * log (n - k);
  };
  double lambda =
      ((8 + 2 * epsilon) * G.num_nodes() *
      (log(G.num_nodes()) +
       logBinomial(G.num_nodes(), k)) +
       log(2.0)) / (epsilon * epsilon);
  spdlog::get("console")->info("lambda = {}", lambda);

  return ceil(lambda / kpt);
}

template <typename GraphTy, typename execution_tag>
std::unordered_set<typename GraphTy::vertex_type>
TIM(const GraphTy &G, size_t k, double epsilon, execution_tag&& tag) {
  using vertex_type = typename GraphTy::vertex_type;
  std::unordered_set<vertex_type> seedSet;

  auto theta =
      ThetaEstimation(G, k, epsilon, std::forward<execution_tag>(tag));

  spdlog::get("console")->info("theta = {}", theta);

  auto RR = GenerateRRRSets(G, theta, std::forward<execution_tag>(tag));

  spdlog::get("console")->info("RRSize = {}", RR.size());

  auto seeds = FindMostInfluentialSet(G, k, RR);

  return seeds.second;
}

}  // namespace im

#endif /* IM_TIM_H */
