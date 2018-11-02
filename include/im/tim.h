//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_TIM_H
#define IM_TIM_H

#include <algorithm>
#include <cassert>
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

#include "im/diffusion_simulation.h"
#include "im/utility.h"

#include "CLI11/CLI11.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

namespace im {

//! \brief The configuration data structure for the TIM+ algorithm.
struct TIMConfiguration {
  size_t k{10};                      //!< The size of the seedset
  double epsilon{0.50};              //!< The epsilon of the IM algorithm
  bool parallel{false};              //!< The sequential vs parallel algorithm
  std::string diffusionModel{"IC"};  //!< The diffusion model to use.

  void addCmdOptions(CLI::App &app) {
    app.add_option("-k,--seed-set-size", k, "The size of the seed set.")
        ->required()
        ->group("Algorithm Options");
    app.add_option("-e,--epsilon", epsilon, "The size of the seed set.")
        ->required()
        ->group("Algorithm Options");
    app.add_flag("-p,--parallel", parallel,
                 "Trigger the parallel implementation")
        ->group("Algorithm Options");
    app.add_option("-d,--diffusion-model", diffusionModel,
                   "The diffusion model to be used (LT|IC)")
        ->required()
        ->group("Algorithm Options");
  }
};


struct TIMExecutionRecord {
  size_t NumThreads;
  size_t Theta;
  std::chrono::duration<double, std::milli> KptEstimation;
  std::chrono::duration<double, std::milli> KptRefinement;
  std::chrono::duration<double, std::milli> GenerateRRRSets;
  std::chrono::duration<double, std::milli> FindMostInfluentialSet;
  std::chrono::duration<double, std::milli> Total;
};


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
template <typename GraphTy, typename PRNG, typename diff_model_tag>
size_t WR(GraphTy &G, typename GraphTy::vertex_type r, PRNG &generator,
          diff_model_tag &&) {
  using vertex_type = typename GraphTy::vertex_type;

  trng::uniform01_dist<float> value;

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);

  queue.push(r);
  visited[r] = true;
  size_t wr = 0;

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    wr += G.in_degree(v);

    if (std::is_same<diff_model_tag, im::independent_cascade_tag>::value) {
      for (auto u : G.in_neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
        }
      }
    } else if (std::is_same<diff_model_tag, im::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.in_neighbors(v)) {
        threshold -= u.weight;

        if (threshold > 0) continue;

        if (visited[u.vertex]) break;

        queue.push(u.vertex);
        visited[u.vertex] = true;
      }
    } else {
      throw;
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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
double KptEstimation(GraphTy &G, size_t k, PRNGeneratorTy &generator,
                     diff_model_tag &&model_tag, sequential_tag &&) {
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

      double wr =
          WR(G, v, generator[0], std::forward<diff_model_tag>(model_tag));
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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
double KptEstimation(GraphTy &G, size_t k, PRNGeneratorTy &generator,
                     diff_model_tag &&model_tag, omp_parallel_tag &&) {
  double KPTStar = 1.0;

  for (size_t i = 2; i < G.num_nodes(); i <<= 1) {
    size_t c_i = (6 * log(G.num_nodes()) + 6 * log(log2(G.num_nodes()))) * i;
    double sum = 0;

#pragma omp parallel reduction(+ : sum)
    {
      size_t rank = omp_get_thread_num();
      trng::uniform_int_dist root(0, G.num_nodes());

#pragma omp for schedule(guided)
      for (size_t j = 0; j < c_i; ++j) {
        // Pick a random vertex
        typename GraphTy::vertex_type v = root(generator[rank]);

        double wr =
            WR(G, v, generator[rank], std::forward<diff_model_tag>(model_tag));
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

template <typename GraphTy>
using RRRset = std::vector<typename GraphTy::vertex_type>;

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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
void AddRRRSet(GraphTy &G, typename GraphTy::vertex_type r,
               PRNGeneratorTy &generator, RRRset<GraphTy> &result,
               diff_model_tag &&tag) {
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

    if (std::is_same<diff_model_tag, im::independent_cascade_tag>::value) {
      for (auto u : G.in_neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
        }
      }
    } else if (std::is_same<diff_model_tag, im::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.in_neighbors(v)) {
        threshold -= u.weight;

        if (threshold > 0) continue;

        if (visited[u.vertex]) break;

        queue.push(u.vertex);
        visited[u.vertex] = true;
        result.push_back(u.vertex);
      }
    } else {
      throw;
    }
  }

  std::stable_sort(result.begin(), result.end());
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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
std::vector<RRRset<GraphTy>> GenerateRRRSets(GraphTy &G, size_t theta,
                                             PRNGeneratorTy &generator,
                                             diff_model_tag &&model_tag,
                                             sequential_tag &&) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<RRRset<GraphTy>> rrrSets(theta);

  trng::uniform_int_dist start(0, G.num_nodes());

  for (size_t i = 0; i < theta; ++i) {
    typename GraphTy::vertex_type r = start(generator[0]);
    AddRRRSet(G, r, generator[0], rrrSets[i],
              std::forward<diff_model_tag>(model_tag));
  }
  return rrrSets;
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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
std::vector<RRRset<GraphTy>> GenerateRRRSets(GraphTy &G, size_t theta,
                                             PRNGeneratorTy &generator,
                                             diff_model_tag &&model_tag,
                                             omp_parallel_tag &&) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<RRRset<GraphTy>> rrrSets(theta);

#pragma omp parallel
  {
    size_t rank = omp_get_thread_num();
    trng::uniform_int_dist start(0, G.num_nodes());

#pragma omp for schedule(guided)
    for (size_t i = 0; i < theta; ++i) {
      typename GraphTy::vertex_type r = start(generator[rank]);
      AddRRRSet(G, r, generator[rank], rrrSets[i],
                std::forward<diff_model_tag>(model_tag));
    }
  }

  return rrrSets;
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
template <typename GraphTy, typename execution_tag>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            const std::vector<RRRset<GraphTy>> &RRRsets,
                            execution_tag &&) {
  using vertex_type = typename GraphTy::vertex_type;

  std::vector<uint32_t> vertexCoverage(G.num_nodes(), 0);

  auto cmp = [](std::pair<vertex_type, size_t> &a,
                std::pair<vertex_type, size_t> &b) {
    return a.second < b.second;
  };
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, size_t>,
                          std::vector<std::pair<vertex_type, size_t>>,
                          decltype(cmp)>;

  std::vector<std::pair<vertex_type, size_t>> queue_storage(G.num_nodes());

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp parallel
    {
      size_t threadnum = omp_get_thread_num(),
             numthreads = omp_get_num_threads();
      vertex_type low = G.num_nodes() * threadnum / numthreads,
                  high = G.num_nodes() * (threadnum + 1) / numthreads;

      for (size_t i = 0; i < RRRsets.size(); ++i) {
        auto begin =
            std::lower_bound(RRRsets[i].begin(), RRRsets[i].end(), low);
        auto end =
            std::upper_bound(RRRsets[i].begin(), RRRsets[i].end(), high - 1);
        std::for_each(begin, end,
                      [&](const vertex_type v) { vertexCoverage[v] += 1; });
      }
    }
#pragma omp parallel for
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      queue_storage[i] = std::make_pair(i, vertexCoverage[i]);
    }
  } else {
    for (size_t i = 0; i < RRRsets.size(); ++i) {
      std::for_each(RRRsets[i].begin(), RRRsets[i].end(),
                    [&](const vertex_type v) { vertexCoverage[v] += 1; });
    }
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      queue_storage[i] = std::make_pair(i, vertexCoverage[i]);
    }
  }

  priorityQueue queue(cmp, std::move(queue_storage));

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  std::vector<char> removed(RRRsets.size(), false);
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

    if (std::is_same<execution_tag, omp_parallel_tag>::value) {
      for (size_t i = 0; i < RRRsets.size(); ++i) {
        if (removed[i]) continue;

        if (std::binary_search(RRRsets[i].begin(), RRRsets[i].end(),
                               element.first)) {
          removed[i] = true;
#pragma omp parallel for
          for (size_t j = 0; j < RRRsets[i].size(); ++j) {
            vertexCoverage[RRRsets[i][j]] -= 1;
          }
        }
      }
    } else {
      for (size_t i = 0; i < RRRsets.size(); ++i) {
        if (removed[i]) continue;

        if (std::binary_search(RRRsets[i].begin(), RRRsets[i].end(),
                               element.first)) {
          removed[i] = true;
          for (size_t j = 0; j < RRRsets[i].size(); ++j) {
            vertexCoverage[RRRsets[i][j]] -= 1;
          }
        }
      }
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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag,
          typename execution_tag>
size_t ThetaEstimation(GraphTy &G, size_t k, double epsilon,
                       PRNGeneratorTy &generator, TIMExecutionRecord &R,
                       diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  auto start = std::chrono::high_resolution_clock::now();
  double kpt =
      KptEstimation(G, k, generator, std::forward<diff_model_tag>(model_tag),
                    std::forward<execution_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();
  R.KptEstimation = end - start;

  start = std::chrono::high_resolution_clock::now();
  // epsPrime is set according to the equation at the bottom of Section 4.1
  double epsPrime = 5 * cbrt((epsilon * epsilon) / (k + 1));

  // The following block implements the refinement algorithm (Algorithm 3)
  size_t thetaPrime = (2 + epsPrime) * G.num_nodes() * log(G.num_nodes()) /
                      (epsPrime * epsPrime * kpt);

  auto RR = GenerateRRRSets(G, thetaPrime, generator,
                            std::forward<diff_model_tag>(model_tag),
                            std::forward<execution_tag>(ex_tag));

  auto seeds =
      FindMostInfluentialSet(G, k, RR, std::forward<execution_tag>(ex_tag));
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
template <typename GraphTy, typename PRNG, typename diff_model_tag,
          typename execution_tag>
auto TIM(const GraphTy &G, size_t k, double epsilon, PRNG &gen,
         diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  TIMExecutionRecord Record;

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

  auto theta = ThetaEstimation(G, k, epsilon, generator, Record,
                               std::forward<diff_model_tag>(model_tag),
                               std::forward<execution_tag>(ex_tag));

  Record.Theta = theta;

  auto start = std::chrono::high_resolution_clock::now();
  auto RR = GenerateRRRSets(G, theta, generator,
                            std::forward<diff_model_tag>(model_tag),
                            std::forward<execution_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();
  Record.GenerateRRRSets = end - start;

  start = std::chrono::high_resolution_clock::now();
  auto seeds =
      FindMostInfluentialSet(G, k, RR, std::forward<execution_tag>(ex_tag));
  end = std::chrono::high_resolution_clock::now();
  Record.FindMostInfluentialSet = end - start;

  return std::make_pair(seeds.second, Record);
}

}  // namespace im

#endif /* IM_TIM_H */
