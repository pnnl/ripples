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

#include "im/bfs.h"
#include "im/utility.h"

#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "im/diffusion_simulation.h"

namespace im {

struct TIMExecutionRecord {
  size_t NumThreads;
  std::chrono::duration<double, std::milli> KptEstimation;
  std::chrono::duration<double, std::milli> KptRefinement;
  std::chrono::duration<double, std::milli> GenerateRRRSets;
  std::chrono::duration<double, std::milli> FindMostInfluentialSet;
  std::chrono::duration<double, std::milli> Total;

  template <typename Ostream>
  friend Ostream &operator<<(Ostream &O, const TIMExecutionRecord &R) {
    O << "{ "
      << "\"NumThreads\" : " << R.NumThreads << ", "
      << "\"KptEstimation\" : " << R.KptEstimation.count() << ", "
      << "\"KptRefinement\" : " << R.KptRefinement.count() << ", "
      << "\"GenerateRRRSets\" : " << R.GenerateRRRSets.count() << ", "
      << "\"FindMostInfluentialSet\" : " << R.FindMostInfluentialSet.count()
      << ", "
      << "\"Total\" : " << R.Total.count() << " }";
    return O;
  }
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

using HyperGraphTy = std::vector<std::vector<size_t>>;

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
               PRNGeneratorTy &generator,
               std::vector<typename GraphTy::vertex_type> &result, size_t i,
               HyperGraphTy &HG, diff_model_tag &&tag) {
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
std::pair<std::vector<std::vector<typename GraphTy::vertex_type>>,
          HyperGraphTy>
GenerateRRRSets(GraphTy &G, size_t theta, PRNGeneratorTy &generator,
                diff_model_tag &&model_tag, sequential_tag &&) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<std::vector<vertex_type>> rrrSets(theta);
  HyperGraphTy HyperG(G.num_nodes());

  trng::uniform_int_dist start(0, G.num_nodes());

  for (size_t i = 0; i < theta; ++i) {
    typename GraphTy::vertex_type r = start(generator[0]);
    AddRRRSet(G, r, generator[0], rrrSets[i], i, HyperG,
              std::forward<diff_model_tag>(model_tag));
  }
  return std::make_pair(std::forward<decltype(rrrSets)>(rrrSets),
                        std::forward<decltype(HyperG)>(HyperG));
}

void mergeHG(HyperGraphTy &out,
             HyperGraphTy &in) {
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
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
std::pair<std::vector<std::vector<typename GraphTy::vertex_type>>,
          HyperGraphTy>
GenerateRRRSets(GraphTy &G, size_t theta, PRNGeneratorTy &generator,
                diff_model_tag &&model_tag, omp_parallel_tag &&) {
  std::vector<std::vector<typename GraphTy::vertex_type>> rrrSets(theta);
  HyperGraphTy HyperG(G.num_nodes());


#pragma omp declare reduction(MergeHyperGraph : HyperGraphTy : mergeHG(omp_out, omp_in)) initializer(omp_priv = HyperGraphTy(omp_orig.size()))

#pragma omp parallel reduction(MergeHyperGraph : HyperG)
  {
    size_t rank = omp_get_thread_num();
    trng::uniform_int_dist start(0, G.num_nodes());

    #pragma omp for schedule(guided)
    for (size_t i = 0; i < theta; ++i) {
      typename GraphTy::vertex_type r = start(generator[rank]);
      AddRRRSet(G, r, generator[rank], rrrSets[i], i, HyperG,
                std::forward<diff_model_tag>(model_tag));
    }
  }
  return {rrrSets, HyperG};
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
auto FindMostInfluentialSet(
    const GraphTy &G, size_t k,
    const std::vector<std::vector<typename GraphTy::vertex_type>> &RRRsets,
    const HyperGraphTy &hyperGraph,
    execution_tag&&) {
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

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
    #pragma omp parallel for
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      vertexCoverage[i] = hyperGraph[i].size();
    }
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      queue.push(std::make_pair(i, vertexCoverage[i]));
    }
  } else {
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      vertexCoverage[i] = hyperGraph[i].size();
      queue.push(std::make_pair(i, vertexCoverage[i]));
    }
  }

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
      #pragma omp declare reduction(vec_cumulative_count : std::vector<size_t> : \
                                    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<size_t>())) \
        initializer(omp_priv = omp_orig)

      #pragma omp declare reduction(vec_removed_flag : std::vector<char> : \
                                    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::logical_or<char>())) \
        initializer(omp_priv = omp_orig)

      std::vector<size_t> counters(G.num_nodes(), 0);
      #pragma omp parallel
      {
        #pragma omp for reduction(vec_cumulative_count : counters) reduction(vec_removed_flag : removed)
        for (size_t i = 0; i < hyperGraph[element.first].size(); ++i) {
          auto rrrSetId = hyperGraph[element.first][i];
          if (removed[rrrSetId] != 0) continue;

          for (auto v : RRRsets[rrrSetId]) {
            counters[v] += 1;
          }

          removed[rrrSetId] = true;
        }

        #pragma omp for simd
        for (size_t i = 0; i < counters.size(); ++i) {
          vertexCoverage[i] -= counters[i];
        }
      }
    } else {
      for (auto rrrSetId : hyperGraph[element.first]) {
        if (removed[rrrSetId] != 0) continue;

        for (auto v : RRRsets[rrrSetId]) {
          vertexCoverage[v] -= 1;
        }

        removed[rrrSetId] = 1;
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

  std::vector<std::vector<vertex_type>> RR;
  HyperGraphTy HyperG;
  std::tie(RR, HyperG) = GenerateRRRSets(
      G, thetaPrime, generator, std::forward<diff_model_tag>(model_tag),
      std::forward<execution_tag>(ex_tag));

  auto seeds = FindMostInfluentialSet(G, k, RR, HyperG, std::forward<execution_tag>(ex_tag));
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

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<vertex_type>> RR;
  HyperGraphTy HyperG;
  std::tie(RR, HyperG) = GenerateRRRSets(
      G, theta, generator, std::forward<diff_model_tag>(model_tag),
      std::forward<execution_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();
  Record.GenerateRRRSets = end - start;

  start = std::chrono::high_resolution_clock::now();
  auto seeds = FindMostInfluentialSet(G, k, RR, HyperG, std::forward<execution_tag>(ex_tag));
  end = std::chrono::high_resolution_clock::now();
  Record.FindMostInfluentialSet = end - start;

  return std::make_pair(seeds.second, Record);
}

}  // namespace im

#endif /* IM_TIM_H */
