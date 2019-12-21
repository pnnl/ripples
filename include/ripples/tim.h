//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
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

#ifndef RIPPLES_TIM_H
#define RIPPLES_TIM_H

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

#include "ripples/diffusion_simulation.h"
#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/utility.h"

#include "CLI/CLI.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

namespace ripples {

//! \brief The configuration data structure for the TIM+ algorithm.
struct TIMConfiguration : public AlgorithmConfiguration {
  double epsilon{0.50};  //!< The epsilon of the IM algorithm

  //! \brief Add command line options to configure TIM+.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    AlgorithmConfiguration::addCmdOptions(app);
    app.add_option("-e,--epsilon", epsilon, "The size of the seed set.")
        ->required()
        ->group("Algorithm Options");
  }
};

//! TIM+ execution record.
struct TIMExecutionRecord {
  //! Number of threads used during the execution.
  size_t NumThreads;
  //! Number of RRR sets generated.
  size_t Theta;
  //! Execution time of the Kpt Estimation phase.
  std::chrono::duration<double, std::milli> KptEstimation;
  //! Execution time of the Kpt Refinement phase.
  std::chrono::duration<double, std::milli> KptRefinement;
  //! Execution time of the RRR sets generation phase.
  std::chrono::duration<double, std::milli> GenerateRRRSets;
  //! Execution time of the maximum coverage phase.
  std::chrono::duration<double, std::milli> FindMostInfluentialSet;
  //! Total execution time.
  std::chrono::duration<double, std::milli> Total;
};

//! \brief Compute the number of elements in the RRR set starting at r.
//!
//! \tparam GraphTy The type of the Graph.
//! \tparam PNRG The type of the random number generator.
//! \tparam diff_model_tag The Type-Tag selecting the diffusion model.
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

    wr += G.degree(v);

    if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
      for (auto u : G.neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
        }
      }
    } else if (std::is_same<diff_model_tag,
                            ripples::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.neighbors(v)) {
        threshold -= u.weight;

        if (threshold > 0) continue;

        if (!visited[u.vertex]) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          break;
        }
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
//! \tparam diff_model_tag The Type-Tag selecting the diffusion model.
//!
//! \param G The original graph.
//! \param k The size of the desired seed set.
//! \param generator The random numeber generator.
//! \param model_tag The diffusion model to be used.
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
//! \tparam diff_model_tag The Type-Tag selecting the diffusion model.
//!
//! \param G The original graph.
//! \param k The size of the desired seed set.
//! \param generator The random numeber generator.
//! \param model_tag The diffusion model to use.
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

//! \brief Estimate the number of Random Reverse Reachability Sets to be
//! computed.
//!
//! \tparam GraphTy The graph type.
//! \tparam PRNGeneratorty The type of the Random Number Generator.
//! \tparam diff_model_tag The Type-Tag selecting the diffusion model.
//! \tparam execution_tag The Type-Tag to selecte the execution policy.
//!
//! \param G The original graph.
//! \param k The size of the seed set to be selected.
//! \param epsilon The approximation factor.
//! \param generator The random number generator.
//! \param R The execution record.
//! \param model_tag The diffusion model to use.
//! \param ex_tag The execution policy to use.
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

  std::vector<RRRset<GraphTy>> RR(thetaPrime);
  GenerateRRRSets(G, generator, RR.begin(), RR.end(),
                  std::forward<diff_model_tag>(model_tag),
                  std::forward<execution_tag>(ex_tag));

  auto seeds =
      FindMostInfluentialSet(G, k, RR, std::forward<execution_tag>(ex_tag));
  double f = double(seeds.first);
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
//! \tparam diff_model_tag The Type-Tag selecting the diffusion model.
//! \tparam execution_tag The execution policy tag.
//!
//! \param G The original graph.
//! \param k The size of the seed set.
//! \param epsilon The approximation factor.
//! \param gen A parallel random number generator.
//! \param model_tag The diffusion model to use.
//! \param ex_tag The execution policy to use.
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
  std::vector<RRRset<GraphTy>> RR(theta);

  GenerateRRRSets(G, generator, RR.begin(), RR.end(),
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

}  // namespace ripples

#endif /* RIPPLES_TIM_H */
