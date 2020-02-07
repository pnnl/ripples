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

#ifndef RIPPLES_HILL_CLIMBING_H
#define RIPPLES_HILL_CLIMBING_H

#include <algorithm>
#include <chrono>
#include <queue>
#include <type_traits>
#include <vector>

#include "nlohmann/json.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/hill_climbing_engine.h"

#include "omp.h"

namespace ripples {

//! The Hill Climbing Algorithm configuration descriptor.
struct HillClimbingConfiguration : public AlgorithmConfiguration {
  size_t samples{10000};

  //! \brief Add command line options to configure the Hill Climbing Algorithm.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    AlgorithmConfiguration::addCmdOptions(app);
    app.add_option(
        "--samples", samples,
        "The number of samples used in the Hill Climbing Algorithm.");
  }
};

//! The Hill Climbing Execution Record.
struct HillClimbingExecutionRecord {
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  using ex_time_ns = std::chrono::nanoseconds;

  //! Number of threads used during the execution.
  size_t NumThreads;
  //! Sampling time.
  ex_time_ms Sampling;
  //! Seed Selection time.
  ex_time_ms SeedSelection;
  //! Total execution time.
  ex_time_ms Total;
};

template <typename GraphTy, typename GeneratorTy, typename diff_model_tag>
auto SampleFrom(GraphTy &G, std::size_t num_samples, GeneratorTy &gen,
                HillClimbingExecutionRecord &record,
                diff_model_tag &&diff_model) {
  using vertex_type = typename GraphTy::vertex_type;
  using edge_mask = std::vector<bool>;
  std::vector<edge_mask> samples(num_samples,
                                 std::vector<bool>(G.num_edges(), false));
  auto start = std::chrono::high_resolution_clock::now();
  size_t num_threads = 1;
#pragma omp single
  { num_threads = omp_get_max_threads(); }

  using iterator_type = typename std::vector<edge_mask>::iterator;
  SamplingEngine<GraphTy, iterator_type, GeneratorTy, diff_model_tag> SE(
      G, gen, num_threads, 0);
  SE.exec(samples.begin(), samples.end());
  auto end = std::chrono::high_resolution_clock::now();
  record.Sampling = end - start;
  return samples;
}

namespace {
template <typename GraphTy, typename GraphMaskTy, typename Itr>
size_t BFS(GraphTy &G, GraphMaskTy &M, Itr b, Itr e,
           std::vector<bool> &visited) {
  using vertex_type = typename GraphTy::vertex_type;

  std::queue<vertex_type> queue;
  for (; b != e; ++b) {
    queue.push(*b);
  }

  while (!queue.empty()) {
    vertex_type u = queue.front();
    queue.pop();

    size_t edge_number =
        std::distance(G.neighbors(0).begin(), G.neighbors(u).begin());

    for (auto v : G.neighbors(u)) {
      if (M[edge_number] && !visited[v.vertex]) {
        queue.push(v.vertex);
      }

      ++edge_number;
    }

    visited[u] = true;
  }
  return std::count(visited.begin(), visited.end(), true);
}

template <typename GraphTy, typename GraphMaskTy>
size_t BFS(GraphTy &G, GraphMaskTy &M, typename GraphTy::vertex_type v,
           std::vector<bool> visited) {
  using vertex_type = typename GraphTy::vertex_type;

  std::queue<vertex_type> queue;

  queue.push(v);
  while (!queue.empty()) {
    vertex_type u = queue.front();
    queue.pop();

    size_t edge_number =
        std::distance(G.neighbors(0).begin(), G.neighbors(u).begin());
    for (auto v : G.neighbors(u)) {
      if (M[edge_number] && !visited[v.vertex]) {
        queue.push(v.vertex);
      }
      ++edge_number;
    }

    visited[u] = true;
  }
  return std::count(visited.begin(), visited.end(), true);
}
}  // namespace

template <typename GraphTy, typename GraphMaskItrTy>
auto SeedSelection(GraphTy &G, GraphMaskItrTy B, GraphMaskItrTy E,
                   std::size_t k, HillClimbingExecutionRecord &record) {
  using vertex_type = typename GraphTy::vertex_type;

  std::set<vertex_type> S;
  std::vector<size_t> count(G.num_nodes());

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < k; ++i) {
#pragma omp parallel for
    for (size_t i = 0; i < count.size(); ++i) count[i] = 0;

#pragma omp parallel for
    for (auto itr = B; itr < E; ++itr) {
      std::vector<bool> visited(G.num_nodes(), false);
      size_t base_count = BFS(G, *itr, S.begin(), S.end(), visited);

      for (vertex_type v = 0; v < G.num_nodes(); ++v) {
        if (S.find(v) != S.end()) continue;
        size_t update_count = base_count + 1;
        if (!visited[v]) update_count = BFS(G, *itr, v, visited);
#pragma omp atomic
        count[v] += update_count;
      }
    }

    vertex_type v = std::distance(count.begin(),
                                  std::max_element(count.begin(), count.end()));

    S.insert(v);
  }
  auto end = std::chrono::high_resolution_clock::now();
  record.SeedSelection = end - start;

  return S;
}

template <typename GraphTy, typename GeneratorTy, typename diff_model_tag>
auto HillClimbing(GraphTy &G, std::size_t k, std::size_t num_samples,
                  GeneratorTy &gen, HillClimbingExecutionRecord &record,
                  diff_model_tag &&model_tag) {
  auto sampled_graphs = SampleFrom(G, num_samples, gen, record,
                                   std::forward<diff_model_tag>(model_tag));

  auto S =
      SeedSelection(G, sampled_graphs.begin(), sampled_graphs.end(), k, record);

  return S;
}

}  // namespace ripples

#endif
