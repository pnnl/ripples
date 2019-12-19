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

#include <type_traits>
#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"

namespace ripples {

//! The Hill Climbing Algorithm configuration descriptor.
struct HillClimbingConfiguration {
  size_t samples{10000};

  //! \brief Add command line options to configure the Hill Climbing Algorithm.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    app.add_option("--samples", samples,
                   "The number of samples used in the Hill Climbing Algorithm.")
  }
};

template <typename GraphTy, typename GeneratorTy, typename diff_model_tag>
auto SampleFrom(GraphTy &G, std::size_t num_samples, GeneratorTy &gen,
                diff_model_tag &&diff_model) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<GraphTy> samples(num_samples);

  for (size_t i = 0; i < num_samples; ++i) {
    // 1 - Sample a list of Edges from G.
    std::vector<Edge> EL;
    if (std::is_same<diff_model_tag, independent_cascade_tag>::value) {
      for (vertex_type v = 0; v < G.num_nodes(); ++v) {
        for (auto &e : G.neighbors(v)) {
          if (rand < e.weight) continue;

          EL.emplace_back(Edge{v, e.vertex});
        }
      }
    } else if (std::is_same<diff_model_tag, linear_threshold_tag>::value) {
      for (vertex_type v = 0; v < G.num_nodes(); ++v) {
        double threshold = random;
        for (auto &e : G.neighbors(v)) {
          threshold -= e.weight;
          if (threshold <= 0) {
            EL.emplace_back(Edge{v, e.vertex});
          }
        }
      }
    } else {
      // Unsupported.
    }
    // 2 - Create an unweighted graph from it.
    // 3 - Add it to the list.
  }

  return samples;
}

template <typename GraphTy, typename GraphItrTy>
auto SeedSelection(GraphTy &G, GraphItrTy B, GraphItrTy E, std::size_t k) {
  using graph_type = std::iterator<GraphItrTy>::value_type;
  using vertex_type = graph_type::vertex_type;

  std::set<vertex_type> S;
  std::vector<size_t> count(G.num_vertices());

  for (size_t i = 0; i < k; ++i) {
    // #pragma omp parallel for with vector reduction
    for (auto itr = B; itr < E; ++itr) {
      for (vertex_type v : *itr) {
        vertex_type original_v = B->transformID(v);
        if (S.find(B->transformID(v))) continue;

        count[original_v] += BFS(*itr, S.begin(), S.end(), v);
      }
    }

    auto v = *std::max_element(count.begin(), count.end());

    S.insert(v);
  }

  return S;
}

template <typename GraphTy, typename GeneratorTy, typename diff_model_tag>
auto HillClimbing(GraphTy &G, std::size_t k, std::size_t num_samples,
                  GeneratorTy &gen, diff_model_tag &&model_tag) {
  auto sampled_graphs = SampleFrom(G, num_samples, gen);

  auto S = SeedSelection(sampled_graphs.begin(), sampled_graphs.end(), k);

  return S;
}

}  // namespace ripples

#endif
