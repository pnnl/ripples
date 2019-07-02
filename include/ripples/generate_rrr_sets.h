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

#ifndef RIPPLES_GENERATE_RRR_SETS_H
#define RIPPLES_GENERATE_RRR_SETS_H

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/utility.h"

#include "ripples/cuda/cuda_generate_rrr_sets.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

namespace ripples {

//! \brief The Random Reverse Reachability Sets type
template <typename GraphTy>
using RRRset = std::vector<typename GraphTy::vertex_type>;

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam PRNGGeneratorTy The type of pseudo the random number generator.
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The graph instance.
//! \param r The starting point for the exploration.
//! \param generator The pseudo random number generator.
//! \param result The RRR set
//! \param tag The diffusion model tag.
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

    if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
      for (auto u : G.neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
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
          result.push_back(u.vertex);
        }
        break;
      }
    } else {
      throw;
    }
  }

  std::stable_sort(result.begin(), result.end());
}

//! \brief Generate Random Reverse Reachability Sets - sequential.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param generator The random numeber generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
std::vector<RRRset<GraphTy>> GenerateRRRSets(GraphTy &G, size_t theta,
                                             PRNGeneratorTy &generator,
                                             diff_model_tag &&model_tag,
                                             sequential_tag &&ex_tag) {
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

//! \brief Generate Random Reverse Reachability Sets - OpenMP.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param generator The random numeber generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
std::vector<RRRset<GraphTy>> GenerateRRRSets(GraphTy &G, size_t theta,
                                             PRNGeneratorTy &generator,
                                             diff_model_tag &&model_tag,
                                             omp_parallel_tag &&ex_tag) {
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

//! \brief Generate Random Reverse Reachability Sets - CUDA.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param theta The number of RRR sets to be generated.
//! \param generator The random numeber generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
//!
//! \return A list of theta Random Reverse Rachability Sets.
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
std::vector<RRRset<GraphTy>> GenerateRRRSets(GraphTy &G, size_t theta,
                                             PRNGeneratorTy &,
                                             diff_model_tag &&model_tag,
                                             cuda_parallel_tag &&ex_tag) {
  return CudaGenerateRRRSets(theta, std::forward<diff_model_tag>(model_tag));
}

}  // namespace ripples

#endif  // RIPPLES_GENERATE_RRR_SETS_H
