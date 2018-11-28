//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_GENERATE_RRR_SETS_H
#define IM_GENERATE_RRR_SETS_H

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "im/diffusion_simulation.h"
#include "im/graph.h"
#include "im/utility.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

namespace im {

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

    if (std::is_same<diff_model_tag, im::independent_cascade_tag>::value) {
      for (auto u : G.neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
        }
      }
    } else if (std::is_same<diff_model_tag, im::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.neighbors(v)) {
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

//! \brief Generate Random Reverse Reachability Sets.
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

}  // namespace im

#endif  // IM_GENERATE_RRR_SETS_H
