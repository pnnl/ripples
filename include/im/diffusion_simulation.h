//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <set>

#include "trng/uniform01_dist.hpp"

#include "im/graph.h"

namespace im {

//! \brief Type-tag for the Independent Cascade Model.
struct independent_cascade_tag {};
//! \brief Type-tag for the Linear Threshold Model.
struct linear_threshold_tag {};

namespace impl {

//! \brief Simulate using the Independent Cascade Model.
//!
//! \tparam GraphTy The type of the Graph.
//! \tparam Iterator The Iterator type of the sequence of seeds.
//! \tparam PRNG The type of the parallel random number generator.
//!
//! \param G The input graph.
//! \param begin The start of the sequence of seeds.
//! \param end The end of the sequence of seeds.
//! \param generator The parallel random number generator.
template <typename GraphTy, typename Iterator, typename PRNG>
auto run_simulation(const GraphTy &G, Iterator begin, Iterator end,
                    PRNG &generator, const independent_cascade_tag &) {
  using vertex_type = typename GraphTy::vertex_type;
  using edge_weight_type = typename GraphTy::edge_weight_type;

  trng::uniform01_dist<edge_weight_type> value;

  std::vector<vertex_type> queue;
  queue.reserve(G.num_nodes());

  std::vector<bool> visited(G.num_nodes(), false);

  std::for_each(begin, end, [&](const vertex_type &v) {
    queue.push_back(v);
    visited[v] = true;
  });

  auto itr = queue.begin();
  auto level_end = queue.end();
  size_t level = 0;

  while (itr != queue.end()) {
    vertex_type v = *itr;

    for (auto u : G.neighbors(v)) {
      if (!visited[u.vertex] && value(generator) <= u.weight) {
        visited[u.vertex] = true;
        queue.push_back(u.vertex);
      }
    }

    if (++itr == level_end) {
      level_end = queue.end();
      ++level;
    }
  }

  return std::make_pair(std::count(visited.begin(), visited.end(), true),
                        level);
}

//! Run the simulation for the Linear Threshold Model.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam Iterator The type of the iterator for the sequence of seeds.
//! \tparam PRNG The type of the Parallel Random Number Generator.
//!
//! \param G The input graph.
//! \param begin The start of the sequence of seeds.
//! \param end The end of the sequence of seeds.
//! \param generator The Parallel Random Number Generator.
//! \return a pair (A, S), where A is the number of activated nodes and S is the
//! number of steps the simulation run.
template <typename GraphTy, typename Iterator, typename PRNG>
auto run_simulation(const GraphTy &G, Iterator begin, Iterator end,
                    PRNG &generator, const linear_threshold_tag &) {
  using vertex_type = typename GraphTy::vertex_type;
  using edge_weight_type = typename GraphTy::edge_weight_type;

  auto transposedG = G.get_transpose();

  trng::uniform01_dist<edge_weight_type> thresholds_generator;

  std::vector<edge_weight_type> thresholds(G.num_nodes());
  std::generate(
      thresholds.begin(), thresholds.end(),
      [&]() -> edge_weight_type { return thresholds_generator(generator); });

  std::set<vertex_type> active(begin, end);
  std::set<vertex_type> tobe_activated;

  size_t level(0);
  do {
    std::set<vertex_type> tobe_processed;
    tobe_activated.clear();
    for (auto v : active) {
      for (auto u : G.neighbors(v)) {
        if (active.find(u.vertex) == active.end()) {
          tobe_processed.insert(u.vertex);
        }
      }
    }

    for (auto v : tobe_processed) {
      edge_weight_type total(0);

      for (auto u : transposedG.neighbors(v)) {
        if (active.find(u.vertex) != active.end()) {
          total += u.weight;
        }
      }

      if (total >= thresholds[v]) {
        tobe_activated.insert(v);
      }
    }

    active.insert(tobe_activated.begin(), tobe_activated.end());
    ++level;
  } while (!tobe_activated.empty());

  return std::make_pair(active.size(), level);
}

}  // namespace impl

//! \brief Simulate the diffusion on the input graph.
//!
//! \tparam GraphTy The type of the Graph.
//! \tparam Iterator The Iterator type of the sequence of seeds.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam Model The type-tag of the diffusion model to be used.
//!
//! \param G The input graph.
//! \param begin The start of the sequence of seeds.
//! \param end The end of the sequence of seeds.
//! \param generator The parallel random number generator.
//! \param M The diffusion model tag.
template <typename GraphTy, typename Iterator, typename PRNG, typename Model>
auto simulate(const GraphTy &G, Iterator begin, Iterator end, PRNG &generator,
              const Model &M) {
  return impl::run_simulation(G, begin, end, generator, M);
}

}  // namespace im
