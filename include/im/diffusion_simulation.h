//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>

#include "trng/uniform01_dist.hpp"

#include "im/graph.h"

namespace im {

struct independent_cascade_tag {};
struct linear_threshold_tag {};

namespace impl {

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

}  // namespace impl

template <typename GraphTy, typename Iterator, typename PRNG, typename Model>
auto simulate(const GraphTy &G, Iterator begin, Iterator end, PRNG &generator,
              const Model &M) {
  return impl::run_simulation(G, begin, end, generator, M);
}

}  // namespace im
