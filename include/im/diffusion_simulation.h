//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "trng/uniform01_dist.hpp"

#include "im/graph.h"


namespace im {

struct independent_cascade_tag {};
struct linear_threshold_tag {};

namespace impl {

template <typename GraphTy, typename Iterator, typename PRNG>
size_t run_simulation(const GraphTy &G, Iterator begin, Iterator end,
                      PRNG &generator, const independent_cascade_tag &) {
  using vertex_type = typename GraphTy::vertex_type;
  using edge_weight_type = typename GraphTy::edge_weight_type;

  trng::uniform01_dist<edge_weight_type> value;

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);

  std::for_each(begin, end,
                [&](const vertex_type & v) {
                  queue.push(v);
                  visited[v] = true;
                });

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();
    visited[v] = true;

    for (auto u : G.in_neighbors(v)) {
      if (!visited[u.vertex] && value(generator) <= u.weight) {
        queue.push(u.vertex);
      }
    }
  }

  return std::count(visited.begin(), visited.end(), true);
}

}  // namespace impl

template <typename GraphTy, typename Iterator, typename PRNG, typename Model>
size_t simulate(const GraphTy &G, Iterator begin, Iterator end, PRNG &generator, const Model & M) {
  return impl::run_simulation(G, begin, end, generator, M);
}

}  // namespace im
