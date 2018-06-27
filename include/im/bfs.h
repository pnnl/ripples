//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_BFS_H
#define IM_BFS_H

#include <queue>
#include <unordered_set>

#include "trng/uniform01_dist.hpp"


namespace im {

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \return A random RR set.
template <
  typename GraphTy,
  typename PRNGeneratorTy>
std::unordered_set<typename GraphTy::vertex_type>
BFSOnRandomGraph(GraphTy &G, typename GraphTy::vertex_type r, PRNGeneratorTy &generator) {
  using vertex_type = typename GraphTy::vertex_type;

  trng::uniform01_dist<float> value;

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);
  std::unordered_set<vertex_type> result;

  queue.push(r);
  visited[r] = true;

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    for (auto u : G.in_neighbors(v)) {
      if (!visited[u.vertex] &&
          value(generator) < u.weight) {
        visited[u.vertex] = true;
        result.insert(u.vertex);
        queue.push(u.vertex);
      }
    }
  }

  return result;
}

} // namespace im

#endif  // IM_BFS_H
