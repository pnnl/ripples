//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_BFS_H
#define IM_BFS_H

#include <unordered_set>
#include <random>

#include "im/configuration.h"
#include "im/bloomfilter.h"

namespace im {

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \return A random RR set.
template <
  typename GraphTy,
  typename PRNGeneratorTy,
  typename VisitedSetTy = std::unordered_set<typename GraphTy::vertex_type>
  >
VisitedSetTy BFSOnRandomGraph(GraphTy &G, PRNGeneratorTy &generator) {
  // This will be a slightly modified BFS.
  std::uniform_int_distribution<size_t> distribution(0, G.scale() - 1);
  typename GraphTy::vertex_type root = distribution(generator);

  std::uniform_real_distribution<double> value(0.0, 1.0);

  std::vector<typename GraphTy::vertex_type> queue(G.scale());
  auto start = std::begin(queue);
  auto end = std::begin(queue) + 1;
  auto next = std::begin(queue) + 1;

  *start = root;
  VisitedSetTy result;
  result.insert(root);

  while (start != end) {
    for(auto itr = start; itr != end; ++itr) {
      auto vertex = *itr;
      for (auto neighbor : G.in_neighbors(vertex)) {
        if (result.find(neighbor.v) == result.end() &&
            value(generator) <= neighbor.attribute) {
          *next = neighbor.v;
          result.insert(neighbor.v);
          ++next;
        }
      }
    }

    start = end;
    end = next;
  }

  return result;
}

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \return A random RR set.
template <typename GraphTy, typename PRNGeneratorTy>
im::bloomfilter<uint64_t> BFSOnRandomGraph(
    GraphTy &G, PRNGeneratorTy &generator, double p) {
  // This will be a slightly modified BFS.
  std::uniform_int_distribution<size_t> distribution(0, G.scale() - 1);
  typename GraphTy::vertex_type root = distribution(generator);

  std::uniform_real_distribution<double> value(0.0, 1.0);

  std::vector<typename GraphTy::vertex_type> queue(G.scale());
  auto start = std::begin(queue);
  auto end = std::begin(queue) + 1;
  auto next = std::begin(queue) + 1;

  *start = root;

  size_t m = std::ceil(-1.44 * G.scale() * std::log2(p));
  size_t k = std::ceil(-std::log2(p));

  im::bloomfilter<uint64_t> result(m, k);

  result.insert(root);

  while (start != end) {
    for(auto itr = start; itr != end; ++itr) {
      auto vertex = *itr;
      for (auto neighbor : G.in_neighbors(vertex)) {
        if (!result.find(neighbor.v) &&
            value(generator) <= neighbor.attribute) {
          *next = neighbor.v;
          result.insert(neighbor.v);
          ++next;
        }
      }
    }

    start = end;
    end = next;
  }

  return result;
}

} // namespace im

#endif  // IM_BFS_H
