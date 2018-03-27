//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2017 Pacific Northwest National Laboratory
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
//===----------------------------------------------------------------------===//

#ifndef IM_BFS_H
#define IM_BFS_H

#include <set>
#include <random>

#include "im/configuration.h"

namespace im {

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \return A random RR set.
template <typename GraphTy>
std::set<typename GraphTy::vertex_type> BFSOnRandomGraph(GraphTy &G) {
  // This will be a slightly modified BFS.
  std::uniform_int_distribution<size_t> distribution(0, G.scale() - 1);
  typename GraphTy::vertex_type root = distribution(CFG.generator);

  std::uniform_real_distribution<double> value(0.0, 1.0);

  std::vector<typename GraphTy::vertex_type> queue(G.scale());
  auto start = std::begin(queue);
  auto end = std::begin(queue) + 1;
  auto next = std::begin(queue) + 1;

  *start = root;
  std::set<typename GraphTy::vertex_type> result;
  result.insert(root);

  while (start != end) {
    for(auto itr = start; itr != end; ++itr) {
      auto vertex = *itr;
      for (auto neighbor : G.in_neighbors(vertex)) {
        if (result.find(neighbor.v) == result.end() &&
            value(CFG.generator) <= neighbor.attribute) {
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
