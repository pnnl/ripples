//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_FIND_MOST_INFLUENTIAL_H
#define IM_FIND_MOST_INFLUENTIAL_H

#include <algorithm>
#include <queue>
#include <vector>

#include <omp.h>

#include "im/utility.h"

namespace im {

//! \brief Select k seeds starting from the a list of Random Reverse
//! Reachability Sets.
//!
//! \tparam GraphTy The graph type.
//! \tparam RRRset The type storing Random Reverse Reachability Sets.
//! \tparam execution_tag The execution policy.
//!
//! \param G The input graph.
//! \param k The size of the seed set.
//! \param RRRsets A vector of Random Reverse Reachability sets.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.
template <typename GraphTy, typename RRRset, typename execution_tag>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            const std::vector<RRRset> &RRRsets,
                            execution_tag &&) {
  using vertex_type = typename GraphTy::vertex_type;

  std::vector<uint32_t> vertexCoverage(G.num_nodes(), 0);

  auto cmp = [](std::pair<vertex_type, size_t> &a,
                std::pair<vertex_type, size_t> &b) {
    return a.second < b.second;
  };
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, size_t>,
                          std::vector<std::pair<vertex_type, size_t>>,
                          decltype(cmp)>;

  std::vector<std::pair<vertex_type, size_t>> queue_storage(G.num_nodes());

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp parallel
    {
      size_t threadnum = omp_get_thread_num(),
             numthreads = omp_get_num_threads();
      vertex_type low = G.num_nodes() * threadnum / numthreads,
                  high = G.num_nodes() * (threadnum + 1) / numthreads;

      for (size_t i = 0; i < RRRsets.size(); ++i) {
        auto begin =
            std::lower_bound(RRRsets[i].begin(), RRRsets[i].end(), low);
        auto end =
            std::upper_bound(RRRsets[i].begin(), RRRsets[i].end(), high - 1);
        std::for_each(begin, end,
                      [&](const vertex_type v) { vertexCoverage[v] += 1; });
      }
    }
#pragma omp parallel for
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      queue_storage[i] = std::make_pair(i, vertexCoverage[i]);
    }
  } else {
    for (size_t i = 0; i < RRRsets.size(); ++i) {
      std::for_each(RRRsets[i].begin(), RRRsets[i].end(),
                    [&](const vertex_type v) { vertexCoverage[v] += 1; });
    }
    for (vertex_type i = 0; i < G.num_nodes(); ++i) {
      queue_storage[i] = std::make_pair(i, vertexCoverage[i]);
    }
  }

  priorityQueue queue(cmp, std::move(queue_storage));

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  std::vector<char> removed(RRRsets.size(), false);
  size_t uncovered = RRRsets.size();

  while (result.size() < k && uncovered != 0) {
    auto element = queue.top();
    queue.pop();

    if (element.second > vertexCoverage[element.first]) {
      element.second = vertexCoverage[element.first];
      queue.push(element);
      continue;
    }

    uncovered -= element.second;

    if (std::is_same<execution_tag, omp_parallel_tag>::value) {
      for (size_t i = 0; i < RRRsets.size(); ++i) {
        if (removed[i]) continue;

        if (std::binary_search(RRRsets[i].begin(), RRRsets[i].end(),
                               element.first)) {
          removed[i] = true;
#pragma omp parallel for
          for (size_t j = 0; j < RRRsets[i].size(); ++j) {
            vertexCoverage[RRRsets[i][j]] -= 1;
          }
        }
      }
    } else {
      for (size_t i = 0; i < RRRsets.size(); ++i) {
        if (removed[i]) continue;

        if (std::binary_search(RRRsets[i].begin(), RRRsets[i].end(),
                               element.first)) {
          removed[i] = true;
          for (size_t j = 0; j < RRRsets[i].size(); ++j) {
            vertexCoverage[RRRsets[i][j]] -= 1;
          }
        }
      }
    }

    result.push_back(element.first);
  }

  return std::make_pair(RRRsets.size() - uncovered, result);
}

}  // namespace im

#endif // IM_FIND_MOST_INFLUENTIAL_H
