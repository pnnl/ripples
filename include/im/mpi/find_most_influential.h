//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_MPI_FIND_MOST_INFLUENTIAL_H
#define IM_MPI_FIND_MOST_INFLUENTIAL_H

#include "im/find_most_influential.h"
#include "im/utility.h"

#include "spdlog/spdlog.h"

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
template <typename GraphTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            const std::vector<RRRset> &RRRsets,
                            mpi_omp_parallel_tag &ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<uint32_t> vertexCoverage(G.num_nodes(), 0);
  std::vector<uint32_t> reduceCoverageInfo(G.num_nodes(), 0);

  auto cmp = [](std::pair<vertex_type, uint32_t> &a,
                std::pair<vertex_type, uint32_t> &b) {
    return a.second < b.second;
  };
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, uint32_t>,
                          std::vector<std::pair<vertex_type, uint32_t>>,
                          decltype(cmp)>;

  CountOccurrencies(RRRsets.begin(), RRRsets.end(), vertexCoverage.begin(),
                    vertexCoverage.end(),
                    std::forward<omp_parallel_tag>(omp_parallel_tag{}));

  MPI_Reduce(vertexCoverage.data(), reduceCoverageInfo.data(), G.num_nodes(),
             MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

  std::vector<std::pair<vertex_type, uint32_t>> queue_storage;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    queue_storage.resize(G.num_nodes());
    InitHeapStorage(reduceCoverageInfo.begin(), reduceCoverageInfo.end(),
                    queue_storage.begin(), queue_storage.end(),
                    std::forward<omp_parallel_tag>(omp_parallel_tag{}));
  }
  priorityQueue queue(cmp, std::move(queue_storage));

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  std::vector<char> removed(RRRsets.size(), false);
  uint64_t covered = 0;

  while (result.size() < k) {
    vertex_type selected;
    if (rank == 0) {
      auto element = queue.top();
      queue.pop();

      if (element.second > reduceCoverageInfo[element.first]) {
        element.second = reduceCoverageInfo[element.first];
        queue.push(element);
        continue;
      }
      selected = element.first;
      covered += element.second;
    }

    MPI_Bcast(&selected, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    UpdateCounters(selected, RRRsets, removed, vertexCoverage,
                   omp_parallel_tag{});

    MPI_Reduce(vertexCoverage.data(), reduceCoverageInfo.data(), G.num_nodes(),
               MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

    result.push_back(selected);
  }

  MPI_Bcast(&covered, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  double f = double(covered) / (world_size * RRRsets.size());

  return std::make_pair(f, result);
}

}  // namespace im

#endif  // IM_MPI_FIND_MOST_INFLUENTIAL_H
