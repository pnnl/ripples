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

#ifndef RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H
#define RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H

#include "ripples/find_most_influential.h"
#include "ripples/utility.h"

#include "spdlog/spdlog.h"

namespace ripples {

//! \brief Select k seeds starting from the a list of Random Reverse
//! Reachability Sets.
//!
//! \tparam GraphTy The graph type.
//! \tparam RRRset The type storing Random Reverse Reachability Sets.
//!
//! \param G The input graph.
//! \param k The size of the seed set.
//! \param RRRsets A vector of Random Reverse Reachability sets.
//! \param ex_tag The MPI+OpenMP execution tag.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.
template <typename GraphTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            std::vector<RRRset> &RRRsets,
                            mpi_omp_parallel_tag &&ex_tag) {
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

  MPI_Win win;
  MPI_Win_create(reduceCoverageInfo.data(), G.num_nodes() * sizeof(uint32_t),
                 sizeof(uint32_t), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  CountOccurrencies(RRRsets.begin(), RRRsets.end(), vertexCoverage.begin(),
                    vertexCoverage.end(),
                    std::forward<omp_parallel_tag>(omp_parallel_tag{}));

  MPI_Win_fence(0, win);
  MPI_Accumulate(vertexCoverage.data(), G.num_nodes(), MPI_UINT32_T, 0, 0,
                 G.num_nodes(), MPI_UINT32_T, MPI_SUM, win);
  MPI_Win_fence(0, win);

  MPI_Win_free(&win);

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

  auto end = RRRsets.end();
  uint32_t coveredAndSelected[2] = {0, 0};

  while (result.size() < k) {
    if (rank == 0) {
      auto element = queue.top();
      queue.pop();

      if (element.second > reduceCoverageInfo[element.first]) {
        element.second = reduceCoverageInfo[element.first];
        queue.push(element);
        continue;
      }
      coveredAndSelected[0] += element.second;
      coveredAndSelected[1] = element.first;
    }

    MPI_Bcast(&coveredAndSelected, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    vertex_type v = coveredAndSelected[1];
    auto cmp = [=](const RRRset &a) -> auto {
      return !std::binary_search(a.begin(), a.end(), v);
    };

    auto itr = partition(RRRsets.begin(), end, cmp, omp_parallel_tag{});

    if (std::distance(itr, end) < std::distance(RRRsets.begin(), itr)) {
      UpdateCounters(itr, end, vertexCoverage, omp_parallel_tag{});
    } else {
#pragma omp parallel for simd
      for (size_t i = 0; i < vertexCoverage.size(); ++i) vertexCoverage[i] = 0;

      CountOccurrencies(RRRsets.begin(), itr, vertexCoverage.begin(),
                        vertexCoverage.end(), omp_parallel_tag{});
    }

    end = itr;

    MPI_Reduce(vertexCoverage.data(), reduceCoverageInfo.data(), G.num_nodes(),
               MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

    result.push_back(v);
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  double f = double(coveredAndSelected[0]) / (world_size * RRRsets.size());

  return std::make_pair(f, result);
}

}  // namespace ripples

#endif  // RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H
