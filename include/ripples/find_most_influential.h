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

#ifndef RIPPLES_FIND_MOST_INFLUENTIAL_H
#define RIPPLES_FIND_MOST_INFLUENTIAL_H

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <vector>

#include <omp.h>
#include "ripples/counting.h"
#include "ripples/imm_execution_record.h"
#include "ripples/partition.h"
#include "ripples/streaming_find_most_influential.h"
#include "ripples/utility.h"

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#ifdef RIPPLES_ENABLE_CUDA
#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/find_most_influential.h"
#include "thrust/count.h"
#include "thrust/device_ptr.h"
#endif

namespace ripples {

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
//! \param ex_tag The execution policy tag.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.
template <typename GraphTy, typename ConfTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, const ConfTy &CFG,
                            std::vector<RRRset> &RRRsets,
                            IMMExecutionRecord &record, bool enableGPU,
                            sequential_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;

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

  auto counting = measure<>::exec_time([&]() {
    CountOccurrencies(RRRsets.begin(), RRRsets.end(), vertexCoverage.begin(),
                      vertexCoverage.end(),
                      std::forward<sequential_tag>(ex_tag));
  });

  InitHeapStorage(vertexCoverage.begin(), vertexCoverage.end(),
                  queue_storage.begin(), queue_storage.end(),
                  std::forward<sequential_tag>(ex_tag));

  priorityQueue queue(cmp, std::move(queue_storage));

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  size_t uncovered = RRRsets.size();

  auto end = RRRsets.end();
  typename IMMExecutionRecord::ex_time_ms pivoting;

  while (result.size() < k && uncovered != 0) {
    auto element = queue.top();
    queue.pop();

    if (element.second > vertexCoverage[element.first]) {
      element.second = vertexCoverage[element.first];
      queue.push(element);
      continue;
    }

    uncovered -= element.second;

    auto cmp = [=](const RRRset &a) -> auto {
      return !std::binary_search(a.begin(), a.end(), element.first);
    };

    auto start = std::chrono::high_resolution_clock::now();
    auto itr = partition(RRRsets.begin(), end, cmp,
                         std::forward<sequential_tag>(ex_tag));
    pivoting += (std::chrono::high_resolution_clock::now() - start);

    counting += measure<>::exec_time([&]() {
      if (std::distance(itr, end) < std::distance(RRRsets.begin(), itr)) {
        UpdateCounters(itr, end, vertexCoverage,
                       std::forward<sequential_tag>(ex_tag));
      } else {
        std::fill(vertexCoverage.begin(), vertexCoverage.end(), 0);
        CountOccurrencies(RRRsets.begin(), itr, vertexCoverage.begin(),
                          vertexCoverage.end(),
                          std::forward<sequential_tag>(ex_tag));
      }
    });
    end = itr;
    result.push_back(element.first);
  }

  double f = double(RRRsets.size() - uncovered) / RRRsets.size();

  record.Counting.push_back(
      std::chrono::duration_cast<typename IMMExecutionRecord::ex_time_ms>(
          counting));
  record.Pivoting.push_back(pivoting);
  return std::make_pair(f, result);
}

template <typename GraphTy, typename ConfTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, const ConfTy &CFG,
                            std::vector<RRRset> &RRRsets,
                            IMMExecutionRecord &record, bool enableGPU,
                            omp_parallel_tag &&ex_tag) {
  size_t num_gpu = 0;
  size_t num_max_cpu = 0;
#pragma omp single
  {
    num_max_cpu =
        std::min<size_t>(omp_get_max_threads(), CFG.seed_select_max_workers);
  }
#ifdef RIPPLES_ENABLE_CUDA
  if (enableGPU) {
    num_gpu = std::min(cuda_num_devices(), CFG.seed_select_max_gpu_workers);
  }
#endif
  StreamingFindMostInfluential<GraphTy> SE(G, RRRsets, num_max_cpu, num_gpu);
  return SE.find_most_influential_set(CFG.k);
}

#if RIPPLES_ENABLE_CUDA
template <typename Itr>
void MoveRRRSets(Itr in_begin, Itr in_end, uint32_t *d_rrr_index,
                 uint32_t *d_rrr_sets, size_t rrr_index_size,
                 size_t rrr_sets_size) {
  std::vector<uint32_t> buffer(rrr_sets_size);
  std::vector<uint32_t> buffer2(rrr_sets_size);

  auto position = buffer.begin();
  auto position2 = buffer2.begin();
  uint32_t id = 0;
  for (auto itr = in_begin; itr < in_end; ++itr, ++id) {
    position = std::copy(itr->begin(), itr->end(), position);

    std::fill(position2, position2 + itr->size(), id);
    std::advance(position2, itr->size());
  }

  cuda_h2d(reinterpret_cast<void *>(d_rrr_index),
           reinterpret_cast<void *>(buffer2.data()),
           sizeof(uint32_t) * rrr_sets_size);
  cuda_h2d(reinterpret_cast<void *>(d_rrr_sets),
           reinterpret_cast<void *>(buffer.data()),
           sizeof(uint32_t) * rrr_sets_size);
}

template <typename GraphTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            std::vector<RRRset> &RRRsets,
                            IMMExecutionRecord &record,
                            cuda_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  size_t rrr_sets_size = 0;
#pragma omp parallel for reduction(+ : rrr_sets_size)
  for (auto itr = RRRsets.begin(); itr < RRRsets.end(); ++itr) {
    rrr_sets_size += itr->size();
  }
  size_t rrr_index_size = rrr_sets_size;

  uint32_t *d_Counters;
  cuda_malloc(reinterpret_cast<void **>(&d_Counters),
              sizeof(uint32_t) * G.num_nodes());
  cuda_memset(reinterpret_cast<void **>(d_Counters), 0,
              sizeof(uint32_t) * G.num_nodes());
  uint32_t *d_rrr_index;
  uint32_t *d_rrr_sets;
  cuda_malloc(reinterpret_cast<void **>(&d_rrr_index),
              sizeof(uint32_t) * rrr_index_size);
  cuda_malloc(reinterpret_cast<void **>(&d_rrr_sets),
              sizeof(uint32_t) * rrr_sets_size);

  char *d_rr_mask;
  cuda_malloc(reinterpret_cast<void **>(&d_rr_mask),
              sizeof(char) * RRRsets.size());
  cuda_memset(reinterpret_cast<void *>(d_rr_mask), 0,
              sizeof(char) * RRRsets.size());

  auto counting = measure<>::exec_time([&]() {
    MoveRRRSets(RRRsets.begin(), RRRsets.end(), d_rrr_index, d_rrr_sets,
                rrr_index_size, rrr_sets_size);
  });

  counting += measure<>::exec_time([&]() {
    CudaCountOccurrencies(d_Counters, d_rrr_sets, rrr_sets_size, G.num_nodes());
  });

  std::vector<vertex_type> result;
  size_t uncovered = RRRsets.size();

  while (uncovered != 0) {
    // Find Max
    auto v = CudaMaxElement(d_Counters, G.num_nodes());

    result.push_back(v.first);
    uncovered -= v.second;

    std::cout << "Reference Selected : " << v.first << " " << v.second
              << std::endl;
    if (result.size() == k) break;

    // Update Counters
    counting += measure<>::exec_time([&]() {
      CudaUpdateCounters(rrr_sets_size, d_rrr_index, d_rrr_sets, d_rr_mask,
                         d_Counters, G.num_nodes(), v.first);
    });
  }

  cuda_free(d_Counters);
  cuda_free(d_rrr_index);
  cuda_free(d_rrr_sets);
  cuda_free(d_rr_mask);

  double f = double(RRRsets.size() - uncovered) / RRRsets.size();
  record.Counting.push_back(
      std::chrono::duration_cast<typename IMMExecutionRecord::ex_time_ms>(
          counting));
  return std::make_pair(f, result);
}
#endif

}  // namespace ripples

#endif  // RIPPLES_FIND_MOST_INFLUENTIAL_H
