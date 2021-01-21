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

#ifndef RIPPLES_LOUVAIN_IMM_H
#define RIPPLES_LOUVAIN_IMM_H

#include <queue>
#include <string>
#include <type_traits>
#include <vector>

#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/imm.h"

#include "ripples/imm_execution_record.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace ripples {

struct LouvainIMMConfiguration : public IMMConfiguration {
  std::string communityList;
  void addCmdOptions(CLI::App &app) {
    IMMConfiguration::addCmdOptions(app);

    app.add_option("--community-map", communityList,
                   "The filename of the community map.")
        ->required()
        ->group("Algorithm Options");
  }
};

struct LouvainIMMExecutionRecord : public IMMExecutionRecord {};

namespace {
template <typename vertex_type>
struct Compare {
  bool operator()(std::pair<vertex_type, size_t> &a,
                  std::pair<vertex_type, size_t> &b) const {
    return a.second < b.second;
  }
};
}  // namespace

template <typename GraphTy, typename RRRset, typename execution_tag>
auto FindMostInfluentialSet(const std::vector<GraphTy> &communities, size_t k,
                            std::vector<std::vector<RRRset>> &RRRcollection,
                            execution_tag &&ex_tag) {
  spdlog::get("console")->info("SeedSelect start");

  using vertex_type = typename GraphTy::vertex_type;

  Compare<vertex_type> cmp;

  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, size_t>,
                          std::vector<std::pair<vertex_type, size_t>>,
                          decltype(cmp)>;

  // Count occurrencies for all communities
  std::vector<std::vector<uint32_t>> coverageVectors(communities.size());
  std::vector<priorityQueue> queues(communities.size());
  std::vector<typename std::vector<RRRset>::iterator> ends(communities.size());

  double total_delta = 0;
#pragma omp parallel for reduction(+ : total_delta)
  for (size_t i = 0; i < communities.size(); ++i) {
    coverageVectors[i] = std::vector<uint32_t>(communities[i].num_nodes(), 0);

    CountOccurrencies(RRRcollection[i].begin(), RRRcollection[i].end(),
                      coverageVectors[i].begin(), coverageVectors[i].end(),
                      std::forward<execution_tag>(ex_tag));

    std::vector<std::pair<vertex_type, size_t>> queue_storage(
        communities[i].num_nodes());

    InitHeapStorage(coverageVectors[i].begin(), coverageVectors[i].end(),
                    queue_storage.begin(), queue_storage.end(),
                    std::forward<execution_tag>(ex_tag));

    queues[i] = std::move(priorityQueue(cmp, std::move(queue_storage)));
    ends[i] = RRRcollection[i].end();

    total_delta += RRRcollection[i].size();
  }

  spdlog::get("console")->flush();

  // Init on heap per community
  using vertex_contribution_pair = std::pair<vertex_type, double>;
  std::vector<vertex_contribution_pair> global_heap(
      k + 1, vertex_contribution_pair{-1, -1.0});
  std::vector<uint64_t> active_communities(communities.size(), 1);

  auto heap_cmp = [](const vertex_contribution_pair &a,
                     const vertex_contribution_pair &b) -> bool {
    return a.second > b.second;
  };

  std::make_heap(global_heap.begin(), global_heap.end(), heap_cmp);
  // std::mutex global_heap_mutex;

  // for each communities do in parallel
  size_t iteration = 0;
  while (!std::all_of(active_communities.begin(), active_communities.end(),
                      [](const uint64_t &v) -> bool { return v == 0; })) {
    for (size_t i = 0; i < communities.size(); ++i) {
      if (active_communities[i] == 0) continue;

      if (queues[i].empty()) {
        active_communities[i] = 0;
        continue;
      }

      auto element = queues[i].top();
      queues[i].pop();

      while (element.second > coverageVectors[i][element.first]) {
        element.second = coverageVectors[i][element.first];
        queues[i].push(element);

        element = queues[i].top();
        queues[i].pop();
      }

      auto cmp = [=](const RRRset &a) -> auto {
        return !std::binary_search(a.begin(), a.end(), element.first);
      };

      auto itr = partition(RRRcollection[i].begin(), ends[i], cmp,
                           std::forward<execution_tag>(ex_tag));

      if (std::distance(itr, ends[i]) <
          std::distance(RRRcollection[i].begin(), itr)) {
        UpdateCounters(itr, ends[i], coverageVectors[i],
                       std::forward<execution_tag>(ex_tag));
      } else {
        if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp parallel for simd
          for (size_t j = 0; j < coverageVectors[i].size(); ++j)
            coverageVectors[i][j] = 0;
        } else {
          std::fill(coverageVectors[i].begin(), coverageVectors[i].end(), 0);
        }
        CountOccurrencies(RRRcollection[i].begin(), itr,
                          coverageVectors[i].begin(), coverageVectors[i].end(),
                          std::forward<execution_tag>(ex_tag));
      }

      ends[i] = itr;

      double contribution = RRRcollection[i].size()
                                ? element.second / RRRcollection[i].size()
                                : 0;
      vertex_contribution_pair vcp{communities[i].convertID(element.first),
                                   contribution};

      // Handle the global index insertion
      // std::lock_guard<std::mutex> _(global_heap_mutex);
      std::pop_heap(global_heap.begin(), global_heap.end(), heap_cmp);
      global_heap.back() = vcp;
      std::push_heap(global_heap.begin(), global_heap.end(), heap_cmp);

      if (global_heap.front() == vcp) active_communities[i] = 0;
    }
  }

  std::pop_heap(global_heap.begin(), global_heap.end(), heap_cmp);
  global_heap.pop_back();

  double coverage = 0;
  std::vector<typename GraphTy::vertex_type> seeds;
  seeds.reserve(k);
  for (auto e : global_heap) {
    seeds.push_back(e.first);
    coverage += e.second;
  }

  return seeds;
}

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename RecordTy, typename diff_model_tag>
auto LouvainIMM(const std::vector<GraphTy> &communities, ConfTy &CFG, double l,
                GeneratorTy &gen, std::vector<RecordTy> &records, diff_model_tag &&model_tag,
                sequential_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  using RRRsetCollection = std::vector<RRRset<GraphTy>>;
  std::vector<RRRsetCollection> R(communities.size());

  // For each community do ThetaEstimation and Sampling
  for (size_t i = 0; i < communities.size(); ++i) {
    double l_1 = l * (1 + 1 / std::log2(communities[i].num_nodes()));

    R[i] = Sampling(communities[i], CFG, l_1, gen, records[i],
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<sequential_tag>(ex_tag));
  }

  // Global seed selection using the heap
  auto S = FindMostInfluentialSet(communities, k, R,
                                  std::forward<sequential_tag>(ex_tag));

  return std::make_pair(S, records);
}
//! Influence Maximization using Community Structure.
//!
//! The algorithm uses the Louvain method for community detection and then
//! IMM to select seeds frome the communities.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param communities The input graphs.  The graphs are transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
auto LouvainIMM(const std::vector<GraphTy> &communities, ConfTy &CFG, double l,
                std::vector<GeneratorTy> &gen, diff_model_tag &&model_tag,
                omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  using RRRsetCollection = std::vector<RRRset<GraphTy>>;
  std::vector<RRRsetCollection> R(communities.size());

  // For each community do ThetaEstimation and Sampling
  for (size_t i = 0; i < communities.size(); ++i) {
    double l_1 = l * (1 + 1 / std::log2(communities[i].num_nodes()));

    R[i] = Sampling(communities[i], CFG, l_1, gen[i], gen[i].execution_record(),
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<omp_parallel_tag>(ex_tag));
  }

  // Global seed selection using the heap
  auto S = FindMostInfluentialSet(communities, k, R,
                                  std::forward<omp_parallel_tag>(ex_tag));
  std::vector<IMMExecutionRecord> records(communities.size());

  for (auto & generator : gen) {
    records.push_back(generator.execution_record());
  }

  return std::make_pair(S, records);
}

}  // namespace ripples

#endif /* RIPPLES_LOUVAIN_IMM_H */
