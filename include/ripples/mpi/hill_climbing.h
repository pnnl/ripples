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

#ifndef RIPPLES_MPI_HILL_CLIMBING_H
#define RIPPLES_MPI_HILL_CLIMBING_H

#include "ripples/hill_climbing.h"
#include "spdlog/spdlog.h"

#include "mpi.h"

namespace ripples {
namespace mpi {
template <typename GraphTy, typename GraphMaskItrTy>
auto SeedSelection(GraphTy &G, GraphMaskItrTy B, GraphMaskItrTy E,
                   std::size_t k) {
  using vertex_type = typename GraphTy::vertex_type;

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::set<vertex_type> S;
  size_t vertex_block_size = (G.num_nodes() / world_size) + 1;
  std::vector<long> global_count(vertex_block_size, 0);
  std::vector<long> local_count(vertex_block_size, 0);

  MPI_Win win;
  MPI_Win_create(global_count.data(), vertex_block_size * sizeof(long),
                 sizeof(long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  MPI_Win_fence(0, win);
  std::vector<std::vector<bool>> frontier_cache(std::distance(B, E));

  for (size_t i = 0; i < k; ++i) {

#pragma openmp parallel for schedule(dynamic)
    for (auto itr = B; itr < E; ++itr) {
      frontier_cache[std::distance(B, itr)].resize(G.num_nodes(), false);
      BFS(G, *itr, S.begin(), S.end(), frontier_cache[std::distance(B, itr)]);
    }

    for (int p = 0; p < world_size; ++p) {
      int current_block = (p + rank) % world_size;

      spdlog::get("console")->info("Rank {} - Block {}", rank, current_block);
      vertex_type start = current_block * vertex_block_size;
      vertex_type end = std::min(start + vertex_block_size, G.num_nodes());
      for (auto itr = B; itr < E; ++itr) {
        size_t sample_id = std::distance(B, itr);
        size_t base = std::count(frontier_cache[sample_id].begin(),
                                 frontier_cache[sample_id].end(), true);
        // if (frontier_cache[sample_id].empty()) {
        //   frontier_cache[sample_id].resize(G.num_nodes(), false);
        //   base = BFS(G, *itr, S.begin(), S.end(), frontier_cache[sample_id]);
        // } else {
        //   base = std::count(frontier_cache[sample_id].begin(),
        //                     frontier_cache[sample_id].end(), true);
        // }
#pragma omp parallel for schedule(dynamic)
        for (vertex_type v = start; v < end; ++v) {
          if (S.find(v) != S.end()) continue;
          long count = base + 1;
          if (!frontier_cache[sample_id][v]) count = BFS(G, *itr, v, frontier_cache[sample_id]);
          local_count[v - start] += count;
        }
      }

      MPI_Accumulate(local_count.data(), vertex_block_size, MPI_LONG,
                     current_block, 0, vertex_block_size, MPI_LONG, MPI_SUM,
                     win);

#pragma omp parallel for
      for (size_t i = 0; i < local_count.size(); ++i) {
        local_count[i] = 0;
      }
    }

    MPI_Win_fence(0, win);

    vertex_type v = std::distance(
        global_count.begin(),
        std::max_element(global_count.begin(), global_count.end()));

    struct {
      long count;
      int index;
    } local, global;
    local.count = global_count[v];
    local.index = rank * vertex_block_size + v;

    spdlog::get("console")->info("R[{}] ({}, {})", rank, local.count,
                                 local.index);

#pragma omp parallel for
    for (size_t i = 0; i < global_count.size(); ++i) {
      local_count[i] = 0;
      global_count[i] = 0;
    }

#pragma omp parallel for
    for (size_t i = 0; i < frontier_cache.size(); ++i) {
      frontier_cache[i].resize(0);
    }

    MPI_Allreduce(&local, &global, 1, MPI_LONG_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    S.insert(global.index);
  }
  MPI_Win_free(&win);
  return S;
}  // namespace mpi

//! The HillClimbing algorithm for Influence Maximization (MPI Specialization).
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam GeneratorTy The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to select the diffusion model.
//!
//! \param G The input graph.
//! \param k The number of seeds to select.
//! \param num_samples The number of samples to take from G.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution model tag.
//! \returns a set of k vertices of G.
template <typename GraphTy, typename GeneratorTy, typename diff_model_tag>
auto HillClimbing(GraphTy &G, std::size_t k, std::size_t num_samples,
                  GeneratorTy &gen, diff_model_tag &&model_tag) {
  size_t num_threads = 1;
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#pragma omp single
  { num_threads = omp_get_max_threads(); }

  std::vector<trng::lcg64> generator(num_threads, gen);
  for (size_t i = 0; i < num_threads; ++i)
    generator[i].split(world_size * num_threads, rank * num_threads + i);

  num_samples /= world_size;
  auto sampled_graphs = SampleFrom(G, num_samples, generator,
                                   std::forward<diff_model_tag>(model_tag));

  spdlog::get("console")->info("Done with Sampling");
  auto S =
      mpi::SeedSelection(G, sampled_graphs.begin(), sampled_graphs.end(), k);

  return S;
}
}  // namespace mpi
}  // namespace ripples

#endif
