//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_MPI_IMM_H
#define RIPPLES_MPI_IMM_H

#include "mpi.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "trng/lcg64.hpp"

#include "ripples/generate_rrr_sets.h"
#include "ripples/imm.h"
#include "ripples/mpi/find_most_influential.h"
#include "ripples/utility.h"

namespace ripples {

//! Compute ThetaPrime for the MPI implementation.
//!
//! \param x The index of the current iteration.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
inline size_t ThetaPrime(ssize_t x, double epsilonPrime, double l, size_t k,
                         size_t num_nodes, mpi_omp_parallel_tag &) {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  return (ThetaPrime(x, epsilonPrime, l, k, num_nodes, omp_parallel_tag{}) /
          world_size) +
         1;
}

//! Collect a set of Random Reverse Reachable set.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam PRNGeneratorTy The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param generator The parallel random number generator.
//! \param record Data structure storing timing and event counts.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
auto Sampling(const GraphTy &G, std::size_t k, double epsilon, double l,
              PRNGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, mpi_omp_parallel_tag &ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  std::vector<RRRset<GraphTy>> RR;

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime =
        ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(), ex_tag);

    record.ThetaPrimeDeltas.push_back(thetaPrime - RR.size());

    auto timeRRRSets = measure<>::exec_time([&]() {
      auto deltaRR = GenerateRRRSets(G, thetaPrime - RR.size(), generator,
                                     std::forward<diff_model_tag>(model_tag),
                                     omp_parallel_tag{});

      RR.insert(RR.end(), std::make_move_iterator(deltaRR.begin()),
                std::make_move_iterator(deltaRR.end()));
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    double f;
    auto timeMostInfluential = measure<>::exec_time([&]() {
      const auto &S = FindMostInfluentialSet(G, k, RR, ex_tag);
      f = S.first;
    });
    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  size_t thetaLocal = (theta / world_size) + 1;
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;

  start = std::chrono::high_resolution_clock::now();
  if (thetaLocal > RR.size()) {
    auto deltaRR = GenerateRRRSets(G, thetaLocal - RR.size(), generator,
                                   std::forward<diff_model_tag>(model_tag),
                                   omp_parallel_tag{});

    RR.insert(RR.end(), std::make_move_iterator(deltaRR.begin()),
              std::make_move_iterator(deltaRR.end()));
  }
  end = std::chrono::high_resolution_clock::now();

  record.GenerateRRRSets = end - start;

  return RR;
}

//! The IMM algroithm for Influence Maximization (MPI specialization).
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename diff_model_tag, typename PRNG>
auto IMM(const GraphTy &G, std::size_t k, double epsilon, double l, PRNG &gen,
         diff_model_tag &&model_tag, ripples::mpi_omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  IMMExecutionRecord record;

  size_t max_num_threads(1);

#pragma omp single
  max_num_threads = omp_get_max_threads();

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  gen.split(world_size, world_rank);

  std::vector<trng::lcg64> generator(max_num_threads, gen);

#pragma omp parallel
  {
    generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                          omp_get_thread_num());
  }

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  auto R = Sampling(G, k, epsilon, l, generator, record,
                    std::forward<diff_model_tag>(model_tag), ex_tag);

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S = FindMostInfluentialSet(G, k, R, ex_tag);
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  return std::make_pair(S.second, record);
}

}  // namespace ripples

#endif  // RIPPLES_MPI_IMM_H
