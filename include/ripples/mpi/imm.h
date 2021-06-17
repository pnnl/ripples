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

#ifndef RIPPLES_MPI_IMM_H
#define RIPPLES_MPI_IMM_H

#include "mpi.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "trng/lcg64.hpp"

#include "ripples/generate_rrr_sets.h"
#include "ripples/imm.h"
#include "ripples/imm_execution_record.h"
#include "ripples/mpi/find_most_influential.h"
#include "ripples/utility.h"

namespace ripples {
namespace mpi {

template <typename ex_tag>
struct MPI_Plus_X {
  // using generate_ex_tag
  // using seed_selection_ex_tag
};

template <>
struct MPI_Plus_X<mpi_omp_parallel_tag> {
  using generate_ex_tag = omp_parallel_tag;
  using seed_selection_ex_tag = mpi_omp_parallel_tag;
};

//! Compute ThetaPrime for the MPI implementation.
//!
//! \param x The index of the current iteration.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
inline size_t ThetaPrime(ssize_t x, double epsilonPrime, double l, size_t k,
                         size_t num_nodes, mpi_omp_parallel_tag &&) {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  return (ThetaPrime(x, epsilonPrime, l, k, num_nodes, omp_parallel_tag{}) /
          world_size) +
         1;
}

//! Split a random number generator into one sequence per MPI rank.
//!
//! \tparam PRNG The type of the random number generator.
//!
//! \param gen The parallel random number generator to split.
template <typename PRNG>
void split_generator(PRNG &gen) {
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  gen.split(world_size, world_rank);
}

template <typename PRNG>
std::vector<PRNG> rank_split_generator(const PRNG &gen) {
  size_t max_num_threads(1);

#pragma omp single
  max_num_threads = omp_get_max_threads();

  std::vector<trng::lcg64> generator(max_num_threads, gen);

#pragma omp parallel
  {
    generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                          omp_get_thread_num());
  }
  return generator;
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
template <typename GraphTy, typename ConfTy, typename PRNGeneratorTy,
          typename diff_model_tag, typename ExTagTrait>
auto Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              PRNGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, ExTagTrait &&) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  #if defined ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator("/pmem1", 0);
  #elif defined ENABLE_METALL
  RRRsetAllocator<vertex_type> allocator =  metall_manager_instance().get_allocator();
#else
  RRRsetAllocator<vertex_type> allocator;
  #endif
  std::vector<RRRset<GraphTy>> RR;

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    mpi_omp_parallel_tag{});

    size_t delta = thetaPrime - RR.size();
    record.ThetaPrimeDeltas.push_back(thetaPrime - RR.size());

    auto timeRRRSets = measure<>::exec_time([&]() {
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      typename ExTagTrait::generate_ex_tag{});
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    double f;
    auto timeMostInfluential = measure<>::exec_time([&]() {
      const auto &S =
          FindMostInfluentialSet(G, CFG, RR, generator.isGpuEnabled(),
                                 typename ExTagTrait::seed_selection_ex_tag{});
      f = S.first;
    });
    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      // std::cout << "Fraction " << f << std::endl;
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
    size_t final_delta = thetaLocal - RR.size();
    RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

    auto begin = RR.end() - final_delta;

    GenerateRRRSets(G, generator, begin, RR.end(), record,
                    std::forward<diff_model_tag>(model_tag),
                    typename ExTagTrait::generate_ex_tag{});
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
template <typename GraphTy, typename ConfTy, typename diff_model_tag,
          typename GeneratorTy, typename ExTagTrait>
auto IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
         IMMExecutionRecord &record, diff_model_tag &&model_tag,
         ExTagTrait &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  auto R = mpi::Sampling(G, CFG, l, gen, record,
                         std::forward<diff_model_tag>(model_tag),
                         std::forward<ExTagTrait>(ex_tag));

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S =
      FindMostInfluentialSet(G, CFG, R, gen.isGpuEnabled(),
                             typename ExTagTrait::seed_selection_ex_tag{});
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  return S.second;
}

}  // namespace mpi
}  // namespace ripples

#endif  // RIPPLES_MPI_IMM_H
