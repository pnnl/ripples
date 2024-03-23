//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024, Battelle Memorial Institute
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

#ifndef RIPPLES_MPI_OPIMC_H
#define RIPPLES_MPI_OPIMC_H

#include "ripples/opimc.h"
#include "ripples/mpi/find_most_influential.h"

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

template <typename RRRSetsTy, typename SeedSetTy>
size_t FindGlobalCoverage(const RRRSetsTy &RR, const SeedSetTy &S) {
  size_t globalResult = 0;
  size_t localResult = FindCoverage(RR, S);

  MPI_Allreduce(&localResult, &globalResult, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  return globalResult;
}

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag, typename ExTagTrait>
std::vector<typename GraphTy::vertex_type> OPIMC(const GraphTy &G,
                                                 const ConfTy &CFG, double l,
                                                 GeneratorTy &gen,
                                                 OPIMCExecutionRecord &record,
                                                 diff_model_tag &&model_tag,
                                                 ExTagTrait &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  constexpr float A = 0.6321203113733684;  // 1 - 1/e

  auto console = spdlog::get("console");

  size_t thetaZero = ThetaZero(G.num_nodes(), CFG.epsilon, CFG.k, CFG.delta);
  record.ThetaZero = thetaZero;
  size_t thetaMax = ThetaMax(G.num_nodes(), CFG.epsilon, CFG.k, CFG.delta);
  record.ThetaMax = thetaMax;

  size_t iMax = ceil(log2(thetaMax / thetaZero));
  float delta = CFG.delta / (3 * iMax);

  // Compute the local work
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  thetaZero = ceil(thetaZero / world_size);

  RRRsetAllocator<GraphTy> allocator;
  RRRsets<GraphTy> R1(thetaZero, RRRset<GraphTy>(allocator));
  RRRsets<GraphTy> R2(thetaZero, RRRset<GraphTy>(allocator));

  auto timeGenerateRRRSetStart = std::chrono::high_resolution_clock::now();
  GenerateRRRSets(G, gen, R1.begin(), R1.end(), record,
                  std::forward<diff_model_tag>(model_tag),
                  typename ExTagTrait::generate_ex_tag{});
  GenerateRRRSets(G, gen, R2.begin(), R2.end(), record,
                  std::forward<diff_model_tag>(model_tag),
                  typename ExTagTrait::generate_ex_tag{});
  auto timeGenerateRRRSetEnd = std::chrono::high_resolution_clock::now();

  record.GenerateRRRSets.push_back(timeGenerateRRRSetEnd -
                                   timeGenerateRRRSetStart);
  record.RRRSetsGenerated.push_back(R1.size() + R2.size());

  std::vector<vertex_type> results;

  for (size_t i = 0;; ++i) {
    console->info("Round {}", i);

    auto timeFindMostInfluentialStart =
        std::chrono::high_resolution_clock::now();
    const auto &[coverage1, seeds] = FindMostInfluentialSet(
        G, CFG, R1, gen.isGpuEnabled(),
        typename ExTagTrait::seed_selection_ex_tag{});
    auto timeFindMostInfluentialEnd = std::chrono::high_resolution_clock::now();

    record.FindMostInfluentialSet.push_back(timeFindMostInfluentialEnd -
                                            timeFindMostInfluentialStart);

    console->info("Coverage1: {}/{}", coverage1 * R1.size(), R1.size());
    size_t coverage2 = FindGlobalCoverage(R2, seeds);
    console->info("Coverage2: {}/{}", coverage2, R2.size());

    float upperBound = UpperBound(coverage1 * R1.size() * world_size, delta,
                                  G.num_nodes(), R1.size());
    float lowerBound = LowerBound(coverage2, delta, G.num_nodes(), R2.size());

    float alpha = lowerBound / upperBound;

    console->info("Alpha: {}/{} = {}", lowerBound, upperBound, alpha);

    results = std::move(seeds);
    if (alpha >= (A - CFG.epsilon)) break;

    thetaZero *= 2;

    size_t delta = thetaZero - R1.size();

    R1.insert(R1.end(), delta, RRRset<GraphTy>(allocator));
    R2.insert(R2.end(), delta, RRRset<GraphTy>(allocator));

    auto begin1 = R1.end() - delta;
    auto begin2 = R2.end() - delta;

    auto timeGenerateRRRSetStart = std::chrono::high_resolution_clock::now();
    GenerateRRRSets(G, gen, begin1, R1.end(), record,
                    std::forward<diff_model_tag>(model_tag),
                    typename ExTagTrait::generate_ex_tag{});
    GenerateRRRSets(G, gen, begin2, R2.end(), record,
                    std::forward<diff_model_tag>(model_tag),
                    typename ExTagTrait::generate_ex_tag{});
    auto timeGenerateRRRSetEnd = std::chrono::high_resolution_clock::now();

    record.GenerateRRRSets.push_back(timeGenerateRRRSetEnd -
                                     timeGenerateRRRSetStart);
    record.RRRSetsGenerated.push_back(std::distance(begin1, R1.end()) +
                                      std::distance(begin2, R2.end()));
  }

  return results;
}

}
}

#endif
