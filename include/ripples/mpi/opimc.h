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

template <typename RRRSetsTy, typename SeedSetTy>
size_t FindGlobalCoverage(const RRRSetsTy &RR, const size_t &RRSize, const SeedSetTy &S) {
  size_t globalResult = 0;
  size_t localResult = FindCoverage(RR, RRSize, S);

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
  float delta_opimc = CFG.delta / (3 * iMax);

  // Compute the local work
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  size_t thetaPrime = ceil(thetaZero / world_size);

  RRRsetAllocator<GraphTy> allocator;
  RRRsets<GraphTy> R1(thetaPrime, RRRset<GraphTy>(allocator));
  RRRsets<GraphTy> R2(thetaPrime, RRRset<GraphTy>(allocator));

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
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  #ifdef PRINTF_TIL_YOU_DROP
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      timeGenerateRRRSetEnd - timeGenerateRRRSetStart);
  size_t rr_set_size = 0;
  for (const auto &rr_set : R1) {
    rr_set_size += rr_set.size();
  }
  for (const auto &rr_set : R2) {
    rr_set_size += rr_set.size();
  }
  console->info("Rank = {}, RRSetsGenerated = {}, Time = {}, Total RRSetSize = {}", world_rank,
                R1.size() + R2.size(), duration_ms.count(), rr_set_size);
  #endif // PRINTF_TIL_YOU_DROP

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

    #ifdef SPECULATIVE_EXECUTION
    
    bool use_speculative_results = true;
    size_t R1Size_prev = R1.size();
    size_t R2Size_prev = R2.size();

    if(world_rank != 0) {

      thetaPrime = ceil((thetaZero * (2 << i)) / (world_size - 1));

      size_t delta = thetaPrime - R1.size();

      // Check to see if there is enough free memory to allocate the new RRRsets
      size_t free_memory = GetAvailableMemory();
      size_t num_rr_sets = R1.size() + R2.size();
      size_t rr_set_size = 0;
      #pragma omp parallel
      {
        #pragma omp for reduction(+:rr_set_size)
        for (const auto &rr_set : R1) {
          rr_set_size += rr_set.size();
        }
        #pragma omp for reduction(+:rr_set_size)
        for (const auto &rr_set : R2) {
          rr_set_size += rr_set.size();
        }
      }
      size_t average_rr_set_size = rr_set_size / num_rr_sets;
      size_t new_rr_set_size = average_rr_set_size * delta;
      if (new_rr_set_size > free_memory) {
        console->info("Rank = {}, Not enough memory to allocate new RRRsets", world_rank);
        use_speculative_results = false;
      }
      else{
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
        #ifdef PRINTF_TIL_YOU_DROP
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          timeGenerateRRRSetEnd - timeGenerateRRRSetStart);
        rr_set_size = 0;
        for (const auto &rr_set : R1) {
          rr_set_size += rr_set.size();
        }
        for (const auto &rr_set : R2) {
          rr_set_size += rr_set.size();
        }
        console->info(
            "Rank = {}, RRSetsGenerated = {}, Time = {}, Total RRSetSize = {}", world_rank,
            std::distance(begin1, R1.end()) + std::distance(begin2, R2.end()),
            duration_ms.count(), rr_set_size);
        #endif  // PRINTF_TIL_YOU_DROP
      }
    }
    size_t R1_total_size = R1Size_prev;
    MPI_Allreduce(MPI_IN_PLACE, &R1_total_size, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    uint32_t coverage1_nonconst = coverage1;
    MPI_Bcast(&coverage1_nonconst, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    console->info("Coverage1: {}/{}", coverage1_nonconst, R1_total_size);

    size_t R2_total_size = R2Size_prev;
    MPI_Allreduce(MPI_IN_PLACE, &R2_total_size, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    results.resize(CFG.k);
    if(world_rank == 0) {
      results = std::move(seeds);
    }
    MPI_Bcast(results.data(), CFG.k, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    size_t coverage2 = FindGlobalCoverage(R2, R2Size_prev, results);
    console->info("Coverage2: {}/{}", coverage2, R2_total_size);

    float upperBound = UpperBound(coverage1_nonconst, delta_opimc,
                                  G.num_nodes(), R1_total_size);
    float lowerBound = LowerBound(coverage2, delta_opimc, G.num_nodes(), R2_total_size);

    float alpha = lowerBound / upperBound;

    console->info("Alpha: {}/{} = {}", lowerBound, upperBound, alpha);

    if (alpha >= (A - CFG.epsilon)) break;
    // Reduce use_speculative_results
    MPI_Allreduce(MPI_IN_PLACE, &use_speculative_results, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (!use_speculative_results){
      console->info("Out of memory! Ending early with current results.");
      break;
    }
    #else
    console->info("Coverage1: {}/{}", coverage1 * R1.size() * world_size, R1.size() * world_size);
    size_t coverage2 = FindGlobalCoverage(R2, seeds);
    console->info("Coverage2: {}/{}", coverage2, R2.size() * world_size);

    float upperBound = UpperBound(coverage1 * R1.size() * world_size, delta_opimc,
                                  G.num_nodes(), R1.size() * world_size);
    float lowerBound = LowerBound(coverage2, delta_opimc, G.num_nodes(), R2.size() * world_size);

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
    #ifdef PRINTF_TIL_YOU_DROP
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      timeGenerateRRRSetEnd - timeGenerateRRRSetStart);
    size_t rr_set_size = 0;
    for (const auto &rr_set : R1) {
      rr_set_size += rr_set.size();
    }
    for (const auto &rr_set : R2) {
      rr_set_size += rr_set.size();
    }
    console->info(
        "Rank = {}, RRSetsGenerated = {}, Time = {}, Total RRSetSize = {}", world_rank,
        std::distance(begin1, R1.end()) + std::distance(begin2, R2.end()),
        duration_ms.count(), rr_set_size);
    #endif  // PRINTF_TIL_YOU_DROP

    #endif // SPECULATIVE_EXECUTION

  }

  return results;
}

}
}

#endif
