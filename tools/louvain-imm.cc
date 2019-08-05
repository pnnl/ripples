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

#include <algorithm>
#include <fstream>
#include <iterator>

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/loaders.h"
#include "ripples/utility.h"
#include "ripples/louvain_imm.h"

#include "omp.h"

namespace ripples {

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<ripples::LouvainIMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  nlohmann::json experiment{
      {"Algorithm", "CommunityDetection-IMM"},
      {"Input", CFG.IFileName },
      {"Output", CFG.OutputFile},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"K", CFG.k},
      {"L", 1},
      {"NumThreads", R.NumThreads},
      {"Total", R.Total},
      {"ThetaPrimeDeltas", R.ThetaPrimeDeltas},
      {"ThetaEstimation", R.ThetaEstimationTotal},
      {"ThetaEstimationGenerateRRR", R.ThetaEstimationGenerateRRR},
      {"ThetaEstimationMostInfluential", R.ThetaEstimationMostInfluential},
      {"Theta", R.Theta},
      {"GenerateRRRSets", R.GenerateRRRSets},
      {"FindMostInfluentialSet", R.FindMostInfluentialSet},
      {"Seeds", seeds}};
  return experiment;
}

}  // namespace ripples

int main(int argc, char *argv[]) {
  ripples::ToolConfiguration<ripples::LouvainIMMConfiguration> CFG;
  CFG.ParseCmdOptions(argc, argv);

  spdlog::set_level(spdlog::level::info);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using GraphFwd =
      ripples::Graph<uint32_t, float, ripples::ForwardDirection<uint32_t>>;
  using GraphBwd =
      ripples::Graph<uint32_t, float, ripples::BackwardDirection<uint32_t>>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");
  GraphFwd Gf = ripples::loadGraph<GraphFwd>(CFG, weightGen);
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", Gf.num_nodes());
  console->info("Number of Edges : {}", Gf.num_edges());

  std::vector<typename GraphFwd::vertex_type> communityVector;
  communityVector.reserve(Gf.num_nodes());

  {
    std::ifstream f(CFG.communityList);
    std::istream_iterator<typename GraphBwd::vertex_type> eos;
    std::istream_iterator<typename GraphBwd::vertex_type> itr(f);

    std::copy(itr, eos, std::back_inserter(communityVector));
  }

  console->info("Communities Vector Size : {}", communityVector.size());


  const auto communities = ripples::getCommunitiesSubgraphs<GraphBwd>(Gf, communityVector);
  console->info("Number of Communities : {}", communities.size());

  nlohmann::json executionLog;

  std::vector<typename GraphBwd::vertex_type> seeds;
  ripples::LouvainIMMExecutionRecord R;

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  if (CFG.OMPStrongScaling) {
    size_t max_threads = 1;
    std::ofstream perf(CFG.OutputFile);
#pragma omp single
    max_threads = omp_get_max_threads();

    for (size_t num_threads = max_threads; num_threads >= 1; --num_threads) {
      if (num_threads != 1) {
        omp_set_num_threads(num_threads);

        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) = LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
                                   ripples::independent_cascade_tag{},
                                   ripples::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
                  ripples::linear_threshold_tag{}, ripples::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }

        R.NumThreads = num_threads;

        console->info("Louvain IMM parallel : {}ms, T={}/{}", R.Total.count(),
                      num_threads, max_threads);

        auto experiment = GetExperimentRecord(CFG, R, seeds);
        executionLog.push_back(experiment);
      } else {
        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) = LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
                                   ripples::independent_cascade_tag{},
                                   ripples::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
                  ripples::linear_threshold_tag{}, ripples::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }
        console->info("Louvain IMM squential : {}ms, T={}/{}", R.Total.count(),
                      num_threads, max_threads);

        R.NumThreads = num_threads;

        auto experiment = GetExperimentRecord(CFG, R, seeds);
        executionLog.push_back(experiment);
      }
      perf.seekp(0);
      perf << executionLog.dump(2);
    }
  } else if (CFG.parallel) {
    std::ofstream perf(CFG.OutputFile);
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
              ripples::independent_cascade_tag{}, ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
              ripples::linear_threshold_tag{}, ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("Louvain IMM parallel : {}ms", R.Total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);

    perf << executionLog.dump(2);
  } else {
    std::ofstream perf(CFG.OutputFile);
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
              ripples::independent_cascade_tag{}, ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          LouvainIMM(communities, CFG.k, CFG.epsilon, 1, generator,
              ripples::linear_threshold_tag{}, ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("Louvain IMM squential : {}ms", R.Total.count());

    R.NumThreads = 1;

    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);
    perf << executionLog.dump(2);
  }

  return EXIT_SUCCESS;
}
