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
#include "ripples/louvain_imm.h"
#include "ripples/utility.h"

#include "omp.h"

namespace ripples {
auto GetWalkIterationRecord(
    const typename IMMExecutionRecord::walk_iteration_prof &iter) {
  nlohmann::json res{{"NumSets", iter.NumSets}, {"Total", iter.Total}};
  for (size_t wi = 0; wi < iter.CPUWalks.size(); ++wi) {
    std::stringstream wname;
    wname << "CPU Worker " << wi;
    res[wname.str()] = nlohmann::json{{"NumSets", iter.CPUWalks[wi].NumSets},
                                      {"Total", iter.CPUWalks[wi].Total}};
  }
  for (size_t wi = 0; wi < iter.GPUWalks.size(); ++wi) {
    std::stringstream wname;
    wname << "GPU Worker " << wi;
    res[wname.str()] = nlohmann::json{{"NumSets", iter.GPUWalks[wi].NumSets},
                                      {"Total", iter.GPUWalks[wi].Total},
                                      {"Kernel", iter.GPUWalks[wi].Kernel},
                                      {"D2H", iter.GPUWalks[wi].D2H},
                                      {"Post", iter.GPUWalks[wi].Post}};
  }
  return res;
}

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<LouvainIMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  nlohmann::json experiment{
      {"Algorithm", "IMM"},
      {"Input", CFG.IFileName},
      {"Output", CFG.OutputFile},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"K", CFG.k},
      {"L", 1},
      {"NumThreads", R.NumThreads},
      {"NumWalkWorkers", CFG.streaming_workers},
      {"NumGPUWalkWorkers", CFG.streaming_gpu_workers},
      {"Total", R.Total},
      {"ThetaPrimeDeltas", R.ThetaPrimeDeltas},
      {"ThetaEstimation", R.ThetaEstimationTotal},
      {"ThetaEstimationGenerateRRR", R.ThetaEstimationGenerateRRR},
      {"ThetaEstimationMostInfluential", R.ThetaEstimationMostInfluential},
      {"Theta", R.Theta},
      {"Counting", R.Counting},
      {"Pivoting", R.Pivoting},
      {"RRRSetSizeBytes", R.RRRSetSize},
      {"GenerateRRRSets", R.GenerateRRRSets},
      {"FindMostInfluentialSet", R.FindMostInfluentialSet},
      {"Seeds", seeds}};
  for (auto &ri : R.WalkIterations) {
    experiment["Iterations"].push_back(GetWalkIterationRecord(ri));
  }
  return experiment;
}

}  // namespace ripples

ripples::ToolConfiguration<ripples::LouvainIMMConfiguration> CFG;

void parse_command_line(int argc, char **argv) {
  CFG.ParseCmdOptions(argc, argv);
#pragma omp single
  CFG.streaming_workers = omp_get_max_threads();

  if (CFG.seed_select_max_workers == 0)
    CFG.seed_select_max_workers = CFG.streaming_workers;
  if (CFG.seed_select_max_gpu_workers == std::numeric_limits<size_t>::max())
    CFG.seed_select_max_gpu_workers = CFG.streaming_gpu_workers;
}

int main(int argc, char *argv[]) {
  auto console = spdlog::stdout_color_st("console");
  parse_command_line(argc, argv);

  spdlog::set_level(spdlog::level::info);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using dest_type = ripples::WeightedDestination<uint32_t, float>;
  using GraphFwd =
      ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>>;
  using GraphBwd =
      ripples::Graph<uint32_t, dest_type, ripples::BackwardDirection<uint32_t>>;
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

  const auto communities =
      ripples::getCommunitiesSubgraphs<GraphBwd>(Gf, communityVector);
  console->info("Number of Communities : {}", communities.size());

  nlohmann::json executionLog;

  std::vector<typename GraphBwd::vertex_type> seeds;
  std::vector<ripples::IMMExecutionRecord> R(communities.size());

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  using StreamingGeneratorIC = ripples::StreamingRRRGenerator<
      GraphBwd, decltype(generator),
      typename ripples::RRRsets<GraphBwd>::iterator,
      ripples::independent_cascade_tag>;
  using StreamingGeneratorLT = ripples::StreamingRRRGenerator<
      GraphBwd, decltype(generator),
      typename ripples::RRRsets<GraphBwd>::iterator,
      ripples::linear_threshold_tag>;

  std::ofstream perf(CFG.OutputFile);
  if (CFG.parallel) {
    auto workers = CFG.streaming_workers;
    auto gpu_workers = CFG.streaming_gpu_workers;

    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<StreamingGeneratorIC> gen;
      for (size_t i = 0; i < communities.size(); ++i) {
        auto local_gen = generator;
        local_gen.split(communities.size(), i);
        gen.push_back(StreamingGeneratorIC(communities[i], local_gen, R[i],
                                           workers - gpu_workers, gpu_workers,
                                           CFG.worker_to_gpu));
      }
      std::tie(seeds, R) = LouvainIMM(communities, CFG, 1, gen,
                                      ripples::independent_cascade_tag{},
                                      ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R[0].Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<StreamingGeneratorLT> gen;
      for (size_t i = 0; i < communities.size(); ++i) {
        auto local_gen = generator;
        local_gen.split(communities.size(), i);
        gen.push_back(StreamingGeneratorLT(communities[i], local_gen, R[i],
                                           workers - gpu_workers, gpu_workers,
                                           CFG.worker_to_gpu));
      }
      std::tie(seeds, R) =
          LouvainIMM(communities, CFG, 1, gen, ripples::linear_threshold_tag{},
                     ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R[0].Total = end - start;
    }
    console->info("Louvain IMM parallel : {}ms", R[0].Total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();

    for (auto &record : R) {
      record.NumThreads = num_threads;
      auto experiment = GetExperimentRecord(CFG, record, seeds);
      executionLog.push_back(experiment);
    }
    perf << executionLog.dump(2);
  } else {
    std::vector<trng::lcg64> gen;
    for (size_t i = 0; i < communities.size(); ++i) {
      auto local_gen = generator;
      local_gen.split(communities.size(), i);
      gen.emplace_back(local_gen);
    }
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) = LouvainIMM(communities, CFG, 1, gen, R,
                                      ripples::independent_cascade_tag{},
                                      ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R[0].Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) = LouvainIMM(communities, CFG, 1, gen, R,
                                      ripples::linear_threshold_tag{},
                                      ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R[0].Total = end - start;
    }
    console->info("Louvain IMM squential : {}ms", R[0].Total.count());

    for (auto &record : R) {
      record.NumThreads = 1;
      auto experiment = GetExperimentRecord(CFG, record, seeds);
      executionLog.push_back(experiment);
    }
    perf << executionLog.dump(2);
  }

  return EXIT_SUCCESS;
}
