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

#include "ripples/hill_climbing.h"
#include "ripples/configuration.h"
#include "ripples/graph.h"
#include "ripples/loaders.h"
#include "ripples/utility.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "trng/lcg64.hpp"

#include "nlohmann/json.hpp"

namespace ripples {
ToolConfiguration<HillClimbingConfiguration>& configuration() {
  static ToolConfiguration<HillClimbingConfiguration> CFG;
  return CFG;
}

template <typename SeedSet>
auto GetExperimentRecord(
    const ToolConfiguration<HillClimbingConfiguration>& CFG,
    const HillClimbingExecutionRecord& R, const SeedSet& seeds) {
  nlohmann::json experiment{{"Algorithm", "HillClimbing"},
                            {"Input", CFG.IFileName},
                            {"Output", CFG.OutputFile},
                            {"DiffusionModel", CFG.diffusionModel},
                            {"K", CFG.k},
                            {"Seeds", seeds},
                            {"NumThreads", R.NumThreads},
                            {"NumWalkWorkers", CFG.streaming_workers},
                            {"NumGPUWalkWorkers", CFG.streaming_gpu_workers},
                            {"Total", R.Total},
                            {"Sampling", R.Sampling},
                            {"SeedSelection", R.SeedSelection},
                            {"SamplingTasks", R.SamplingTasks},
                            {"SeedSelectionTasks", R.SeedSelectionTasks}};

  return experiment;
}

void parse_command_line(int argc, char** argv) {
  configuration().ParseCmdOptions(argc, argv);
#pragma omp single
  configuration().streaming_workers = omp_get_max_threads();
}
}  // namespace ripples

int main(int argc, char** argv) {
  auto console = spdlog::stdout_color_st("console");
  spdlog::set_level(spdlog::level::trace);

  ripples::parse_command_line(argc, argv);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using GraphFwd =
      ripples::Graph<uint32_t, ripples::WeightedDestination<uint32_t, float>>;
  console->info("Loading...");
  GraphFwd G =
      ripples::loadGraph<GraphFwd>(ripples::configuration(), weightGen);
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json executionLog;

  std::vector<typename GraphFwd::vertex_type> seeds;

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  ripples::HillClimbingExecutionRecord R;

  ripples::configuration().streaming_workers -=
      ripples::configuration().streaming_gpu_workers;

  if (ripples::configuration().diffusionModel == "IC") {
    auto start = std::chrono::high_resolution_clock::now();
    seeds = HillClimbing(G, ripples::configuration(), generator, R,
                         ripples::independent_cascade_tag{});
    auto end = std::chrono::high_resolution_clock::now();
    R.Total = end - start;
  } else if (ripples::configuration().diffusionModel == "LT") {
    auto start = std::chrono::high_resolution_clock::now();
    seeds = HillClimbing(G, ripples::configuration(), generator, R,
                         ripples::linear_threshold_tag{});
    auto end = std::chrono::high_resolution_clock::now();
    R.Total = end - start;
  }

  console->info("HillClimbing : {}ms", R.Total.count());

  size_t num_threads;
#pragma omp single
  num_threads = omp_get_max_threads();
  R.NumThreads = num_threads;
  std::vector<typename GraphFwd::vertex_type> out_seeds(seeds.size());
  G.convertID(seeds.begin(), seeds.end(), out_seeds.begin());

  auto experiment = GetExperimentRecord(ripples::configuration(), R, out_seeds);
  executionLog.push_back(experiment);
  std::ofstream perf(ripples::configuration().OutputFile);
  perf << executionLog.dump(2);
  return EXIT_SUCCESS;
}
