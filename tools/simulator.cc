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

#include <string>

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/loaders.h"

#include "omp.h"

namespace ripples {

struct SimulatorConfiguration {
  std::string EFileName;
  std::string diffusionModel;
  std::size_t Replicas;

  void addCmdOptions(CLI::App &app) {
    app.add_option("-e,--experiment-file", EFileName,
                   "The file storing the experiments form a run of an inf-max "
                   "algorithm.")
        ->group("Simulator Options")
        ->required();
    app.add_option("--replicas", Replicas,
                   "The number of experimental replicas.")
        ->group("Simulator Options")
        ->required();
  }
};

template <typename Sims>
auto GetExperimentRecord(const ToolConfiguration<SimulatorConfiguration> &CFG,
                         const nlohmann::json & experimentRecord,
                         const Sims &experiments) {
  nlohmann::json experiment{{"Input", experimentRecord["Input"]},
                            {"Output", CFG.OutputFile},
                            {"Algorithm", experimentRecord["Algorithm"]},
                            {"DiffusionModel", CFG.diffusionModel},
                            {"Epsilon", experimentRecord["Epsilon"]},
                            {"K", experimentRecord["K"]},
                            {"Seeds", experimentRecord["Seeds"]},
                            {"Simulations", experiments}};
  return experiment;
}

}  // namespace ripples

using Configuration =
    ripples::ToolConfiguration<ripples::SimulatorConfiguration>;

int main(int argc, char **argv) {
  Configuration CFG;
  CFG.ParseCmdOptions(argc, argv);

  auto simRecord =
      spdlog::rotating_logger_st("simRecord", CFG.OutputFile, 0, 3);
  simRecord->set_pattern("%v");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  nlohmann::json experimentRecord;

  std::ifstream experimentRecordIS(CFG.EFileName);

  experimentRecordIS >> experimentRecord;
  CFG.diffusionModel = experimentRecord[0]["DiffusionModel"];

  using Graph =
      ripples::Graph<uint32_t, float, ripples::ForwardDirection<uint32_t>>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading ...");
  Graph G = ripples::loadGraph<Graph>(CFG, weightGen);
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json simRecordLog;
  for (auto &record : experimentRecord) {
    using vertex_type = typename Graph::vertex_type;

    std::vector<std::pair<size_t, size_t>> experiments(CFG.Replicas);

    std::vector<vertex_type> seeds = record["Seeds"];

    G.transformID(seeds.begin(), seeds.end(), seeds.begin());

    std::vector<trng::lcg64> generator;
#pragma omp single
    generator.resize(omp_get_max_threads());

#pragma omp parallel
    {
      generator[omp_get_thread_num()].seed(0UL);
      generator[omp_get_thread_num()].split(2, 1);
      generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                            omp_get_thread_num());
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < experiments.size(); ++i) {
      if (CFG.diffusionModel == "IC") {
        experiments[i] = simulate(G, seeds.begin(), seeds.end(),
                                  generator[omp_get_thread_num()],
                                  ripples::independent_cascade_tag{});
      } else if (CFG.diffusionModel == "LT") {
        experiments[i] = simulate(G, seeds.begin(), seeds.end(),
                                  generator[omp_get_thread_num()],
                                  ripples::linear_threshold_tag{});
      } else {
        throw std::string("Not Yet Implemented");
      }
    }
    simRecordLog.push_back(ripples::GetExperimentRecord(
        CFG, record, experiments));
  }
  simRecord->info("{}", simRecordLog.dump(2));

  return EXIT_SUCCESS;
}
