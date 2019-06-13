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

#include <string>

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"

#include "im/configuration.h"
#include "im/diffusion_simulation.h"
#include "im/graph.h"
#include "im/loaders.h"

#include "omp.h"

namespace im {

struct SimulatorConfiguration {
  std::string EFileName;
  std::string diffusionModel;
  std::size_t Replicas;

  void addCmdOptions(CLI::App &app) {
    app.add_option(
        "-e,--experiment-file", EFileName,
        "The file storing the experiments form a run of an inf-max algorithm.")
        ->group("Simulator Options")
        ->required();
    app.add_option("--replicas", Replicas,
                   "The number of experimental replicas.")
        ->group("Simulator Options")
        ->required();
  }
};


template <typename Sims>
auto GetExperimentRecord(const SimulatorConfiguration &CFG,
                         size_t seeds, float epsilon, const Sims &experiments) {
  nlohmann::json experiment{
      {"Algorithm", "IMM"},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", epsilon},
      {"K", seeds},
      {"Simulations", experiments}
  };
  return experiment;
}

}  // namespace im

using Configuration =
    im::ToolConfiguration<im::SimulatorConfiguration>;

int main(int argc, char **argv) {
  Configuration CFG;
  CFG.ParseCmdOptions(argc, argv);

  auto simRecord = spdlog::rotating_logger_st("simRecord", CFG.OutputFile, 0, 3);
  simRecord->set_pattern("%v");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  nlohmann::json experimentRecord;

  std::ifstream experimentRecordIS(CFG.EFileName);

  experimentRecordIS >> experimentRecord;
  CFG.diffusionModel = experimentRecord[0]["DiffusionModel"];

  using Graph = im::Graph<uint32_t, float, im::ForwardDirection<uint32_t>>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading ...");
  Graph G = im::loadGraph<Graph>(CFG, weightGen);
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
                                    im::independent_cascade_tag{});
        } else if (CFG.diffusionModel == "LT") {
          experiments[i] = simulate(G, seeds.begin(), seeds.end(),
                                    generator[omp_get_thread_num()],
                                    im::linear_threshold_tag{});
        } else {
          throw std::string("Not Yet Implemented");
        }
      }
      simRecordLog.push_back(
          im::GetExperimentRecord(CFG, std::distance(seeds.begin(), seeds.end()), record["Epsilon"], experiments));
  }
  simRecord->info("{}", simRecordLog.dump(2));

  return EXIT_SUCCESS;
}
