//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <string>

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/sinks/basic_file_sink.h"
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
  std::size_t Tries;

  void addCmdOptions(CLI::App &app) {
    app.add_option(
        "-e,--experiment-file", EFileName,
        "The file storing the experiments form a run of an inf-max algorithm.")
        ->group("Simulator Options")
        ->required();
    app.add_option("-d,--diffusion-model", diffusionModel,
                   "The diffusion process to simulate on the input network.")
        ->group("Simulator Options")
        ->required();
    app.add_option("--replicas", Replicas,
                   "The number of experimental replicas.")
        ->group("Simulator Options")
        ->required();
    app.add_option("--tries", Tries, "The number of tries for each replica.")
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

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

  auto simRecord = spdlog::basic_logger_st("simRecord", CFG.OutputFile);
  simRecord->set_pattern("%v");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  auto edgeList = im::loadEdgeList<im::Edge<uint32_t, float>>(CFG, weightGen);
  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json experimentRecord;

  std::ifstream experimentRecordIS(CFG.EFileName);

  experimentRecordIS >> experimentRecord;

  nlohmann::json simRecordLog;
  for (auto &record : experimentRecord) {
    using vertex_type = typename im::Graph<uint32_t, float>::vertex_type;

    std::vector<std::vector<std::pair<size_t, size_t>>> experiments(
        CFG.Replicas, std::vector<std::pair<size_t, size_t>>(CFG.Tries));

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

    for (auto itr = seeds.begin() + 1; itr <= seeds.end(); ++itr) {
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < experiments.size(); ++i) {
        for (auto &v : experiments[i]) {
          v = simulate(G, seeds.begin(), itr,
                       generator[omp_get_thread_num()],
                       im::independent_cascade_tag{});
        }
      }
      simRecordLog.push_back(
          im::GetExperimentRecord(CFG, std::distance(seeds.begin(), itr), record["Epsilon"], experiments));
    }
  }
  simRecord->info("{}", simRecordLog.dump(2));

  return EXIT_SUCCESS;
}
