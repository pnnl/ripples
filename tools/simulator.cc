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

#include "im/diffusion_simulation.h"
#include "im/graph.h"
#include "im/loaders.h"

#include "omp.h"

namespace im {

struct SimulatorConfiguration {
  std::string IFileName;
  std::string OFileName;
  std::string EFileName;
  std::string diffusionModel;
  bool weighted{false};
  std::size_t Replicas;
  std::size_t Tries;
  bool undirected{false};
};

auto ParseCmdOptions(int argc, char **argv) {
  SimulatorConfiguration CFG;
  CLI::App app("Yet Another tool to simulate spread in social networks");
  app.add_option("-i,--input-grah", CFG.IFileName,
                 "The input file storing the edge-list.")
      ->required();
  app.add_flag("-u,--undirected", CFG.undirected,
               "The input graph is undirected");
  app.add_option(
         "-e,--experiment-file", CFG.EFileName,
         "The file storing the experiments form a run of an inf-max algorithm.")
      ->required();
  app.add_option("-d,--diffusion-model", CFG.diffusionModel,
                 "The diffusion process to simulate on the input network.")
      ->required();
  app.add_option("-o,--output", CFG.OFileName,
                 "The file where to store the results of the simulations")
      ->required();
  app.add_option("--replicas", CFG.Replicas,
                 "The number of experimental replicas.")
      ->required();
  app.add_option("--tries", CFG.Tries, "The number of tries for each replica.")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return CFG;
}

}  // namespace im

int main(int argc, char **argv) {
  im::SimulatorConfiguration CFG = im::ParseCmdOptions(argc, argv);

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

  auto simRecord = spdlog::basic_logger_st("simRecord", CFG.OFileName);
  simRecord->set_pattern("%v");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  std::vector<im::Edge<uint32_t, float>> edgeList;
  if (CFG.weighted) {
    console->info("Loading with input weights");
    if (CFG.diffusionModel == "IC") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen,
          im::weighted_edge_list_tsv{}, im::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen,
          im::weighted_edge_list_tsv{}, im::linear_threshold_tag{});
    }
  } else {
    console->info("Loading with random weights");
    if (CFG.diffusionModel == "IC") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv{},
          im::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv{},
          im::linear_threshold_tag{});
    }
  }
  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json experimentRecord;

  std::ifstream experimentRecordIS(CFG.EFileName);

  experimentRecordIS >> experimentRecord;

  for (auto &record : experimentRecord) {
    using vertex_type = typename im::Graph<uint32_t, float>::vertex_type;

    std::vector<std::vector<std::pair<size_t, size_t>>> experiments(
        CFG.Replicas, std::vector<std::pair<size_t, size_t>>(CFG.Tries));

    std::vector<vertex_type> seeds = record["Seeds"];

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
      for (auto &v : experiments[i]) {
        v = simulate(G, seeds.begin(), seeds.end(),
                     generator[omp_get_thread_num()],
                     im::independent_cascade_tag{});
      }
    }

    for (auto &replica : experiments) {
      record["simulations"].push_back(replica);
    }
  }
  simRecord->info("{}", experimentRecord.dump(2));

  return EXIT_SUCCESS;
}
