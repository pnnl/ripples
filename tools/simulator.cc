//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <string>

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"


#include "im/graph.h"
#include "im/loaders.h"
#include "im/diffusion_simulation.h"

#include "omp.h"

namespace im {

struct SimulatorConfiguration {
  std::string IFileName;
  std::string OFileName;
  std::string EFileName;
  std::string DiffusionModel;
  std::size_t Replicas;
  std::size_t Tries;
};

auto ParseCmdOptions(int argc, char **argv) {
  SimulatorConfiguration CFG;
  CLI::App app("Yet Another tool to simulate spread in social networks");
  app.add_option("-i,--input-grah", CFG.IFileName,
                 "The input file storing the edge-list.")->required();
  app.add_option("-e,--experiment-file", CFG.EFileName,
                 "The file storing the experiments form a run of an inf-max algorithm.")
      ->required();
  app.add_option("-d,--diffusion-model", CFG.DiffusionModel,
                 "The diffusion process to simulate on the input network.")
      ->required();
  app.add_option("-o,--output", CFG.OFileName,
                 "The file where to store the results of the simulations");
  app.add_option("--replicas", CFG.Replicas,
                 "The number of experimental replicas.");
  app.add_option("--tries", CFG.Tries,
                 "The number of tries for each replica.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return CFG;
}

}

int main(int argc, char **argv) {
  im::SimulatorConfiguration CFG = im::ParseCmdOptions(argc, argv);

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

  auto edgeList =
      im::load<im::Edge<uint32_t, float>>(CFG.IFileName, im::edge_list_tsv());
  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  using vertex_type = typename im::Graph<uint32_t, float>::vertex_type;

  std::vector<std::vector<size_t>> experiments(CFG.Replicas, std::vector<size_t>(CFG.Tries));

  std::vector<trng::lcg64> generator;
  #pragma omp single
  generator.resize(omp_get_max_threads());

  #pragma omp parallel
  {
    generator[omp_get_thread_num()].seed(0UL);
    generator[omp_get_thread_num()].split(omp_get_num_threads(), omp_get_thread_num());
  }

  #pragma omp parallel for
  for (size_t i = 0; i < experiments.size(); ++i) {
    trng::uniform_int_dist seed(0, G.num_nodes() - 1);
    std::vector<vertex_type> seeds(10);

    for (auto & v : experiments[i]) {
      std::generate(seeds.begin(), seeds.end(),
                    [&]() -> auto { return seed(generator[omp_get_thread_num()]);});
      v = simulate(G, seeds.begin(), seeds.end(),
                   generator[omp_get_thread_num()],
                   im::independent_cascade_tag{});
    }
  }

  nlohmann::json experiments_log;

  for (auto & replica : experiments) {
    experiments_log["simulations"].push_back(replica);
    for (auto & v : replica)
      console->info("Spread : {}", v);
  }

  console->info("{}", experiments_log.dump(2));

  return EXIT_SUCCESS;
}
