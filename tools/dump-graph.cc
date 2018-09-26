//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <experimental/filesystem>

#include "im/graph.h"
#include "im/loaders.h"

#include "spdlog/spdlog.h"
#include "CLI11/CLI11.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"

struct Configuration {
  std::string IFileName;
  bool undirected{false};
  std::string ODirName;
};

namespace im {

Configuration ParseCmdOptions(int argc, char **argv) {
  Configuration CFG;

  CLI::App app{"Dump graph in IMM and TIM formats"};

  app.add_option("-i,--input-graph", CFG.IFileName,
                 "The input file with the edge-list.")
      ->required();
  app.add_flag("-u,--undirected", CFG.undirected, "The input graph is undirected");
  app.add_option("-o,--output-dir", CFG.ODirName,
                 "The name of the output directory")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return CFG;
}

}

int main(int argc, char **argv) {
  Configuration CFG = im::ParseCmdOptions(argc, argv);

  namespace fs = std::experimental::filesystem;

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  spdlog::set_level(spdlog::level::info);

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

  auto edgeList =
      im::load<im::Edge<uint32_t, float>>(CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv());

  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());
  {
    std::ofstream attribute(CFG.ODirName + "/attribute.txt");
    attribute << "n=" << G.num_nodes() << std::endl;
    attribute << "m=" << G.num_edges() << std::endl;
  }

  {
    std::ofstream graph_ic(CFG.ODirName + "/graph_ic.inf");
    for (uint32_t v = 0; v < G.num_nodes(); ++v)
      for (auto & e : G.out_neighbors(v))
        graph_ic << v << " "
                 << e.vertex << " "
                 << e.weight << std::endl;
  }
  return EXIT_SUCCESS;
}
