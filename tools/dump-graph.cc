//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <experimental/filesystem>
#include <iostream>
#include <string>

#include "im/configuration.h"
#include "im/graph.h"
#include "im/loaders.h"
#include "im/graph_dump.h"

#include "CLI11/CLI11.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"


struct DumpOutputConfiguration {
  std::string OName{"output"};
  bool binaryDump{false};

  void addCmdOptions(CLI::App &app) {
    app.add_option("-o,--output", OName,
                   "The name of the output file name")
        ->required()
        ->group("Output Options");
    app.add_flag("--dump-binary", binaryDump,
                 "Dump the Graph in binary format.")
        ->group("Output Options");
  }
};


struct DumpConfiguration {
  std::string diffusionModel{"IC"};  //!< The diffusion model to use.

  void addCmdOptions(CLI::App &app) {
    app.add_option("-d,--diffusion-model", diffusionModel,
                   "The diffusion model to be used (LT|IC)")
        ->required()
        ->group("Tool Options");
  }
};


using Configuration =
    im::ToolConfiguration<DumpConfiguration, DumpOutputConfiguration>;


int main(int argc, char **argv) {
  Configuration CFG;
  CFG.ParseCmdOptions(argc, argv);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  spdlog::set_level(spdlog::level::info);

  using Graph = im::Graph<uint32_t, float, im::ForwardDirection<uint32_t>>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");
  Graph G = im::loadGraph<Graph>(CFG, weightGen);
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  if (CFG.binaryDump) {
    // Dump in binary format
    auto file = std::fstream(CFG.OName, std::ios::out | std::ios::binary);
    G.dump_binary(file);
    file.close();
  } else {
    auto file = std::fstream(CFG.OName, std::ios::out);
    dumpGraph(G, file);
    file.close();
  }

  return EXIT_SUCCESS;
}
