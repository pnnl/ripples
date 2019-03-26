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

#include "CLI11/CLI11.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"


struct DumpOutputConfiguration {
  std::string OName{"output.bin"};

  void addCmdOptions(CLI::App &app) {
    app.add_option("-o,--output", OName,
                   "The name of the output file name")
        ->required()
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

  using Graph = im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");
  Graph G = im::loadGraph<Graph>(CFG, weightGen);
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  // Dump in binary format
  auto file = std::fstream(CFG.OName, std::ios::out | std::ios::binary);
  G.dump_binary(file);
  file.close();

  return EXIT_SUCCESS;
}
