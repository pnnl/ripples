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

struct DumpConfiguration {
  std::string OName{"."};
  std::string diffusionModel{"IC"};  //!< The diffusion model to use.
  bool binary{false};

  void addCmdOptions(CLI::App &app) {
    app.add_option("-o,--output", OName,
                   "The name of the output file/directory")
        ->required();
    app.add_option("-d,--diffusion-model", diffusionModel,
                   "The diffusion model to be used (LT|IC)")
        ->required();
    app.add_flag("--binary", binary,
                 "Dump in binary format");
  }
};

int main(int argc, char **argv) {
  im::ToolConfiguration<DumpConfiguration> CFG;
  CFG.ParseCmdOptions(argc, argv);

  namespace fs = std::experimental::filesystem;

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  spdlog::set_level(spdlog::level::info);

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

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
  im::Graph<uint32_t, float,im::BackwardDirection<uint32_t>> GBackward(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());
  if (!CFG.binary) {
    {
      std::ofstream attribute(CFG.OName + "/attribute.txt");
      attribute << "n=" << G.num_nodes() << std::endl;
      attribute << "m=" << G.num_edges() << std::endl;
    }

    {
      std::ofstream graph_ic(CFG.OName + "/graph_ic.inf");
      for (uint32_t v = 0; v < G.num_nodes(); ++v)
        for (auto &e : G.neighbors(v))
          graph_ic << v << " " << e.vertex << " " << e.weight << std::endl;
    }
    return EXIT_SUCCESS;
  }

  // Dump in binary format
  auto file = std::fstream(CFG.OName, std::ios::out | std::ios::binary);
  GBackward.dump_binary(file);
  file.close();

  return EXIT_SUCCESS;
}
