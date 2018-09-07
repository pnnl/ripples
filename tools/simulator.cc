//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <string>

#include "CLI11/CLI11.hpp"
#include "spdlog/spdlog.h"

#include "im/graph.h"
#include "im/loaders.h"

namespace im {

struct SimulatorConfiguration {
  std::string IFileName;
  std::string OFileName;
  std::string EFileName;
  std::string DiffusionModel;
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
  return EXIT_SUCCESS;
}
