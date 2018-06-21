//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "im/configuration.h"
#include "im/graph.h"
#include "im/tim.h"
#include "im/loaders.h"

#include "CLI11/CLI11.hpp"

#include "omp.h"

namespace im {

Configuration ParseCmdOptions(int argc, char **argv) {
  Configuration CFG;

  bool tim = true;

  CLI::App app{"Yet another tool to experiment with INF-MAX"};
  app.add_option("-i,--input-graph", CFG.IFileName,
                 "The input file with the edge-list.")
      ->required();
  app.add_option("-k,--seed-set-size", CFG.k,
                 "The size of the seed set.")
      ->required();
  app.add_option("-e,--epsilon", CFG.epsilon,
                 "The size of the seed set.")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError & e) {
    exit(app.exit(e));
  }

  return CFG;
}

}  // namespace im

int main(int argc, char **argv) {
  im::Configuration CFG = im::ParseCmdOptions(argc, argv);

  std::cout << "Loading..." << std::endl;
  auto edgeList = im::load<im::Edge<uint32_t, float>>(CFG.IFileName, im::edge_list_tsv());
  std::cout << "Loading Done!" << std::endl;

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  std::cout << "Number of Nodes : " << G.num_nodes() << std::endl;
  std::cout << "Number of Edges : " << G.num_edges() << std::endl;

  return EXIT_SUCCESS;
}
