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
  auto edgeList = im::load<im::Edge<uint32_t, double>>(CFG.IFileName, im::edge_list_tsv());
  std::cout << "Loading Done!" << std::endl;

  im::Graph<uint32_t, double> G(edgeList.begin(), edgeList.end());
  std::cout << "Number of Nodes : " << G.num_nodes() << std::endl;
  std::cout << "Number of Edges : " << G.num_edges() << std::endl;

  for (size_t i = 0; i < G.num_nodes(); ++i) {
    std::cout << i
              << " " <<  G.out_degree(i) << " :";
    for (auto n : G.out_neighbors(i))
      std::cout << " " <<  n;
    std::cout << std::endl;
  }

  // for (auto v : G) {
  //   std::cout << v << "{ ";
  //   for (auto & e : G[v]) {
  //     std::cout << "(" << e.first << ", " << e.second << "), ";
  //   }
  //   std::cout << "}" << std::endl;
  // }

  // size_t KPT = KptEstimation(G, CFG.k, CFG.epsilon, 0.5);

  // std::cout << "#### KPT" << KPT << std::endl;

  // std::cout << "Size: " << G.size() << std::endl;
  // std::cout << "Scale: " << G.scale() << std::endl;

  // std::unordered_set<typename im::Graph<uint32_t>::vertex_type> seedSet;

  // switch (CFG.algo) {
  //   case im::Algorithm::TIM:
  //     seedSet = im::TIM(G, CFG.k, CFG.epsilon);
  //     break;
  //   default:
  //     throw std::string("Unknown algorithm requested");
  // }

  // std::cout << "Seed Set : {";
  // for (auto v : seedSet) std::cout << " " << v;
  // std::cout << " }" << std::endl;

  return EXIT_SUCCESS;
}
