//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2017 Pacific Northwest National Laboratory
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "im/configuration.h"
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

  im::Graph<uint32_t> G;

  std::cout << "Loading..." << std::endl;
  im::load(CFG.IFileName, G, im::weighted_edge_list_tsv());

  std::cout << "Size: " << G.size() << std::endl;
  std::cout << "Scale: " << G.scale() << std::endl;

  std::unordered_set<typename im::Graph<uint32_t>::vertex_type> seedSet;

  switch (CFG.algo) {
    case im::Algorithm::TIM:
      seedSet = im::influence_maximization(G, CFG.k, CFG.epsilon, im::tim_tag());
      break;
    default:
      throw std::string("Unknown algorithm requested");
  }

  std::cout << "Seed Set : {";
  for (auto v : seedSet) std::cout << " " << v;
  std::cout << " }" << std::endl;

  return EXIT_SUCCESS;
}
