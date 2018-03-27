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
#include "im/influence_maximization.h"
#include "im/loaders.h"

#include "boost/program_options.hpp"

#include "omp.h"

namespace im {

Configuration CFG;

Configuration ParseCmdOptions(int argc, char **argv) {
  namespace po = boost::program_options;
  using seed_type = typename std::default_random_engine::result_type;

  seed_type seed;

  bool tim;

  po::options_description general("General Options");
  general.add_options()("help,h", "Print this help message")(
      "input-graph,i", po::value<std::string>(&CFG.IFileName)->required(),
      "The input file with the edge-list.")(
      "seed-set-size,k", po::value<size_t>(&CFG.k), "The size of the seed set")(
      "epsilon,e", po::value<double>(&CFG.epsilon)->default_value(0.001),
      "The approximation factor.")(
       "seed,s",
       po::value<seed_type>(&seed)->default_value(0),
       "The seed of the random number generator.");

  po::options_description algorithm("Algorithm Selection");
  algorithm.add_options()("tim", po::bool_switch(&tim)->default_value(false),
                          "The TIM algorithm (Tang Y. et all)");

  po::options_description all;
  all.add(general).add(algorithm);

  po::variables_map VM;
  try {
    po::store(po::parse_command_line(argc, argv, all), VM);

    if (VM.count("help")) {
      std::cout << argv[0] << " [options]" << std::endl;
      std::cout << all << std::endl;
      exit(0);
    }

    po::notify(VM);

    if (seed != 0) {
      CFG.generator.seed(seed);
    }
  } catch (po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << all << std::endl;
    exit(-1);
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

  std::cout << G << std::endl;

  auto seedSet =
      im::influence_maximization(G, CFG.k, CFG.epsilon, im::tim_tag());

  std::cout << "Seed Set : {";
  for (auto v : seedSet) std::cout << " " << v;
  std::cout << " }" << std::endl;

  return 0;
}
