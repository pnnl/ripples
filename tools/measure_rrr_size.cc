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
#include <chrono>

#include "im/influence_maximization.h"
#include "im/configuration.h"
#include "im/loaders.h"

#include "boost/program_options.hpp"

#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/ostreamwrapper.h"

namespace im {

struct RRConfiguration : public Configuration {
  std::string ReportFileName;
};

RRConfiguration ParseCmdOptions(int argc, char **argv) {
  namespace po = boost::program_options;
  using seed_type = typename std::default_random_engine::result_type;

  RRConfiguration CFG;

  seed_type seed;

  po::options_description general("General Options");
  general.add_options()("help,h", "Print this help message")(
      "input-graph,i", po::value<std::string>(&CFG.IFileName)->required(),
      "The input file with the edge-list.")(
      "report-file,o", po::value<std::string>(&CFG.ReportFileName)->required(),
      "The input file with the edge-list.")(
      "seed-set-size,k", po::value<size_t>(&CFG.k), "The size of the seed set")(
      "epsilon,e", po::value<double>(&CFG.epsilon)->default_value(0.001),
      "The approximation factor.")(
       "seed,s",
       po::value<seed_type>(&seed)->default_value(0),
       "The seed of the random number generator.");

  po::variables_map VM;
  try {
    po::store(po::parse_command_line(argc, argv, general), VM);

    if (VM.count("help")) {
      std::cout << argv[0] << " [options]" << std::endl;
      std::cout << general << std::endl;
      exit(0);
    }

    po::notify(VM);

    if (seed != 0) {
      CFG.generator.seed(seed);
    }
  } catch (po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << general << std::endl;
    exit(-1);
  }

  return CFG;
}

}

int main(int argc, char **argv) {
  auto CFG = im::ParseCmdOptions(argc, argv);

  im::Graph<uint32_t> G;
  im::load(CFG.IFileName, G, im::weighted_edge_list_tsv());

  std::cout << "# Graph loaded..." << std::endl;
  std::cout << "# Number of vertices : " << G.scale() << std::endl;
  std::cout << "# Number of edges : " << G.size() << std::endl;

  auto startTheta = std::chrono::high_resolution_clock::now();

  size_t theta = thetaEstimation(G, CFG.k, CFG.epsilon);

  auto endTheta = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> exTimeTheta =
      std::chrono::duration_cast<std::chrono::duration<double>>(endTheta - startTheta);
  std::cout << "# Theta estimate as " << theta << " in " << exTimeTheta.count() << " seconds" << std::endl;

  auto startRRRGeneration = std::chrono::high_resolution_clock::now();
  auto RRRSetsSizes = generateRandomRRSet(G, theta, im::rr_size_measure_tag());
  auto endRRRGeneration = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> exTimeRRRGeneration =
      std::chrono::duration_cast<std::chrono::duration<double>>(endRRRGeneration - startRRRGeneration);
  std::cout << "# RRR sets generated in " << exTimeRRRGeneration.count() << " seconds" << std::endl;

  rapidjson::Document root;
  rapidjson::Document::AllocatorType& Allocator = root.GetAllocator();
  root.SetArray();

  for (auto size : RRRSetsSizes) {
    rapidjson::Value node(rapidjson::kObjectType);

    node.AddMember("size", uint64_t(size), Allocator);

    root.PushBack(node, Allocator);
  }

  std::ofstream reportStream(CFG.ReportFileName);
  rapidjson::OStreamWrapper OSW(reportStream);
  rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(OSW);
  root.Accept(writer);

  return 0;
}
