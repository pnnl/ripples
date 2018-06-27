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
#include "spdlog/spdlog.h"

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

  spdlog::set_level(spdlog::level::info);

  auto console = spdlog::stdout_color_st("console");
  auto perf = spdlog::basic_logger_mt("perf", "perf.log");
  spdlog::set_async_mode(8192);
  perf->set_level(spdlog::level::trace);

  console->info("Loading...");
  auto edgeList = im::load<im::Edge<uint32_t, float>>(CFG.IFileName, im::edge_list_tsv());
  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());
  {
    auto start = std::chrono::high_resolution_clock::now();
    auto kpt = ThetaEstimation(G, CFG.k, CFG.epsilon, im::sequential_tag());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exTime = end - start;
    console->info("kpt : {} {}ms", kpt, exTime.count());
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    auto kpt_parallel = ThetaEstimation(G, CFG.k, CFG.epsilon, im::omp_parallel_tag());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exTime = end - start;
    console->info("kpt : {} {}ms", kpt_parallel, exTime.count());
  }

  return EXIT_SUCCESS;
}
