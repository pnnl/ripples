//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "im/configuration.h"
#include "im/graph.h"
#include "im/loaders.h"
#include "im/imm.h"
#include "im/utility.h"
#include "im/diffusion_simulation.h"

#include "omp.h"

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"


namespace im {

Configuration ParseCmdOptions(int argc, char **argv) {
  Configuration CFG;

  bool tim = true;

  CLI::App app{"Yet another tool to experiment with INF-MAX"};
  app.add_option("-i,--input-graph", CFG.IFileName,
                 "The input file with the edge-list.")
      ->required();
  app.add_option("-k,--seed-set-size", CFG.k, "The size of the seed set.")
      ->required();
  app.add_option("-e,--epsilon", CFG.epsilon, "The size of the seed set.")
      ->required();
  app.add_flag("-p,--parallel", CFG.parallel, "Trigger the parallel implementation");
  app.add_flag("-u,--undirected", CFG.undirected, "The input graph is undirected");
  app.add_option("-d,--diffusion-model", CFG.diffusionModel,
                 "The diffusion model to be used (LT|IC)")
      ->required();
  app.add_option("-l,--log", CFG.LogFile, "The file name of the log.");
  app.add_flag("--omp_strong_scaling", CFG.OMPStrongScaling, "Trigger strong scaling experiments");
  
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return CFG;
}

}  // namespace im

int main(int argc, char **argv) {
  im::Configuration CFG = im::ParseCmdOptions(argc, argv);

  spdlog::set_level(spdlog::level::info);

  auto console = spdlog::stdout_color_st("console");
  auto perf = spdlog::basic_logger_st("perf", CFG.LogFile);

  perf->set_pattern("%v");

  console->info("Loading...");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  auto edgeList =
      im::load<im::Edge<uint32_t, float>>(CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv());
  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json executionLog;

  std::vector<typename im::Graph<uint32_t, float>::vertex_type> seeds;
  im::IMMExecutionRecord R;

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  if (CFG.OMPStrongScaling) {
    size_t max_threads = 1;
#pragma omp single
    max_threads = omp_get_max_threads();

    for (size_t num_threads = 1; num_threads <= max_threads; ++num_threads) {
      if (num_threads != 1) {
        omp_set_num_threads(num_threads);

        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                                   im::independent_cascade_tag{},
                                   im::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                                   im::linear_threshold_tag{},
                                   im::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }

        R.NumThreads = num_threads;

        console->info("IMM parallel : {}ms, T={}/{}", R.Total.count(), num_threads, max_threads);

        nlohmann::json experiment{
          { "Algorithm", "IMM" },
          { "DiffusionModel", CFG.diffusionModel },
          { "Epsilon", CFG.epsilon },
          { "K", CFG.k },
          { "L", 1 },
          { "NumThreads", R.NumThreads },
          { "Total", R.Total.count() },
          { "ThetaEstimation", R.ThetaEstimation.count() },
          { "GenerateRRRSets", R.GenerateRRRSets.count() },
          { "FindMostInfluentialSet", R.FindMostInfluentialSet.count() },
          { "Seeds", seeds }
        };

        executionLog.push_back(experiment);
      } else {
        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                                   im::independent_cascade_tag{},
                                   im::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                                   im::linear_threshold_tag{},
                                   im::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }
        console->info("IMM squential : {}ms, T={}/{}", R.Total.count(), num_threads, max_threads);

        R.NumThreads = num_threads;

        nlohmann::json experiment{
          { "Algorithm", "IMM" },
          { "Epsilon", CFG.epsilon },
          { "DiffusionModel", CFG.diffusionModel },
          { "K", CFG.k },
          { "L", 1 },
          { "NumThreads", R.NumThreads },
          { "Total", R.Total.count() },
          { "ThetaEstimation", R.ThetaEstimation.count() },
          { "GenerateRRRSets", R.GenerateRRRSets.count() },
          { "FindMostInfluentialSet", R.FindMostInfluentialSet.count() },
          { "Seeds", seeds }
        };

        executionLog.push_back(experiment);
      }
    }

    perf->info("{}", executionLog.dump(2));
  } else if (CFG.parallel) {
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                               im::independent_cascade_tag{},
                               im::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                               im::linear_threshold_tag{},
                               im::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("IMM parallel : {}ms", R.Total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    nlohmann::json experiment{
      { "Algorithm", "IMM" },
      { "DiffusionModel", CFG.diffusionModel },
      { "Epsilon", CFG.epsilon },
      { "K", CFG.k },
      { "L", 1 },
      { "NumThreads", R.NumThreads },
      { "Total", R.Total.count() },
      { "ThetaEstimation", R.ThetaEstimation.count() },
      { "GenerateRRRSets", R.GenerateRRRSets.count() },
      { "FindMostInfluentialSet", R.FindMostInfluentialSet.count() },
      { "Seeds", seeds }
    };

    executionLog.push_back(experiment);

    perf->info("{}", executionLog.dump(2));
  } else {
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                               im::independent_cascade_tag{},
                               im::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) = IMM(G, CFG.k, CFG.epsilon, 1, generator,
                               im::linear_threshold_tag{},
                               im::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("IMM squential : {}ms", R.Total.count());

    R.NumThreads = 1;

    nlohmann::json experiment{
      { "Algorithm", "IMM" },
      { "DiffusionModel", CFG.diffusionModel },
      { "Epsilon", CFG.epsilon },
      { "K", CFG.k },
      { "L", 1 },
      { "NumThreads", R.NumThreads },
      { "Total", R.Total.count() },
      { "ThetaEstimation", R.ThetaEstimation.count() },
      { "GenerateRRRSets", R.GenerateRRRSets.count() },
      { "FindMostInfluentialSet", R.FindMostInfluentialSet.count() },
      { "Seeds", seeds }
    };

    executionLog.push_back(experiment);
    perf->info("{}", executionLog.dump(2));
  }

  return EXIT_SUCCESS;
}
