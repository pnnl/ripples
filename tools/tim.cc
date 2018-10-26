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
#include "im/tim.h"
#include "im/utility.h"

#include "omp.h"

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

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
  app.add_flag("-p,--parallel", CFG.parallel,
               "Trigger the parallel implementation");

  app.add_flag("-u,--undirected", CFG.undirected,
               "The input graph is undirected");
  app.add_flag("-w,--weighted", CFG.weighted, "The input graph is weighted");
  app.add_option("-d,--diffusion-model", CFG.diffusionModel,
                 "The diffusion model to be used (LT|IC)")
      ->required();
  app.add_option("-l,--log", CFG.LogFile, "The file name of the log.");
  app.add_flag("--omp_strong_scaling", CFG.OMPStrongScaling,
               "Trigger strong scaling experiments");

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
  console->info("Loading...");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  std::vector<im::Edge<uint32_t, float>> edgeList;
  if (CFG.weighted) {
    console->info("Loading with input weights");
    edgeList = im::load<im::Edge<uint32_t, float>>(
        CFG.IFileName, CFG.undirected, weightGen, im::weighted_edge_list_tsv());
  } else {
    console->info("Loading with random weights");
    edgeList = im::load<im::Edge<uint32_t, float>>(
        CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv());
  }
  console->info("Loading Done!");

  im::Graph<uint32_t, float> G(edgeList.begin(), edgeList.end());
  edgeList.clear();
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json executionLog;

  std::vector<typename im::Graph<uint32_t, float>::vertex_type> seeds;
  im::TIMExecutionRecord R;

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  if (CFG.OMPStrongScaling) {
    size_t max_threads = 1;
    std::ofstream perf(CFG.LogFile);
#pragma omp single
    max_threads = omp_get_max_threads();

    for (size_t num_threads = max_threads; num_threads >= 1; --num_threads) {
      if (num_threads != 1) {
        omp_set_num_threads(num_threads);

        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              TIM(G, CFG.k, CFG.epsilon, generator,
                  im::independent_cascade_tag{}, im::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              TIM(G, CFG.k, CFG.epsilon, generator, im::linear_threshold_tag{},
                  im::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }

        R.NumThreads = num_threads;

        console->info("TIM parallel : {}ms, T={}/{}", R.Total.count(),
                      num_threads, max_threads);

        G.convertID(seeds.begin(), seeds.end(), seeds.begin());
        nlohmann::json experiment{
            {"Algorithm", "TIM"},
            {"DiffusionModel", CFG.diffusionModel},
            {"Epsilon", CFG.epsilon},
            {"K", CFG.k},
            {"L", 1},
            {"NumThreads", R.NumThreads},
            {"Total", R.Total.count()},
            {"KptEstimation", R.KptEstimation.count()},
            {"KptRefinement", R.KptRefinement.count()},
            {"GenerateRRRSets", R.GenerateRRRSets.count()},
            {"FindMostInfluentialSet", R.FindMostInfluentialSet.count()},
            {"Seeds", seeds}};

        executionLog.push_back(experiment);
      } else {
        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              TIM(G, CFG.k, CFG.epsilon, generator,
                  im::independent_cascade_tag{}, im::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              TIM(G, CFG.k, CFG.epsilon, generator, im::linear_threshold_tag{},
                  im::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }
        console->info("TIM squential : {}ms, T={}/{}", R.Total.count(),
                      num_threads, max_threads);

        R.NumThreads = num_threads;

        G.convertID(seeds.begin(), seeds.end(), seeds.begin());
        nlohmann::json experiment{
            {"Algorithm", "TIM"},
            {"DiffusionModel", CFG.diffusionModel},
            {"Epsilon", CFG.epsilon},
            {"K", CFG.k},
            {"L", 1},
            {"NumThreads", R.NumThreads},
            {"Total", R.Total.count()},
            {"KptEstimation", R.KptEstimation.count()},
            {"KptRefinement", R.KptRefinement.count()},
            {"GenerateRRRSets", R.GenerateRRRSets.count()},
            {"FindMostInfluentialSet", R.FindMostInfluentialSet.count()},
            {"Seeds", seeds}};

        executionLog.push_back(experiment);
      }

      perf.seekp(0);
      perf << executionLog.dump(2);
    }
  } else if (CFG.parallel) {
    std::ofstream perf(CFG.LogFile);
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          TIM(G, CFG.k, CFG.epsilon, generator, im::independent_cascade_tag{},
              im::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          TIM(G, CFG.k, CFG.epsilon, generator, im::linear_threshold_tag{},
              im::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("TIM parallel : {}ms", R.Total.count());

    size_t max_num_threads;
#pragma omp single
    max_num_threads = omp_get_max_threads();

    R.NumThreads = max_num_threads;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    nlohmann::json experiment{
        {"Algorithm", "TIM"},
        {"DiffusionModel", CFG.diffusionModel},
        {"Epsilon", CFG.epsilon},
        {"K", CFG.k},
        {"L", 1},
        {"NumThreads", R.NumThreads},
        {"Total", R.Total.count()},
        {"KptEstimation", R.KptEstimation.count()},
        {"KptRefinement", R.KptRefinement.count()},
        {"GenerateRRRSets", R.GenerateRRRSets.count()},
        {"FindMostInfluentialSet", R.FindMostInfluentialSet.count()},
        {"Seeds", seeds}};

    executionLog.push_back(experiment);

    perf << executionLog.dump(2);
  } else {
    std::ofstream perf(CFG.LogFile);
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          TIM(G, CFG.k, CFG.epsilon, generator, im::independent_cascade_tag{},
              im::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          TIM(G, CFG.k, CFG.epsilon, generator, im::linear_threshold_tag{},
              im::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }

    console->info("TIM squential : {}ms", R.Total.count());

    R.NumThreads = 1;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    nlohmann::json experiment{
        {"Algorithm", "TIM"},
        {"DiffusionModel", CFG.diffusionModel},
        {"Epsilon", CFG.epsilon},
        {"K", CFG.k},
        {"L", 1},
        {"NumThreads", R.NumThreads},
        {"Total", R.Total.count()},
        {"KptEstimation", R.KptEstimation.count()},
        {"KptRefinement", R.KptRefinement.count()},
        {"GenerateRRRSets", R.GenerateRRRSets.count()},
        {"FindMostInfluentialSet", R.FindMostInfluentialSet.count()},
        {"Seeds", seeds}};

    executionLog.push_back(experiment);

    perf << executionLog.dump(2);
  }

  return EXIT_SUCCESS;
}
