//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "im/configuration.h"
#include "im/diffusion_simulation.h"
#include "im/graph.h"
#include "im/imm.h"
#include "im/loaders.h"
#include "im/utility.h"

#include "omp.h"

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace im {

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<IMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  nlohmann::json experiment{
      {"Algorithm", "IMM"},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"K", CFG.k},
      {"L", 1},
      {"NumThreads", R.NumThreads},
      {"Total", R.Total.count()},
      {"ThetaEstimation", R.ThetaEstimation.count()},
      {"Theta", R.Theta},
      {"GenerateRRRSets", R.GenerateRRRSets.count()},
      {"FindMostInfluentialSet", R.FindMostInfluentialSet.count()},
      {"Seeds", seeds}};
  return experiment;
}

}  // namespace im

int main(int argc, char **argv) {
  im::ToolConfiguration<im::IMMConfiguration> CFG;
  CFG.ParseCmdOptions(argc, argv);

  spdlog::set_level(spdlog::level::info);

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  auto edgeList = im::loadEdgeList<im::Edge<uint32_t, float>>(CFG, weightGen);
  console->info("Loading Done!");

  im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>> G(edgeList.begin(), edgeList.end());
  edgeList.clear();
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
    std::ofstream perf(CFG.OutputFile);
#pragma omp single
    max_threads = omp_get_max_threads();

    for (size_t num_threads = max_threads; num_threads >= 1; --num_threads) {
      if (num_threads != 1) {
        omp_set_num_threads(num_threads);

        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              IMM(G, CFG.k, CFG.epsilon, 1, generator,
                  im::independent_cascade_tag{}, im::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              IMM(G, CFG.k, CFG.epsilon, 1, generator,
                  im::linear_threshold_tag{}, im::omp_parallel_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }

        R.NumThreads = num_threads;

        console->info("IMM parallel : {}ms, T={}/{}", R.Total.count(),
                      num_threads, max_threads);

        G.convertID(seeds.begin(), seeds.end(), seeds.begin());
        auto experiment = GetExperimentRecord(CFG, R, seeds);
        executionLog.push_back(experiment);
      } else {
        if (CFG.diffusionModel == "IC") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              IMM(G, CFG.k, CFG.epsilon, 1, generator,
                  im::independent_cascade_tag{}, im::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        } else if (CFG.diffusionModel == "LT") {
          auto start = std::chrono::high_resolution_clock::now();
          std::tie(seeds, R) =
              IMM(G, CFG.k, CFG.epsilon, 1, generator,
                  im::linear_threshold_tag{}, im::sequential_tag{});
          auto end = std::chrono::high_resolution_clock::now();
          R.Total = end - start;
        }
        console->info("IMM squential : {}ms, T={}/{}", R.Total.count(),
                      num_threads, max_threads);

        R.NumThreads = num_threads;

        G.convertID(seeds.begin(), seeds.end(), seeds.begin());
        auto experiment = GetExperimentRecord(CFG, R, seeds);
        executionLog.push_back(experiment);
      }
      perf.seekp(0);
      perf << executionLog.dump(2);
    }
  } else if (CFG.parallel) {
    std::ofstream perf(CFG.OutputFile);
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          IMM(G, CFG.k, CFG.epsilon, 1, generator,
              im::independent_cascade_tag{}, im::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          IMM(G, CFG.k, CFG.epsilon, 1, generator, im::linear_threshold_tag{},
              im::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("IMM parallel : {}ms", R.Total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);

    perf << executionLog.dump(2);
  } else {
    std::ofstream perf(CFG.OutputFile);
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          IMM(G, CFG.k, CFG.epsilon, 1, generator,
              im::independent_cascade_tag{}, im::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      std::tie(seeds, R) =
          IMM(G, CFG.k, CFG.epsilon, 1, generator, im::linear_threshold_tag{},
              im::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("IMM squential : {}ms", R.Total.count());

    R.NumThreads = 1;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);
    perf << executionLog.dump(2);
  }

  return EXIT_SUCCESS;
}
