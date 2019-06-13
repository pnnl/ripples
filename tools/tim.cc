//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
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

int main(int argc, char **argv) {
  im::ToolConfiguration<im::TIMConfiguration> CFG;
  CFG.ParseCmdOptions(argc, argv);

  spdlog::set_level(spdlog::level::info);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using GraphFwd = im::Graph<uint32_t, float, im::ForwardDirection<uint32_t>>;
  using GraphBwd = im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");
  GraphFwd Gf = im::loadGraph<GraphFwd>(CFG, weightGen);
  GraphBwd G = Gf.get_transpose();
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json executionLog;

  std::vector<typename GraphBwd::vertex_type> seeds;
  im::TIMExecutionRecord R;

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
    std::ofstream perf(CFG.OutputFile);
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
    std::ofstream perf(CFG.OutputFile);
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
