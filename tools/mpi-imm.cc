//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include "mpi.h"
#include "omp.h"

#include <iostream>

#include "im/configuration.h"
#include "im/diffusion_simulation.h"
#include "im/graph.h"
#include "im/loaders.h"
#include "im/mpi/imm.h"
#include "im/utility.h"

#include "CLI11/CLI11.hpp"
#include "nlohmann/json.hpp"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"


namespace im {

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<IMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  nlohmann::json experiment{
      {"Algorithm", "IMM"},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"K", CFG.k},
      {"L", 1},
      {"Rank", world_rank},
      {"WorldSize", world_size},
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


int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);

  im::ToolConfiguration<im::IMMConfiguration> CFG;
  CFG.ParseCmdOptions(argc, argv);

  spdlog::set_level(spdlog::level::info);

  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  std::vector<im::Edge<uint32_t, float>> edgeList;
  if (CFG.weighted) {
    console->info("Loading with input weights");
    if (CFG.diffusionModel == "IC") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen,
          im::weighted_edge_list_tsv{}, im::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen,
          im::weighted_edge_list_tsv{}, im::linear_threshold_tag{});
    }
  } else {
    console->info("Loading with random weights");
    if (CFG.diffusionModel == "IC") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv{},
          im::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = im::load<im::Edge<uint32_t, float>>(
          CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv{},
          im::linear_threshold_tag{});
    }
  }
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

  if (CFG.diffusionModel == "IC") {
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(seeds, R) =
        IMM(G, CFG.k, CFG.epsilon, 1.0, generator,
            im::independent_cascade_tag{}, im::mpi_omp_parallel_tag{});
    auto end = std::chrono::high_resolution_clock::now();
    R.Total = end - start;
  } else if (CFG.diffusionModel == "LT") {
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(seeds, R) =
        IMM(G, CFG.k, CFG.epsilon, 1, generator, im::linear_threshold_tag{},
            im::mpi_omp_parallel_tag{});
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

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    std::ofstream perf(CFG.LogFile);
    perf << executionLog.dump(2);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
