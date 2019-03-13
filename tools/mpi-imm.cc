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
      {"Total", R.Total},
      {"ThetaPrimeDeltas", R.ThetaPrimeDeltas},
      {"ThetaEstimation", R.ThetaEstimationTotal},
      {"ThetaEstimationGenerateRRR", R.ThetaEstimationGenerateRRR},
      {"ThetaEstimationMostInfluential", R.ThetaEstimationMostInfluential},
      {"Theta", R.Theta},
      {"GenerateRRRSets", R.GenerateRRRSets},
      {"FindMostInfluentialSet", R.FindMostInfluentialSet},
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
  im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>> G;
  if (!CFG.reload) {
    auto edgeList = im::loadEdgeList<im::Edge<uint32_t, float>>(CFG, weightGen);
    console->info("Loading Done!");
    im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>> tmpG(edgeList.begin(), edgeList.end());
    edgeList.clear();

    G = std::move(tmpG);
  } else {
    std::ifstream binaryDump(CFG.IFileName, std::ios::binary);
    im::Graph<uint32_t, float, im::BackwardDirection<uint32_t>> tmpG(binaryDump);
    G = std::move(tmpG);
  }
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
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  console->info("IMM World Size : {}", world_size);

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  console->info("IMM Rank : {}", world_rank);

  if (world_rank == 0) {
    std::ofstream perf(CFG.OutputFile);
    perf << executionLog.dump(2);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
