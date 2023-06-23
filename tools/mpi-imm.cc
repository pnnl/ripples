//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "mpi.h"
#include "omp.h"

#include <iostream>

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/loaders.h"
#include "ripples/mpi/imm.h"
#include "ripples/utility.h"

#include "CLI/CLI.hpp"
#include "nlohmann/json.hpp"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace ripples {

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<IMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  nlohmann::json experiment{
      {"Algorithm", "MPI-IMM"},
      {"Input", CFG.IFileName},
      {"Output", CFG.OutputFile},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"K", CFG.k},
      {"L", 1},
      {"Rank", world_rank},
      {"WorldSize", world_size},
      {"NumThreads", R.NumThreads},
      {"NumWalkWorkers", CFG.streaming_workers},
      {"NumGPUWalkWorkers", CFG.streaming_gpu_workers},
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

ToolConfiguration<ripples::IMMConfiguration> CFG;

void parse_command_line(int argc, char **argv) {
  CFG.ParseCmdOptions(argc, argv);
#pragma omp single
  CFG.streaming_workers = omp_get_max_threads();

  if (CFG.seed_select_max_workers == 0)
    CFG.seed_select_max_workers = CFG.streaming_workers;
  if (CFG.seed_select_max_gpu_workers == std::numeric_limits<size_t>::max())
    CFG.seed_select_max_gpu_workers = CFG.streaming_gpu_workers;
}

ToolConfiguration<ripples::IMMConfiguration> configuration() { return CFG; }

}  // namespace ripples

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  spdlog::set_level(spdlog::level::info);
  auto console = spdlog::stdout_color_st("console");

  // process command line
  ripples::parse_command_line(argc, argv);
  auto CFG = ripples::configuration();
  if (CFG.parallel) {
    if (ripples::streaming_command_line(
            CFG.worker_to_gpu, CFG.streaming_workers, CFG.streaming_gpu_workers,
            CFG.gpu_mapping_string) != 0) {
      console->error("invalid command line");
      return -1;
    }
  }

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using edge_type = ripples::WeightedDestination<uint32_t, float>;
  #if defined ENABLE_METALL
  using GraphFwd =
      ripples::Graph<uint32_t, edge_type, ripples::ForwardDirection<uint32_t>, metall::manager::allocator_type<char>>;
  using GraphBwd =
      ripples::Graph<uint32_t, edge_type, ripples::BackwardDirection<uint32_t>, metall::manager::allocator_type<char>>;
  #else
  using GraphFwd =
      ripples::Graph<uint32_t, edge_type, ripples::ForwardDirection<uint32_t>>;
  using GraphBwd =
      ripples::Graph<uint32_t, edge_type, ripples::BackwardDirection<uint32_t>>;
  #endif
  console->info("Loading...");
  #if defined ENABLE_METALL
  bool exists = metall::manager::consistent(CFG.metall_dir.c_str());
  metall::manager manager =
      (exists ? metall::manager(metall::open_only, CFG.metall_dir.c_str())
              : metall::manager(metall::create_only, CFG.metall_dir.c_str()));
  GraphBwd *Gr;
  if (exists) {
    console->info("Previously existing graph exists! Loading...");
    Gr = manager.find<GraphBwd>("graph").first;
  } else {
    console->info("Creating new metall directory...");
    GraphFwd Gf =
        ripples::loadGraph<GraphFwd>(CFG, weightGen, manager.get_allocator());
    Gr = manager.construct<GraphBwd>("graph")(Gf.get_transpose());
  }
  GraphBwd &G(*Gr);
#else
  std::allocator<char> = GraphAllocator;
  GraphFwd Gf = ripples::loadGraph<GraphFwd>(CFG, weightGen, GraphAllocator);
  GraphBwd G = Gf.get_transpose();
#endif
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  nlohmann::json executionLog;

  std::vector<typename GraphBwd::vertex_type> seeds;
  ripples::IMMExecutionRecord R;

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);
  ripples::mpi::split_generator(generator);

  auto workers = CFG.streaming_workers;
  auto gpu_workers = CFG.streaming_gpu_workers;
  if (CFG.diffusionModel == "IC") {
    ripples::StreamingRRRGenerator<
        decltype(G), decltype(generator),
        typename ripples::RRRsets<decltype(G)>::iterator,
        ripples::independent_cascade_tag>
        se(G, generator, R, workers - gpu_workers, gpu_workers,
           CFG.worker_to_gpu);
    auto start = std::chrono::high_resolution_clock::now();
    seeds = ripples::mpi::IMM(
        G, CFG, 1.0, se, R, ripples::independent_cascade_tag{},
        ripples::mpi::MPI_Plus_X<ripples::mpi_omp_parallel_tag>{});
    auto end = std::chrono::high_resolution_clock::now();
    R.Total = end - start;
  } else if (CFG.diffusionModel == "LT") {
    ripples::StreamingRRRGenerator<
        decltype(G), decltype(generator),
        typename ripples::RRRsets<decltype(G)>::iterator,
        ripples::linear_threshold_tag>
        se(G, generator, R, workers - gpu_workers, gpu_workers,
           CFG.worker_to_gpu);
    auto start = std::chrono::high_resolution_clock::now();
    seeds = ripples::mpi::IMM(
        G, CFG, 1.0, se, R, ripples::linear_threshold_tag{},
        ripples::mpi::MPI_Plus_X<ripples::mpi_omp_parallel_tag>{});
    auto end = std::chrono::high_resolution_clock::now();
    R.Total = end - start;
  }
  console->info("IMM MPI+OpenMP+CUDA : {}ms", R.Total.count());

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
