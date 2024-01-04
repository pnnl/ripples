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
#define ENABLE_METALL_RRRSETS

#include <iostream>
#include <sstream>
#include <string>

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/loaders.h"
#include "ripples/utility.h"
#include "ripples/imm_configuration.h"
#include "ripples/imm_interface.h"
#include "ripples/imm.h"

#include "nlohmann/json.hpp"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "omp.h"

namespace ripples {

auto GetWalkIterationRecord(
    const typename IMMExecutionRecord::walk_iteration_prof &iter) {
  nlohmann::json res{{"NumSets", iter.NumSets}, {"Total", iter.Total.count()}};
  for (size_t wi = 0; wi < iter.CPUWalks.size(); ++wi) {
    std::stringstream wname;
    wname << "CPU Worker " << wi;
    res[wname.str()] = nlohmann::json{{"NumSets", iter.CPUWalks[wi].NumSets},
                                      {"Total", iter.CPUWalks[wi].Total.count()}};
  }
  for (size_t wi = 0; wi < iter.GPUWalks.size(); ++wi) {
    std::stringstream wname;
    wname << "GPU Worker " << wi;
    res[wname.str()] = nlohmann::json{{"NumSets", iter.GPUWalks[wi].NumSets},
                                      {"Total", iter.GPUWalks[wi].Total.count()},
                                      {"Kernel", iter.GPUWalks[wi].Kernel.count()},
                                      {"D2H", iter.GPUWalks[wi].D2H.count()},
                                      {"Post", iter.GPUWalks[wi].Post.count()}};
  }
  return res;
}

template <typename TimeTy>
std::vector<double> ConvertToCounts(const TimeTy &times){
  std::vector<double> counts;
  counts.reserve(times.size());
  for(auto time : times){
    counts.push_back(time.count());
  }
  return counts;
}

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<IMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  nlohmann::json experiment{
      {"Algorithm", "IMM"},
      {"Input", CFG.IFileName},
      {"Output", CFG.OutputFile},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"K", CFG.k},
      {"L", 1},
      {"NumThreads", R.NumThreads},
      {"NumWalkWorkers", CFG.streaming_workers},
      {"NumCPUTeams", CFG.streaming_cpu_teams},
      {"NumGPUWalkWorkers", CFG.streaming_gpu_workers},
      {"PauseThreshold", CFG.pause_threshold},
      {"Total", R.Total.count()},
      {"ThetaPrimeDeltas", R.ThetaPrimeDeltas},
      {"ThetaEstimation", R.ThetaEstimationTotal.count()},
      {"ThetaEstimationGenerateRRR", ConvertToCounts(R.ThetaEstimationGenerateRRR)},
      {"ThetaEstimationMostInfluential", ConvertToCounts(R.ThetaEstimationMostInfluential)},
      {"Theta", R.Theta},
      {"Counting", ConvertToCounts(R.Counting)},
      {"Pivoting", ConvertToCounts(R.Pivoting)},
      {"Microbenchmarking", R.Microbenchmarking.count()},
      {"CPUBatchSize", R.CPUBatchSize},
      {"GPUBatchSize", R.GPUBatchSize},
      {"RRRSetSizeBytes", R.RRRSetSize},
      {"GenerateRRRSets", R.GenerateRRRSets.count()},
      {"FindMostInfluentialSet", R.FindMostInfluentialSet.count()},
      {"Seeds", seeds}};
  for (auto &ri : R.WalkIterations) {
    experiment["Iterations"].push_back(GetWalkIterationRecord(ri));
  }
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

int main(int argc, char **argv) {
  auto console = spdlog::stdout_color_st("console");

  // process command line
  ripples::parse_command_line(argc, argv);
  auto CFG = ripples::configuration();
  if (CFG.parallel) {
    if (ripples::streaming_command_line(
            CFG.worker_to_gpu, CFG.streaming_workers, CFG.streaming_cpu_teams, CFG.streaming_gpu_workers,
            CFG.gpu_mapping_string) != 0) {
      console->error("invalid command line");
      return -1;
    }
  }

  spdlog::set_level(spdlog::level::info);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using dest_type = ripples::WeightedDestination<uint32_t, float>;
#if defined ENABLE_METALL
  using GraphFwd =
      ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>,
                     metall::manager::allocator_type<char>>;
  using GraphBwd =
      ripples::Graph<uint32_t, dest_type, ripples::BackwardDirection<uint32_t>,
                     metall::manager::allocator_type<char>>;
#else
  using GraphFwd =
      ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>>;
  using GraphBwd =
      ripples::Graph<uint32_t, dest_type, ripples::BackwardDirection<uint32_t>>;
#endif
  console->info("Loading...");
  auto loading_start = std::chrono::high_resolution_clock::now();
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
  GraphFwd Gf = ripples::loadGraph<GraphFwd>(CFG, weightGen);
  GraphBwd G = Gf.get_transpose();
#endif
  auto loading_end = std::chrono::high_resolution_clock::now();
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());
  const auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                             loading_end - loading_start)
                             .count();
  console->info("Loading took {}ms", load_time);

  nlohmann::json executionLog;

  std::vector<typename ripples::GraphBwd::vertex_type> seeds;
  ripples::IMMExecutionRecord R;

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  std::ofstream perf(CFG.OutputFile);

  #ifdef PROFILE_OVERHEAD
  output_file_name = CFG.OutputFile;
  #endif // PROFILE_OVERHEAD

  #ifdef UTILIZATION_PROFILE
  output_file_name = CFG.OutputFile;
  #endif // UTILIZATION_PROFILE

  if (CFG.parallel) {
    auto workers = CFG.streaming_workers;
    auto cpu_teams = CFG.streaming_cpu_teams;
    auto gpu_workers = CFG.streaming_gpu_workers;
    decltype(R.Total) real_total;
    if (CFG.diffusionModel == "IC") {
      ripples::ICStreamingGenerator se(G, generator, workers - gpu_workers, cpu_teams, gpu_workers,
             CFG.gpu_batch_size, CFG.cpu_batch_size, CFG.worker_to_gpu, CFG.pause_threshold);
      R.GPUBatchSize = CFG.gpu_batch_size;
      if (CFG.cpu_batch_size) {
        R.CPUBatchSize = CFG.cpu_batch_size;
      }
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
      else {
        if (se.isGpuEnabled() && cpu_teams) {
          se.benchmark(2, 4, R);
        }
      }
#endif

      auto start = std::chrono::high_resolution_clock::now();
      if(CFG.num_rr_sets){
        // Override, just generate one set of RR sets
        ssize_t thetaPrime = CFG.num_rr_sets;
        size_t delta = thetaPrime;
        R.ThetaPrimeDeltas.push_back(delta);
        R.Theta = CFG.num_rr_sets;

        using vertex_type = typename ripples::GraphBwd::vertex_type;
#if defined ENABLE_METALL_RRRSETS
        ripples::RRRsetAllocator<vertex_type> allocator =  metall_manager_instance(CFG.rr_dir).get_allocator();
#else
        ripples::RRRsetAllocator<vertex_type> allocator;
#endif
        std::vector<ripples::RRRset<ripples::GraphBwd>> RR;
        auto timeRRRSets = ripples::measure<>::exec_time([&]() {
          RR.insert(RR.end(), delta, ripples::RRRset<ripples::GraphBwd>(allocator));

          auto begin = RR.end() - delta;

          GenerateRRRSets(G, se, begin, RR.end(), R,
                          ripples::independent_cascade_tag{},
                          ripples::omp_parallel_tag{});
        });
        R.ThetaEstimationGenerateRRR.push_back(timeRRRSets);
        R.ThetaEstimationMostInfluential.push_back(timeRRRSets - timeRRRSets);
        seeds = std::vector<typename ripples::GraphBwd::vertex_type>(CFG.k, 1);
      }
      else{
        seeds = IMM(G, CFG, 1, se, R, ripples::independent_cascade_tag{},
                  ripples::omp_parallel_tag{});
      }
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start - R.Total;
      real_total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      ripples::LTStreamingGenerator se(G, generator, workers - gpu_workers, cpu_teams, gpu_workers,
             CFG.gpu_batch_size, CFG.cpu_batch_size, CFG.worker_to_gpu);

      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM(G, CFG, 1, se, R, ripples::linear_threshold_tag{},
                  ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start - R.Total;
      real_total = end - start;
    }

    console->info("IMM Parallel : {}ms", R.Total.count());
    console->info("IMM Parallel Real Total : {}ms", real_total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);
    perf << executionLog.dump(2);
  } else {
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM(G, CFG, 1.0, generator, R, ripples::independent_cascade_tag{},
                  ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM(G, CFG, 1.0, generator, R, ripples::linear_threshold_tag{},
                  ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("IMM squential : {}ms", R.Total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);
    perf << executionLog.dump(2);
  }

  return EXIT_SUCCESS;
}
