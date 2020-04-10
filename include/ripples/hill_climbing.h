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

#ifndef RIPPLES_HILL_CLIMBING_H
#define RIPPLES_HILL_CLIMBING_H

#include <algorithm>
#include <chrono>
#include <queue>
#include <type_traits>
#include <vector>

#include "nlohmann/json.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/hill_climbing_engine.h"

#include "omp.h"

namespace ripples {

//! The Hill Climbing Algorithm configuration descriptor.
struct HillClimbingConfiguration : public AlgorithmConfiguration {
  size_t samples{10000};
  size_t streaming_workers{0};
  size_t streaming_gpu_workers{0};

  //! \brief Add command line options to configure the Hill Climbing Algorithm.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    AlgorithmConfiguration::addCmdOptions(app);
    app.add_option(
        "--samples", samples,
        "The number of samples used in the Hill Climbing Algorithm.");
    app.add_option(
           "--streaming-gpu-workers", streaming_gpu_workers,
           "The number of GPU workers for the CPU+GPU streaming engine.")
        ->group("Streaming-Engine Options");
  }
};

//! The Hill Climbing Execution Record.
struct HillClimbingExecutionRecord {
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  using ex_time_ns = std::chrono::nanoseconds;

  //! Number of threads used during the execution.
  size_t NumThreads;
  //! Number of threads used as GPU worker.
  size_t NumGpuWorkers;
  //! Sampling time.
  ex_time_ms Sampling;
  //! SamplingTask ex time
  std::vector<std::vector<ex_time_ms>> SamplingTasks;
  std::vector<std::vector<ex_time_ms>> SeedSelectionTasks;
  //! Seed Selection Tasks
  std::vector<std::vector<std::vector<ex_time_ms>>> BuildFrontiersTasks;
  std::vector<std::vector<std::vector<ex_time_ms>>> BuildCountersTasks;
  //! Network Communication
  std::vector<ex_time_ms> NetworkReductions;
  //! Seed Selection time.
  ex_time_ms SeedSelection;
  //! Total execution time.
  ex_time_ms Total;
};

template <typename GraphTy, typename GeneratorTy, typename diff_model_tag,
          typename ConfTy>
auto SampleFrom(GraphTy &G, ConfTy &CFG, GeneratorTy &gen,
                HillClimbingExecutionRecord &record,
                diff_model_tag &&diff_model) {
  using vertex_type = typename GraphTy::vertex_type;
  using edge_mask = Bitmask<int>;
  std::vector<edge_mask> samples(CFG.samples,
                                 edge_mask(G.num_edges()));
  auto start = std::chrono::high_resolution_clock::now();

  using iterator_type = typename std::vector<edge_mask>::iterator;
  SamplingEngine<GraphTy, iterator_type, GeneratorTy, diff_model_tag> SE(
      G, gen, CFG.streaming_workers, CFG.streaming_gpu_workers);
  SE.exec(samples.begin(), samples.end(), record.SamplingTasks);
  auto end = std::chrono::high_resolution_clock::now();
  record.Sampling = end - start;
  return samples;
}

template <typename GraphTy, typename GraphMaskItrTy, typename ConfigTy>
auto SeedSelection(GraphTy &G, GraphMaskItrTy B, GraphMaskItrTy E,
                   ConfigTy &CFG, HillClimbingExecutionRecord &record) {
  SeedSelectionEngine<GraphTy, GraphMaskItrTy> countingEngine(
      G, CFG.streaming_workers, CFG.streaming_gpu_workers);
  auto start = std::chrono::high_resolution_clock::now();
  auto S = countingEngine.exec(B, E, CFG.k, record.SeedSelectionTasks);
  auto end = std::chrono::high_resolution_clock::now();
  record.SeedSelection = end - start;

  return S;
}

template <typename GraphTy, typename GeneratorTy, typename diff_model_tag,
          typename ConfTy>
auto HillClimbing(GraphTy &G, ConfTy &CFG, GeneratorTy &gen,
                  HillClimbingExecutionRecord &record,
                  diff_model_tag &&model_tag) {
  auto sampled_graphs =
      SampleFrom(G, CFG, gen, record, std::forward<diff_model_tag>(model_tag));

  auto S = SeedSelection(G, sampled_graphs.begin(), sampled_graphs.end(), CFG,
                         record);

  return S;
}

}  // namespace ripples

#endif
