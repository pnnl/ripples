//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2017 Pacific Northwest National Laboratory
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_set>

#include "benchmark/benchmark.h"

#include "boost/program_options.hpp"

#include "im/graph.h"
#include "im/tim.h"
#include "im/loaders.h"

#include "omp.h"

struct Configuration {
  std::string InputGraph;
  size_t k;
  double epsilon;
};

Configuration Config;

Configuration ParseCmdOptions(int argc, char **argv) {
  namespace po = boost::program_options;

  Configuration CFG;

  po::options_description general("Configuration Options");
  general.add_options()("help,h", "Print this help message")(
      "input-graph,i", po::value<std::string>(&CFG.InputGraph)->required(),
      "The input file with the edge-list.")(
      "seed-set-size,k", po::value<size_t>(&CFG.k), "The size of the seed set")(
      "epsilon,e", po::value<double>(&CFG.epsilon)->default_value(0.001),
      "The approximation factor.");

  po::variables_map VM;
  try {
    po::store(po::parse_command_line(argc, argv, general), VM);

    if (VM.count("help")) {
      std::cout << argv[0] << " [options]" << std::endl;
      std::cout << general << std::endl;
      exit(0);
    }

    po::notify(VM);
  } catch (po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << general << std::endl;
    exit(-1);
  }

  return CFG;
}

class TIM : public benchmark::Fixture {
 public:
  TIM()
      : benchmark::Fixture()
      , GraphPtr_(nullptr)
  {}

  void SetUp(benchmark::State& st) {
    if (!GraphPtr_) {
      GraphPtr_ = std::make_shared<im::Graph<uint32_t>>();
      im::load(Config.InputGraph, *GraphPtr_, im::weighted_edge_list_tsv());
    }
  }

  std::shared_ptr<im::Graph<uint32_t>> GraphPtr_;
  size_t theta;
  std::vector<std::unordered_set<uint32_t>> RRRSet_;
};

BENCHMARK_DEFINE_F(TIM, ThetaEstimation)(benchmark::State& state) {
  omp_set_num_threads(state.range(0));
  for (auto _ : state) {
    theta = thetaEstimation(*GraphPtr_, Config.k, Config.epsilon);
  }
}

BENCHMARK_REGISTER_F(TIM, ThetaEstimation)
->UseRealTime()
->Unit(benchmark::kMillisecond)
->RangeMultiplier(2)
->Range(1, omp_get_max_threads());


BENCHMARK_DEFINE_F(TIM, GenerateRRR)(benchmark::State& state) {
  omp_set_num_threads(state.range(0));

  size_t theta = thetaEstimation(*GraphPtr_, Config.k, Config.epsilon);
  for (auto _ : state) {
    RRRSet_ = std::move(im::generateRandomRRSet(*GraphPtr_, theta, im::tim_tag()));
    state.PauseTiming();
    RRRSet_.clear();
    state.ResumeTiming();
  }
}

BENCHMARK_REGISTER_F(TIM, GenerateRRR)
->UseRealTime()
->Unit(benchmark::kMillisecond)
->RangeMultiplier(2)
->Range(1, omp_get_max_threads());

int main(int argc, char *argv[]) {
  benchmark::Initialize(&argc, argv);
  Config = ParseCmdOptions(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
