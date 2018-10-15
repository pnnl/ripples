//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "CLI11/CLI11.hpp"
#include "benchmark/benchmark.h"

#include "im/graph.h"
#include "im/loaders.h"
#include "im/tim.h"

struct Configuration {
  std::string InputGraph;
  size_t k;
  double epsilon;
};

Configuration Config;

Configuration ParseCmdOptions(int argc, char** argv) {
  Configuration CFG;

  bool tim = true;

  CLI::App app{"Yet another tool to experiment with INF-MAX"};
  app.add_option("-i,--input-graph", CFG.InputGraph,
                 "The input file with the edge-list.")
      ->required();
  app.add_option("-k,--seed-set-size", CFG.k, "The size of the seed set.")
      ->required();
  app.add_option("-e,--epsilon", CFG.epsilon, "The size of the seed set.")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    exit(app.exit(e));
  }

  return CFG;
}

class TIM : public benchmark::Fixture {
 public:
  TIM() : benchmark::Fixture(), GraphPtr_(nullptr) {}

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

BENCHMARK_DEFINE_F(TIM, KptEstimation)(benchmark::State& state) {
  for (auto _ : state) {
    theta = KptEstimation(*GraphPtr_, Config.k, Config.epsilon, 1. / 2.);
  }
}

BENCHMARK_REGISTER_F(TIM, KptEstimation)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

#if 0
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
#endif

int main(int argc, char* argv[]) {
  benchmark::Initialize(&argc, argv);
  Config = ParseCmdOptions(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
