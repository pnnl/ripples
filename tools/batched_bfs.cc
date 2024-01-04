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

#include <algorithm>
#include <bitset>
#include <cassert>
#include <map>
#include <numeric>
#include <ostream>
#include <string>

#include "ripples/configuration.h"
#include "ripples/graph.h"
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"
#include "ripples/gpu/bfs.h"
#include "thrust/device_vector.h"
#include "thrust/for_each.h"
#include "thrust/host_vector.h"
#include "thrust/inner_product.h"
#include "thrust/random.h"
#include "thrust/reduce.h"
#include "thrust/transform_scan.h"
#endif
#include "ripples/loaders.h"

#include "CLI/CLI.hpp"
#include "CLI/Validators.hpp"
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

class BFSToolConfiguration : public ripples::AlgorithmConfiguration {
 public:
  void addCmdOptions(CLI::App &app) {
    std::map<std::string, spdlog::level::level_enum> VerbosityMap{
        {"off", spdlog::level::off},    {"critical", spdlog::level::critical},
        {"error", spdlog::level::err},  {"warning", spdlog::level::warn},
        {"info", spdlog::level::info},  {"debug", spdlog::level::debug},
        {"trace", spdlog::level::trace}};
    app.add_option("--verbosity", logLevel_, "Output verbosity")
        ->group("General")
        ->default_val("6")
        ->transform(CLI::CheckedTransformer(VerbosityMap));
  }

  spdlog::level::level_enum LogLevel() const { return logLevel_; }

 private:
  spdlog::level::level_enum logLevel_;
};


int main(int argc, char **argv) {
  auto console = spdlog::stdout_color_st("console");

  ripples::ToolConfiguration<BFSToolConfiguration> CFG;

  CFG.ParseCmdOptions(argc, argv);

  spdlog::set_level(CFG.LogLevel());

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using dest_type = ripples::WeightedDestination<uint32_t, float>;
  using GraphFwd =
      ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>>;
  console->info("Loading...");
  GraphFwd G = ripples::loadGraph<GraphFwd>(CFG, weightGen);
  using vertex_type = typename GraphFwd::vertex_type;
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());

  std::vector<vertex_type> sources(8);
  trng::uniform_int_dist u(0, G.num_nodes());
  std::generate(sources.begin(), sources.end(), [&]() { return u(weightGen); });

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);

  auto DeviceContext = ripples::make_gpu_context<ripples::HIP>(G, 0);

  console->info("Start BFSs");
  std::vector<std::vector<vertex_type>> rrr_sets(sources.size());
  GPUBatchedBFS(G, *DeviceContext, std::begin(sources), std::end(sources),
                std::begin(rrr_sets), ripples::independent_cascade_tag{});
  console->info("End of BFSs");

  std::for_each(rrr_sets.begin(), rrr_sets.end(), [&](auto &v) {
    console->trace("Visited {} vertices", v.size());
    // std::sort(v.begin(), v.end());
    // for (size_t i = 0; i < v.size(); ++i)
    //   console->trace("--> Visited[{}] = {}", i,  v[i]);
    // console->trace("----------------------------------------------------------------------");
  });
  return EXIT_SUCCESS;
}
