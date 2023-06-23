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

#include <filesystem>
#include <iostream>
#include <string>

#include "ripples/configuration.h"
#include "ripples/graph.h"
#include "ripples/graph_dump.h"
#include "ripples/loaders.h"

#include "CLI/CLI.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/lcg64.hpp"

struct DumpOutputConfiguration {
  std::string OName{"output"};
  bool binaryDump{false};
  bool normalize{false};

  void addCmdOptions(CLI::App &app) {
    app.add_option("-o,--output", OName, "The name of the output file name")
        ->required()
        ->group("Output Options");
    app.add_flag("--dump-binary", binaryDump,
                 "Dump the Graph in binary format.")
        ->group("Output Options");
    app.add_flag("--normalize", normalize,
                 "Dump the Graph in text format with vertices starting from 1")
        ->group("Output Options");
  }
};

struct DumpConfiguration {
  std::string diffusionModel{"IC"};  //!< The diffusion model to use.

  void addCmdOptions(CLI::App &app) {
    app.add_option("-d,--diffusion-model", diffusionModel,
                   "The diffusion model to be used (LT|IC)")
        ->required()
        ->group("Tool Options");
  }
};

using Configuration =
    ripples::ToolConfiguration<DumpConfiguration, DumpOutputConfiguration>;

int main(int argc, char **argv) {
  Configuration CFG;
  CFG.ParseCmdOptions(argc, argv);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  spdlog::set_level(spdlog::level::info);

  using Graph = ripples::Graph<uint32_t>;
  auto console = spdlog::stdout_color_st("console");
  console->info("Loading...");
  auto loading_start = std::chrono::high_resolution_clock::now();
  Graph G = ripples::loadGraph<Graph>(CFG, weightGen);
  auto loading_end = std::chrono::high_resolution_clock::now();
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", G.num_nodes());
  console->info("Number of Edges : {}", G.num_edges());
  const auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                             loading_end - loading_start)
                             .count();
  console->info("Loading took {}ms", load_time);

  if (CFG.binaryDump) {
    // Dump in binary format
    auto file = std::fstream(CFG.OName, std::ios::out | std::ios::binary);
    G.dump_binary(file);
    file.close();
  } else {
    auto file = std::fstream(CFG.OName, std::ios::out);
    dumpGraph(G, file, CFG.normalize);
    file.close();
  }

  return EXIT_SUCCESS;
}
