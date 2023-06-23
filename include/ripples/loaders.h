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

#ifndef RIPPLES_LOADERS_H
#define RIPPLES_LOADERS_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "trng/lcg64.hpp"
#include "trng/truncated_normal_dist.hpp"
#include "trng/uniform01_dist.hpp"

namespace ripples {

//! Edge List in TSV format tag.
struct edge_list_tsv {};
//! Weighted Edge List in TSV format tag.
struct weighted_edge_list_tsv {};

namespace {

//! Load an Edge List in TSV format and generate the weights.
//!
//! \tparam EdgeTy The type of edges.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag The Type-Tag for the diffusion model.
//!
//! \param inputFile The name of the input file.
//! \param undirected When true, the edge list is from an undirected graph.
//! \param rand The random number generator.
template <typename EdgeTy, typename PRNG, typename diff_model_tag>
std::vector<EdgeTy> load(const std::string &inputFile, const bool undirected,
                         PRNG &rand, const edge_list_tsv &&,
                         const diff_model_tag &&) {
  std::ifstream GFS(inputFile);
  size_t lineNumber = 0;

  trng::uniform01_dist<float> probability;

  std::vector<EdgeTy> result;
  for (std::string line; std::getline(GFS, line); ++lineNumber) {
    if (line.empty()) continue;
    if (line.find('%') != std::string::npos) continue;
    if (line.find('#') != std::string::npos) continue;

    std::stringstream SS(line);

    typename EdgeTy::vertex_type source;
    typename EdgeTy::vertex_type destination;
    typename EdgeTy::weight_type weight;
    SS >> source >> destination;

    weight = rand();
    EdgeTy e = {source, destination, weight};
    result.emplace_back(e);

    if (undirected) {
      weight = rand();
      EdgeTy e = {destination, source, weight};
      result.emplace_back(e);
    }
  }

  if (std::is_same<diff_model_tag, ripples::linear_threshold_tag>::value) {
    auto cmp = [](const EdgeTy &a, const EdgeTy &b) -> bool {
      return a.source < b.source;
    };

    std::sort(result.begin(), result.end(), cmp);

    for (auto begin = result.begin(); begin != result.end();) {
      auto end = std::upper_bound(begin, result.end(), *begin, cmp);
      typename EdgeTy::weight_type not_taking = rand();
      typename EdgeTy::weight_type total = std::accumulate(
          begin, end, not_taking,
          [](const typename EdgeTy::weight_type &a, const EdgeTy &b) ->
          typename EdgeTy::weight_type { return a + b.weight; });

      std::transform(begin, end, begin, [=](EdgeTy &e) -> EdgeTy {
        e.weight /= total;
        return e;
      });

      begin = end;
    }
  }

  return result;
}

//! Load a Weighted Edge List in TSV format.
//!
//! \tparam EdgeTy The type of edges.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag The Type-Tag for the diffusion model.
//!
//! \param inputFile The name of the input file.
//! \param undirected When true, the edge list is from an undirected graph.
//! \param rand The random number generator.
template <typename EdgeTy, typename PRNG, typename diff_model_tag>
std::vector<EdgeTy> load(const std::string &inputFile, const bool undirected,
                         PRNG &rand, const weighted_edge_list_tsv &&,
                         diff_model_tag &&) {
  std::ifstream GFS(inputFile);
  size_t lineNumber = 0;

  std::vector<EdgeTy> result;
  for (std::string line; std::getline(GFS, line); ++lineNumber) {
    if (line.empty()) continue;
    if (line.find('%') != std::string::npos) continue;
    if (line.find('#') != std::string::npos) continue;

    std::stringstream SS(line);

    typename EdgeTy::vertex_type source;
    typename EdgeTy::vertex_type destination;
    typename EdgeTy::weight_type weight;
    SS >> source >> destination >> weight;

    EdgeTy e = {source, destination, weight};
    result.emplace_back(e);

    if (undirected) {
      EdgeTy e = {destination, source, weight};
      result.emplace_back(e);
    }
  }
  return result;
}

}  // namespace

template <typename PRNG, typename Distribution>
class WeightGenerator {
 public:
  WeightGenerator(PRNG &gen, Distribution dist, float scale_factor = 1.0)
      : gen_(gen), dist_(dist), scale_factor_(scale_factor) {}

  WeightGenerator(PRNG &gen, float scale_factor = 1.0)
      : WeightGenerator(gen, Distribution(), scale_factor) {}

  float operator()() { return scale_factor_ * dist_(gen_); }

 private:
  PRNG gen_;
  Distribution dist_;
  float scale_factor_;
};

//! Load an Edge List.
//!
//! \tparam EdgeTy The type of edges.
//! \tparam Configuration The type describing the input of the tool.
//! \tparam PRNG The type of the parallel random number generator.
//!
//! \param CFG The input configuration.
//! \param weightGen The random number generator used to generate the weights.
template <typename EdgeTy, typename Configuration, typename PRNG>
std::vector<EdgeTy> loadEdgeList(const Configuration &CFG, PRNG &weightGen) {
  std::vector<EdgeTy> edgeList;
  if (CFG.weighted) {
    if (CFG.diffusionModel == "IC") {
      edgeList = load<EdgeTy>(CFG.IFileName, CFG.undirected, weightGen,
                              ripples::weighted_edge_list_tsv{},
                              ripples::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = load<EdgeTy>(CFG.IFileName, CFG.undirected, weightGen,
                              ripples::weighted_edge_list_tsv{},
                              ripples::linear_threshold_tag{});
    }
  } else {
    if (CFG.diffusionModel == "IC") {
      edgeList = load<EdgeTy>(CFG.IFileName, CFG.undirected, weightGen,
                              ripples::edge_list_tsv{},
                              ripples::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = load<EdgeTy>(CFG.IFileName, CFG.undirected, weightGen,
                              ripples::edge_list_tsv{},
                              ripples::linear_threshold_tag{});
    }
  }
  return edgeList;
}

namespace {
template <typename GraphTy, typename ConfTy, typename PrngTy, typename allocator_t = std::allocator<char>>
GraphTy loadGraph_helper(ConfTy &CFG, PrngTy &PRNG, allocator_t allocator = allocator_t()) {
  GraphTy G(allocator);

  if (!CFG.reload) {
    using vertex_type = typename GraphTy::vertex_type;
    using weight_type = typename GraphTy::edge_type::edge_weight;
    using edge_type = ripples::Edge<vertex_type, weight_type>;
    auto edgeList = ripples::loadEdgeList<edge_type>(CFG, PRNG);
    GraphTy tmpG(edgeList.begin(), edgeList.end(), !CFG.disable_renumbering, allocator);
    G = std::move(tmpG);
  } else {
    std::ifstream binaryDump(CFG.IFileName, std::ios::binary);
    // GraphTy tmpG(binaryDump, allocator);
    // G = std::move(tmpG);
    G.load_binary(binaryDump);
  }

  return G;
}
}  // namespace

//! Load Graphs.
//!
//! \tparam GraphTy The type of the graph to be loaded.
//! \tparam ConfTy  The type of the configuration object.
//! \tparam PrngTy  The type of the parallel random number generator object.
//!
//! \param CFG The configuration object.
//! \param PRNG The parallel random number generator.
//! \return The GraphTy graph loaded from the input file.
template <typename GraphTy, typename ConfTy, typename PrngTy, typename allocator_t = std::allocator<char>>
GraphTy loadGraph(ConfTy &CFG, PrngTy &PRNG, allocator_t allocator = allocator_t()) {
  GraphTy G(allocator);
  if (CFG.distribution == "uniform") {
    WeightGenerator<trng::lcg64, trng::uniform01_dist<float>> gen(
        PRNG, CFG.scale_factor);
    G = loadGraph_helper<GraphTy>(CFG, gen, allocator);
  } else if (CFG.distribution == "normal") {
    WeightGenerator<trng::lcg64, trng::truncated_normal_dist<float>> gen(
        PRNG,
        trng::truncated_normal_dist<float>(CFG.mean, CFG.variance, 0.0, 1.0),
        CFG.scale_factor);
    G = loadGraph_helper<GraphTy>(CFG, gen, allocator);
  } else if (CFG.distribution == "const") {
    auto gen = [&]() -> float { return CFG.mean; };
    G = loadGraph_helper<GraphTy>(CFG, gen, allocator);
  } else {
    throw std::domain_error("Unsupported distribution");
  }
  return G;
}

}  // namespace ripples

#endif /* LOADERS_H */
