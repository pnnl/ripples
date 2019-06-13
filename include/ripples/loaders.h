//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_LOADERS_H
#define RIPPLES_LOADERS_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "trng/uniform01_dist.hpp"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"

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
    if (line.find('%') != std::string::npos) continue;
    if (line.find('#') != std::string::npos) continue;

    std::stringstream SS(line);

    typename EdgeTy::vertex_type source;
    typename EdgeTy::vertex_type destination;
    typename EdgeTy::weight_type weight;
    SS >> source >> destination;

    weight = probability(rand);
    EdgeTy e = {source, destination, weight};
    result.emplace_back(e);

    if (undirected) {
      weight = probability(rand);
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
      typename EdgeTy::weight_type not_taking = probability(rand);
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

}


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
      edgeList =
          load<EdgeTy>(CFG.IFileName, CFG.undirected, weightGen,
                       ripples::edge_list_tsv{}, ripples::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = load<EdgeTy>(CFG.IFileName, CFG.undirected, weightGen,
                              ripples::edge_list_tsv{}, ripples::linear_threshold_tag{});
    }
  }
  return edgeList;
}

//! Load Graphs.
//!
//! \tparam GraphTy The type of the graph to be loaded.
//! \tparam ConfTy  The type of the configuration object.
//! \tparam PrngTy  The type of the parallel random number generator object.
//!
//! \param CFG The configuration object.
//! \param PRNG The parallel random number generator.
//! \return The GraphTy graph loaded from the input file.
template <typename GraphTy, typename ConfTy, typename PrngTy>
GraphTy loadGraph(ConfTy & CFG, PrngTy & PRNG) {
  GraphTy G;

  if (!CFG.reload) {
    using edge_type = typename GraphTy::edge_type;
    auto edgeList = ripples::loadEdgeList<edge_type>(CFG, PRNG);
    GraphTy tmpG(edgeList.begin(), edgeList.end());
    G = std::move(tmpG);
  } else {
    std::ifstream binaryDump(CFG.IFileName, std::ios::binary);
    GraphTy tmpG(binaryDump);
    G = std::move(tmpG);
  }

  return G;
}

}  // namespace ripples

#endif /* LOADERS_H */
