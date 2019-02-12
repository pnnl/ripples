//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_LOADERS_H
#define IM_LOADERS_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "trng/uniform01_dist.hpp"

#include "im/diffusion_simulation.h"
#include "im/graph.h"

namespace im {

struct edge_list_tsv {};
struct weighted_edge_list_tsv {};

template <typename EdgeTy, typename PRNG, typename diff_model_tag>
std::vector<EdgeTy> load(const std::string &inputFile, const bool undirected, PRNG &rand,
                         const edge_list_tsv &&, const diff_model_tag &&) {
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

  if (std::is_same<diff_model_tag, im::linear_threshold_tag>::value) {
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

template <typename EdgeTy, typename PRNG, typename diff_model_tag>
std::vector<EdgeTy> load(const std::string &inputFile, const bool undirected, PRNG &rand,
                         const weighted_edge_list_tsv &&, diff_model_tag &&) {
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


template<typename EdgeTy, typename Configuration, typename PRNG>
std::vector<EdgeTy> loadEdgeList(const Configuration & CFG, PRNG & weightGen) {
  std::vector<EdgeTy> edgeList;
  if (CFG.weighted) {
    if (CFG.diffusionModel == "IC") {
      edgeList = load<EdgeTy>(
          CFG.IFileName, CFG.undirected, weightGen,
          im::weighted_edge_list_tsv{}, im::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = load<EdgeTy>(
          CFG.IFileName, CFG.undirected, weightGen,
          im::weighted_edge_list_tsv{}, im::linear_threshold_tag{});
    }
  } else {
    if (CFG.diffusionModel == "IC") {
      edgeList = load<EdgeTy>(
          CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv{},
          im::independent_cascade_tag{});
    } else if (CFG.diffusionModel == "LT") {
      edgeList = load<EdgeTy>(
          CFG.IFileName, CFG.undirected, weightGen, im::edge_list_tsv{},
          im::linear_threshold_tag{});
    }
  }
  return edgeList;
}

}  // namespace im

#endif /* LOADERS_H */
