//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_LOADERS_H
#define IM_LOADERS_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "trng/uniform01_dist.hpp"

#include "im/graph.h"

namespace im {

struct edge_list_tsv {};
struct weighted_edge_list_tsv {};

template <typename EdgeTy, typename PRNG>
std::vector<EdgeTy> load(std::string &inputFile, bool undirected, PRNG & rand, const edge_list_tsv &&) {
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
  return result;
}

template <typename EdgeTy, typename PRNG>
std::vector<EdgeTy> load(std::string &inputFile, bool undirected, PRNG & rand, const weighted_edge_list_tsv &&) {
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

}  // namespace im

#endif /* LOADERS_H */
