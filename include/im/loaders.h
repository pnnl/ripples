//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_LOADERS_H
#define IM_LOADERS_H

#include <fstream>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <string>

#include "im/graph.h"

namespace im {

struct edge_list_tsv {};

template <typename EdgeTy>
std::vector<EdgeTy>
load(std::string &inputFile, const edge_list_tsv &&) {
  std::ifstream GFS(inputFile);
  size_t lineNumber = 0;

  std::vector<EdgeTy> result;
  for (std::string line; std::getline(GFS, line); ++lineNumber) {
    if (line.find('%') != std::string::npos) continue;

    std::stringstream SS(line);

    std::string source;
    std::string destination;
    double weight;
    SS >> source >> destination;

    if (source.length() == 0 && destination.length() == 0) continue;

    if (!(SS >> weight)) weight = 0.10;
    typename im::Graph<uint32_t, double>::vertex_type s = stoul(source);
    typename im::Graph<uint32_t, double>::vertex_type d = stoul(destination);
    typename im::Graph<uint32_t, double>::edge_type e{s, d, weight};
    result.emplace_back(e);
  }
  return result;
}

}  // namespace im

#endif /* LOADERS_H */
