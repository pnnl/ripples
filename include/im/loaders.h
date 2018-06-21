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

    typename EdgeTy::vertex_type source;
    typename EdgeTy::vertex_type destination;
    typename EdgeTy::weight_type weight;
    SS >> source >> destination;

    if (!(SS >> weight)) weight = 0.10;
    EdgeTy e = {source, destination, weight};
    result.emplace_back(e);
  }
  return result;
}

}  // namespace im

#endif /* LOADERS_H */
