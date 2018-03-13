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

#ifndef IM_LOADERS_H
#define IM_LOADERS_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "boost/lexical_cast.hpp"

#include "im/graph.h"

namespace im {

struct edge_list_tsv {};

template <typename Format>
void load(std::string &inputFile, Graph<uint32_t> &G, const Format &&);

template <>
void
load<edge_list_tsv>(
    std::string &inputFile, Graph<uint32_t> &G, const edge_list_tsv &&) {
  std::ifstream GFS(inputFile);
  size_t lineNumber = 0;
  for (std::string line; std::getline(GFS, line); ++lineNumber) {
    if (line.find('%') != std::string::npos) continue;

    std::stringstream SS(line);

    std::string source;
    std::string destination;
    SS >> source >> destination;

    if (source.length() == 0 && destination.length() == 0) continue;

    try {
      G.add_edge(boost::lexical_cast<uint64_t>(source) - 1,
                 typename Graph<uint32_t>::dest_type(
                     boost::lexical_cast<uint64_t>(destination) - 1, 1.0f));
    } catch (boost::bad_lexical_cast &e) {
      std::cout << inputFile << ":" << lineNumber << " " << e.what()
                << std::endl;
      throw e;
    }
  }
}

struct weighted_edge_list_tsv {};

template <>
void
load<weighted_edge_list_tsv>(
    std::string &inputFile, Graph<uint32_t> &G, const weighted_edge_list_tsv &&) {
  std::ifstream GFS(inputFile);
  size_t lineNumber = 0;
  for (std::string line; std::getline(GFS, line); ++lineNumber) {
    if (line.find('%') != std::string::npos) continue;

    std::stringstream SS(line);

    std::string source;
    std::string destination;
    std::string weight;
    SS >> source >> destination >> weight;

    if (source.length() == 0 && destination.length() == 0) continue;

    try {
      G.add_edge(boost::lexical_cast<uint64_t>(source) - 1,
                 typename Graph<uint32_t>::dest_type(
                     boost::lexical_cast<uint64_t>(destination) - 1,
                     boost::lexical_cast<float>(weight)));
    } catch (boost::bad_lexical_cast &e) {
      std::cout << inputFile << ":" << lineNumber << " " << e.what()
                << std::endl;
      throw e;
    }
  }
}

}  // namespace im

#endif /* LOADERS_H */
