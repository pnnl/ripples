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

#ifndef IM_GRAPH_H
#define IM_GRAPH_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "boost/lexical_cast.hpp"

namespace im {

//! \brief The Graph data structure.
//!
//! \tparam Vertex The type representing the vertex.
//! \tparam Weight The type Weight on the edges.
template <typename Vertex, typename EdgeWeight>
class Graph {
 public:
  using size_type = size_t;
  using vertex_type = Vertex;
  using edge_type = std::pair<vertex_type, EdgeWeight>;
  using edge_list_type = std::vector<edge_type>;

  struct Neighborhood {
    // using forward_star_iterator = 
    std::pair<edge_list_type, edge_list_type> NH_;
  };

  using neighborhood_type = Neighborhood;

  using iterator =
      typename std::unordered_map<vertex_type, edge_list_type>::iterator;
  using const_iterator =
      typename std::unordered_map<vertex_type, edge_list_type>::const_iterator;

  //! \brief The size of the Graph
  //! \return the size of the graph (number of edges).
  size_type size() const noexcept { return size_; }

  //! \brief The scale of the Graph
  //! \return the scale of the graph (number of vertices).
  size_type scale() const noexcept { return graph_.size(); }

 private:
  size_type size_;
  std::unordered_map<vertex_type, neighborhood_type> graph_;
};

}  // namespace im

#endif  // IM_GRAPH_H
