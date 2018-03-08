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
//! \tparam Vertex The type representing the vertex
//! \tparam Allocator The type of the allocator for the data structure.
template <typename Vertex, typename Allocator = std::allocator<Vertex> >
class Graph {
 public:
  using allocator_type = Allocator;
  using size_type = typename allocator_type::size_type;
  using vertex_type = Vertex;
  using edge_list_type = std::vector<std::pair<vertex_type, float>>;

  using iterator =
      typename std::unordered_map<vertex_type, edge_list_type>::iterator;
  using const_iterator =
      typename std::unordered_map<vertex_type, edge_list_type>::const_iterator;

  using edge_iterator = typename edge_list_type::iterator;
  using const_edge_iterator = typename edge_list_type::const_iterator;

  iterator begin() { return graph_.begin(); }
  const_iterator begin() const { return graph_.begin(); }

  iterator end() { return graph_.end(); }
  const_iterator end() const { return graph_.end(); }

  edge_iterator begin_out_edges(const vertex_type v) {
    return graph_[v].first.begin();
  }
  const_edge_iterator begin_out_edges(const vertex_type v) const {
    return graph_[v].first.begin();
  }

  edge_iterator end_out_edges(const vertex_type v) {
    return graph_[v].first.end();
  }
  const_edge_iterator end_out_edges(const vertex_type v) const {
    return graph_[v].first.end();
  }

  edge_iterator begin_in_edges(const vertex_type v) {
    return graph_[v].second.begin();
  }
  const_edge_iterator begin_in_edges(const vertex_type v) const {
    return graph_[v].second.begin();
  }

  edge_iterator end_in_edges(const vertex_type v) {
    return graph_[v].second.end();
  }
  const_edge_iterator end_in_edges(const vertex_type v) const {
    return graph_[v].second.end();
  }

  edge_list_type & operator[](vertex_type v) {
    return graph_[v].first;
  }

  const edge_list_type & operator[](vertex_type v) const {
    return graph_[v].first;
  }

  size_type size() const {
    size_type size = 0;
    for (auto &NL : graph_) size += NL.second.first.size();
    return size;
  }

  size_type scale() const { return graph_.size(); }

 private:
  std::unordered_map<
   vertex_type, std::pair<edge_list_type, edge_list_type>> graph_;
};

}  // namespace im

#endif  // IM_GRAPH_H
