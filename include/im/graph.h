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

template <typename VertexIDTy, typename AttributeTy>
struct Destination {
  Destination() {}

  Destination(const VertexIDTy v, const AttributeTy & a) :
      v(v), attribute(a) {}

  VertexIDTy v;
  AttributeTy attribute;
};

//! \brief The Graph data structure.
//!
//! \tparam VertexTy The type representing the vertex.
template <typename VertexTy, typename DestTy = Destination<VertexTy, float>>
class Graph {
 public:
  using size_type = size_t;
  using vertex_type = VertexTy;
  using dest_type  = DestTy;
  using edge_list_type = std::vector<dest_type>;

  class Neighborhood {
   public:
    using iterator = typename edge_list_type::iterator;

    Neighborhood(const vertex_type & v, edge_list_type & EL)
        : v_(v), edge_list_(&EL) {}

    iterator begin() { return edge_list_->begin(); }
    iterator end() { return edge_list_->end(); }
   private:
    vertex_type v_;
    edge_list_type * edge_list_;
  };

 private:
  using GraphTy_ =
      std::unordered_map<vertex_type, std::pair<edge_list_type, edge_list_type>>;

  class vertex_iterator : public GraphTy_::iterator {
   public:
    vertex_iterator() : GraphTy_::iterator() {}
    vertex_iterator(typename GraphTy_::iterator Itr) : GraphTy_::iterator(Itr) {}

    vertex_type const * operator->() { return &GraphTy_::iterator::operator->()->first; }
    vertex_type operator*() { return GraphTy_::iterator::operator*().first; }
  };

 public:
  using neighborhood_type = Neighborhood;
  using iterator = vertex_iterator;
  using const_iterator = typename GraphTy_::const_iterator;

  void add_edge(const vertex_type &source, const dest_type &destination) {
    graph_[source].first.emplace_back(destination);
    graph_[destination.v].second.emplace_back(
        dest_type(source, destination.attribute));
    ++size_;
  }

  size_type out_degree(const vertex_type &v) const {
    return graph_.at(v).first.size();
  }

  size_type in_degree(const vertex_type &v) const {
    return graph_.at(v).second.size();
  }

  neighborhood_type out_neighbors(const vertex_type & v) {
    return neighborhood_type(v, graph_[v].first);
  }

  neighborhood_type in_neighbors(const vertex_type & v) {
    return neighborhood_type(v, graph_[v].second);
  }

  iterator begin() { return vertex_iterator(graph_.begin()); }
  iterator end() { return vertex_iterator(graph_.end()); }

  //! \brief The size of the Graph
  //! \return the size of the graph (number of edges).
  size_type size() const noexcept { return size_; }

  //! \brief The scale of the Graph
  //! \return the scale of the graph (number of vertices).
  size_type scale() const noexcept { return graph_.size(); }

  friend std::ostream& operator<<(std::ostream &S, Graph &G) {
    S << "G {\n";

    for (auto v : G) {
      S << "[" << v << "]";

      S << " out : {";
      for (auto e : G.out_neighbors(v)) {
        S << " (" << e.v << "," << e.attribute << ")";
      }
      S << " }";
      
      S << " in : {";
      for (auto e : G.in_neighbors(v)) {
        S << " (" << e.v << "," << e.attribute << ")";
      }
      S << " }";

      S << "\n";
    }
    
    S << "}\n";
    return S;
  }

 private:
  size_type size_;
  GraphTy_ graph_;
};

}  // namespace im

#endif  // IM_GRAPH_H
