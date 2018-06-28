//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_GRAPH_H
#define IM_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace im {

template <typename VertexTy, typename WeightTy>
struct Edge {
  using vertex_type = VertexTy;
  using weight_type = WeightTy;
  VertexTy source;
  VertexTy destination;
  WeightTy weight;
};

template <typename VertexTy, typename WeightTy>
class Graph {
 public:
  using size_type = size_t;
  using edge_type = Edge<VertexTy, WeightTy>;
  using vertex_type = VertexTy;

  template <typename EdgeIterator>
  Graph(EdgeIterator begin, EdgeIterator end) {
    std::map<VertexTy, VertexTy> idMap;
    for (auto itr = begin; itr != end; ++itr) {
      idMap[itr->source];
      idMap[itr->destination];
    }

    size_t nodes = idMap.size();
    size_t edges = std::distance(begin, end);

    inIndex = new DestinationTy *[nodes + 1]{nullptr};
    inEdges = new DestinationTy[edges];
    outIndex = new DestinationTy *[nodes + 1]{nullptr};
    outEdges = new DestinationTy[edges];

    numNodes = nodes;
    numEdges = edges;

    VertexTy currentID{0};
    for (auto itr = std::begin(idMap), end = std::end(idMap); itr != end;
         ++itr) {
      itr->second = currentID++;
    }

    for (auto itr = begin; itr != end; ++itr) {
      itr->source = idMap[itr->source];
      itr->destination = idMap[itr->destination];
      *(reinterpret_cast<size_t *>(&inIndex[itr->destination + 1])) +=
          sizeof(DestinationTy);
      *(reinterpret_cast<size_t *>(&outIndex[itr->source + 1])) +=
          sizeof(DestinationTy);
    }

    inIndex[0] = inEdges;
    outIndex[0] = outEdges;
    for (size_t i = 1; i <= nodes; ++i) {
      *reinterpret_cast<size_t *>(&inIndex[i]) +=
          reinterpret_cast<size_t>(inIndex[i - 1]);
      *reinterpret_cast<size_t *>(&outIndex[i]) +=
          reinterpret_cast<size_t>(outIndex[i - 1]);
    }

    std::vector<DestinationTy *> ptrInEdge(inIndex, inIndex + nodes);
    std::vector<DestinationTy *> ptrOutEdge(outIndex, outIndex + nodes);
    for (auto itr = begin; itr != end; ++itr) {
      *ptrInEdge.at(itr->destination) = {itr->source, itr->weight};
      ++ptrInEdge[itr->destination];

      *ptrOutEdge[itr->source] = {itr->destination, itr->weight};
      ++ptrOutEdge[itr->source];
    }
  }

  ~Graph() {
    delete[] inIndex;
    delete[] inEdges;

    delete[] outIndex;
    delete[] outEdges;
  }

  struct DestinationTy {
    VertexTy vertex;
    WeightTy weight;
  };

  class Neighborhood {
   public:
    Neighborhood(DestinationTy *B, DestinationTy *E) : begin_(B), end_(E) {}
    DestinationTy *begin() const { return begin_; }
    DestinationTy *end() const { return end_; }

   private:
    DestinationTy *begin_;
    DestinationTy *end_;
  };

  size_t in_degree(VertexTy v) const { return inIndex[v + 1] - inIndex[v]; }
  size_t out_degree(VertexTy v) const { return outIndex[v + 1] - outIndex[v]; }

  Neighborhood out_neighbors(VertexTy v) const {
    return Neighborhood(outIndex[v], outIndex[v + 1]);
  }

  Neighborhood in_neighbors(VertexTy v) const {
    return Neighborhood(inIndex[v], inIndex[v + 1]);
  }

  size_t num_nodes() const { return numNodes; }
  size_t num_edges() const { return numEdges; }

 private:
  DestinationTy **inIndex;
  DestinationTy *inEdges;

  DestinationTy **outIndex;
  DestinationTy *outEdges;

  size_t numNodes;
  size_t numEdges;
};

}  // namespace im

#endif  // IM_GRAPH_H
