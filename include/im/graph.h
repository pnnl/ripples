//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_GRAPH_H
#define IM_GRAPH_H

#include <iostream>
#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

namespace im {

template <typename VertexTy, typename WeightTy>
struct Edge {
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

  template<typename EdgeIterator>
  Graph(EdgeIterator begin, EdgeIterator end) {
    std::map<VertexTy, VertexTy> idMap;
    for (auto itr = begin; itr != end; ++itr) {
      idMap[itr->source];
      idMap[itr->destination];
    }

    size_t nodes = idMap.size();
    size_t edges = std::distance(begin, end);

    inIndex = new VertexTy* [nodes + 1]{nullptr};
    inEdges = new VertexTy[edges]{0};
    inWeightIndex = new WeightTy* [nodes + 1]{nullptr};
    inEdgeWeights = new WeightTy[edges]{0};
    outIndex = new VertexTy* [nodes + 1]{nullptr};
    outEdges = new VertexTy[edges]{0};
    outWeightIndex = new WeightTy* [nodes + 1]{nullptr};
    outEdgeWeights = new WeightTy[edges]{0};

    numNodes = nodes;
    numEdges = edges;
    
    VertexTy currentID{0};
    for (auto itr = std::begin(idMap), end = std::end(idMap); itr != end; ++itr) {
      itr->second = currentID++;
    }

    for (auto itr = begin; itr != end; ++itr) {
      itr->source = idMap[itr->source];
      itr->destination = idMap[itr->destination];
      *(reinterpret_cast<size_t *>(&inIndex[itr->destination + 1])) += sizeof(VertexTy);
      *(reinterpret_cast<size_t *>(&outIndex[itr->source + 1])) += sizeof(VertexTy);
      *(reinterpret_cast<size_t *>(&inWeightIndex[itr->destination + 1])) += sizeof(VertexTy);
      *(reinterpret_cast<size_t *>(&outWeightIndex[itr->source + 1])) += sizeof(VertexTy);
    }

    inIndex[0] = inEdges;
    inWeightIndex[0] = inEdgeWeights;
    outIndex[0] = outEdges;
    outWeightIndex[0] = outEdgeWeights;
    for (size_t i = 1; i <= nodes; ++i) {
      *reinterpret_cast<size_t *>(&inIndex[i]) += reinterpret_cast<size_t>(inIndex[i - 1]);
      *reinterpret_cast<size_t *>(&inWeightIndex[i]) += reinterpret_cast<size_t>(inWeightIndex[i - 1]);
      *reinterpret_cast<size_t *>(&outIndex[i]) += reinterpret_cast<size_t>(outIndex[i - 1]);
      *reinterpret_cast<size_t *>(&outWeightIndex[i]) += reinterpret_cast<size_t>(outWeightIndex[i - 1]);
    }

    std::vector<VertexTy *> ptrInEdge(inIndex, inIndex + nodes);
    std::vector<VertexTy *> ptrOutEdge(outIndex, outIndex + nodes);
    std::vector<WeightTy *> ptrInWeight(inWeightIndex, inWeightIndex + nodes);
    std::vector<WeightTy *> ptrOutWeight(outWeightIndex, outWeightIndex + nodes);
    for (auto itr = begin; itr != end; ++itr) {
      *ptrInEdge.at(itr->destination) = itr->source;
      ++ptrInEdge[itr->destination];

      *ptrInWeight[itr->destination] = itr->weight;
      ++ptrInWeight[itr->destination];

      *ptrOutEdge[itr->source] = itr->destination;
      ++ptrOutEdge[itr->source];

      *ptrOutWeight[itr->source] = itr->weight;
      ++ptrOutWeight[itr->source];
    }
  }

  ~Graph() {
    delete[] inIndex;
    delete[] inEdges;
    delete[] inWeightIndex;
    delete[] inEdgeWeights;

    delete[] outIndex;
    delete[] outEdges;
    delete[] outWeightIndex;
    delete[] outEdgeWeights;
  }

  class Neighborhood {
   public:
    Neighborhood(VertexTy v, VertexTy * B, VertexTy *E)
        : v_(v), begin_(B), end_(E) {}
    VertexTy * begin() const { return begin_; }
    VertexTy * end() const { return end_; }
   private:
    VertexTy v_;
    VertexTy * begin_;
    VertexTy * end_;
  };

  size_t in_degree(VertexTy v) const { return inIndex[v + 1] - inIndex[v]; }
  size_t out_degree(VertexTy v) const { return outIndex[v + 1] - outIndex[v]; }

  Neighborhood out_neighbors(VertexTy v) const {
    return Neighborhood(v, outIndex[v], outIndex[v + 1]);
  }

  Neighborhood in_neighbors(VertexTy v) const {
    return Neighborhood(v, inIndex[v], inIndex[v + 1]);
  }

  size_t num_nodes() const { return numNodes; }
  size_t num_edges() const { return numEdges; }

 private:
  VertexTy** inIndex;
  VertexTy*  inEdges;
  WeightTy** inWeightIndex;
  WeightTy*  inEdgeWeights;

  VertexTy** outIndex;
  VertexTy*  outEdges;
  WeightTy** outWeightIndex;
  WeightTy*  outEdgeWeights;

  size_t numNodes;
  size_t numEdges;
};


}  // namespace im

#endif  // IM_GRAPH_H
