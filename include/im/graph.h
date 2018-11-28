//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_GRAPH_H
#define IM_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>

namespace im {

template <typename VertexTy>
struct ForwardDirection {
  template <typename ItrTy>
  static VertexTy Source(ItrTy itr) { return itr->source; }

  template <typename ItrTy>
  static VertexTy Destination(ItrTy itr) { return itr->destination; }
};


template <typename VertexTy>
struct BackwardDirection {
  template <typename ItrTy>
  static VertexTy Source(ItrTy itr) { return itr->destination; }

  template <typename ItrTy>
  static VertexTy Destination(ItrTy itr) { return itr->source; }
};


//! \brief The structure storing edges of the a weighted graph.
//!
//! \tparm VertexTy The integer type representing a vertex of the graph.
//! \tparm WeightTy The type representing the weight on the edge.
template <typename VertexTy, typename WeightTy>
struct Edge {
  //! The integer type representing vertices in the graph.
  using vertex_type = VertexTy;
  //! The type representing weight on the edges of the graph.
  using weight_type = WeightTy;
  //! The source of the edge.
  VertexTy source;
  //! The destination of teh edge.
  VertexTy destination;
  //! The weight on the edge.
  WeightTy weight;
};

//! \brief The Graph data structure.
//!
//! This graph data structure is nothing more than a classical CSR
//! representation of the adjacent matrix of the graph.  For efficiency
//! purposes, it stores the matrix and its transpose so that it is easy to walk
//! edges in both directions.
//!
//! The construction method takes care of projecting the vercites in the input
//! edge list into a contiguous space [0; N[ of integers in order to build the
//! CSR representation.  However, the data structure stores a map that allows to
//! project back the IDs into the original space.
//!
//! \tparm VertexTy The integer type representing a vertex of the graph.
//! \tparm WeightTy The type representing the weight on the edge.
template <typename VertexTy, typename WeightTy,
          typename DirectionPolicy = ForwardDirection<VertexTy> >
class Graph {
 public:
  //! The size type.
  using size_type = size_t;
  //! The type of an edge in the graph.
  using edge_type = Edge<VertexTy, WeightTy>;
  //! The integer type representing vertices in the graph.
  using vertex_type = VertexTy;
  //! The type representing the weights on the edges of the graph.
  using edge_weight_type = WeightTy;

  //! \brief Constructor.
  //!
  //! Build a Graph from a sequence of edges.
  //!
  //! \tparam EdgeIterator The iterator type used to visit the input edge list.
  //!
  //! \param begin The start of the edge list.
  //! \param end The end of the edge list.
  template <typename EdgeIterator>
  Graph(EdgeIterator begin, EdgeIterator end) {
    std::map<VertexTy, VertexTy> idMap;
    for (auto itr = begin; itr != end; ++itr) {
      idMap[itr->source];
      idMap[itr->destination];
    }

    size_t nodes = idMap.size();
    size_t edges = std::distance(begin, end);

    inIndex = new DestinationTy *[nodes + 1];
    inEdges = new DestinationTy[edges];

#pragma omp parallel for simd
    for (size_t i = 0; i < nodes + 1; ++i) {
      inIndex[i] = inEdges;
    }

#pragma omp parallel for simd
    for (size_t i = 0; i < edges; ++i) {
      inEdges[i] = DestinationTy();
    }

    numNodes = nodes;
    numEdges = edges;

    VertexTy currentID{0};
    reverseMap.resize(numNodes);
    for (auto itr = std::begin(idMap), end = std::end(idMap); itr != end;
         ++itr) {
      reverseMap[currentID] = itr->first;
      itr->second = currentID++;
    }

    for (auto itr = begin; itr != end; ++itr) {
      itr->source = idMap[itr->source];
      itr->destination = idMap[itr->destination];
      inIndex[DirectionPolicy::Destination(itr) + 1] += 1;

      // *(reinterpret_cast<size_t *>(&outIndex[itr->source + 1])) +=
      //     sizeof(DestinationTy);
    }

    for (size_t i = 1; i <= nodes; ++i) {
      inIndex[i] += inIndex[i - 1] - inEdges;
    }

    std::vector<DestinationTy *> ptrInEdge(inIndex, inIndex + nodes);
    for (auto itr = begin; itr != end; ++itr) {
      *ptrInEdge[DirectionPolicy::Destination(itr)] = { DirectionPolicy::Source(itr), itr->weight };
      ++ptrInEdge[DirectionPolicy::Destination(itr)];

      // *ptrOutEdge[itr->source] = {itr->destination, itr->weight};
      // ++ptrOutEdge[itr->source];
    }
  }

  //! \brief Destuctor.
  ~Graph() {
    delete[] inIndex;
    delete[] inEdges;
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

  //! Returns the in-degree of a vertex.
  //! \param v The input vertex.
  //! \return the in-degree of vertex v in input.
  size_t degree(VertexTy v) const { return inIndex[v + 1] - inIndex[v]; }

  //! Returns the out-degree of a vertex.
  //! \param v The input vertex.
  //! \return the out-degree of vertex v in input.
  // size_t out_degree(VertexTy v) const { return outIndex[v + 1] - outIndex[v]; }

  //! Returns the out-neighbors of a vertex.
  //! \param v The input vertex.
  //! \return the out-neighbors of vertex v in input.
  // Neighborhood out_neighbors(VertexTy v) const {
  //   return Neighborhood(outIndex[v], outIndex[v + 1]);
  // }

  //! Returns the in-neighbors of a vertex.
  //! \param v The input vertex.
  //! \return the in-neighbors of vertex v in input.
  Neighborhood neighbors(VertexTy v) const {
    return Neighborhood(inIndex[v], inIndex[v + 1]);
  }

  //! The number of nodes in the Graph.
  //! \return The number of nodes in the Graph.
  size_t num_nodes() const { return numNodes; }

  //! The number of edges in the Graph.
  //! \return The number of edges in the Graph.
  size_t num_edges() const { return numEdges; }

  //! Convert a list of vertices of G into IDs of the input edge list.
  //!
  //! \tparam Itr The input iterator of the sequence of input vertices.
  //! \tparam OutputItr The output iterator where the translated ids will be
  //! written.
  template <typename Itr, typename OutputItr>
  void convertID(Itr b, Itr e, OutputItr o) const {
    using value_type = typename Itr::value_type;
    std::transform(b, e, o, [&](const value_type &v) -> value_type {
      return reverseMap[v];
    });
  }

 private:
  DestinationTy **inIndex;
  DestinationTy *inEdges;

  std::vector<VertexTy> reverseMap;

  size_t numNodes;
  size_t numEdges;
};

}  // namespace im

#endif  // IM_GRAPH_H
