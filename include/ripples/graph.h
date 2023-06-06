//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_GRAPH_H
#define RIPPLES_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <numeric>
#include <vector>
#include <execution>

#include <ripples/utility.h>

#if defined ENABLE_METALL
#include <metall/container/vector.hpp>
#include <metall/container/unordered_map.hpp>
#include <metall/metall.hpp>
#endif

namespace ripples {

//! \brief Forward Direction Graph loading policy.
//!
//! \tparam VertexTy The type of the vertex in the graph.
template <typename VertexTy>
struct ForwardDirection {
  //! \brief Edge Source
  //!
  //! \tparam ItrTy Edge iterator type.
  //!
  //! \param itr Iterator to the current edge.
  //! \return The source of the egde to be loaded in the graph.
  template <typename ItrTy, typename MapTy>
  static VertexTy Source(ItrTy itr, const MapTy &m) {
    return m.find(itr->source)->second;
  }

  //! \brief Edge Destination
  //!
  //! \tparam ItrTy Edge iterator type.
  //!
  //! \param itr Iterator to the current edge.
  //! \return The destination of the egde to be loaded in the graph.
  template <typename ItrTy, typename MapTy>
  static VertexTy Destination(ItrTy itr, const MapTy &m) {
    return m.find(itr->destination)->second;
  }
};

//! \brief Backward Direction Graph loading policy.
//!
//! \tparam VertexTy The type of the vertex in the graph.
template <typename VertexTy>
struct BackwardDirection {
  //! \brief Edge Source
  //!
  //! \tparam ItrTy Edge iterator type.
  //!
  //! \param itr Iterator to the current edge.
  //! \return The source of the egde to be loaded in the graph.
  template <typename ItrTy, typename MapTy>
  static VertexTy Source(ItrTy itr, const MapTy &m) {
    return m.find(itr->destination)->second;
  }

  //! \brief Edge Destination
  //!
  //! \tparam ItrTy Edge iterator type.
  //!
  //! \param itr Iterator to the current edge.
  //! \return The destination of the egde to be loaded in the graph.
  template <typename ItrTy, typename MapTy>
  static VertexTy Destination(ItrTy itr, const MapTy &m) {
    return m.find(itr->source)->second;
  }
};

//! \brief A weighted edge.
//!
//! \tparam VertexTy The integer type representing a vertex of the graph.
//! \tparam WeightTy The type representing the weight on the edge.
template <typename VertexTy, typename WeightTy = void>
struct Edge {
  //! The integer type representing vertices in the graph.
  using vertex_type = VertexTy;
  //! The type representing weights on the edges of the graph.
  using weight_type = WeightTy;
  //! The source of the edge.
  VertexTy source;
  //! The destination of the edge.
  VertexTy destination;
  //! The weight on the edge.
  WeightTy weight;

  bool operator==(const Edge &O) const {
    return O.source == this->source && O.destination == this->destination &&
           O.weight == this->weight;
  }
};

template <typename VertexTy>
struct Edge<VertexTy, void> {
  using vertex_type = VertexTy;

  VertexTy source;
  VertexTy destination;

  bool operator==(const Edge &O) const {
    return O.source == this->source && O.destination == this->destination;
  }
};

//! \brief CSR Edge for an unweighted graph.
//! \tparam VertexTy The type of the vertex.
template <typename VertexTy>
struct Destination {
  using vertex_type = VertexTy;
  //! The destination vertex of the edge.
  VertexTy vertex;

  bool operator==(const Destination &O) const {
    return this->vertex == O.vertex;
  }
  template <typename Direction, typename Itr, typename IDMap>
  static Destination Create(Itr itr, IDMap &IM) {
    Destination dst{Direction::Destination(itr, IM)};
    return dst;
  }
};

//! \brief The edges stored in the CSR.
template <typename VertexTy, typename WeightTy>
struct WeightedDestination : public Destination<VertexTy> {
  using edge_weight = WeightTy;
  WeightTy weight;  //!< The edge weight.

  WeightedDestination(VertexTy v, WeightTy w)
      : Destination<VertexTy>{v}, weight(w) {}
  WeightedDestination() : WeightedDestination(VertexTy(), WeightTy()) {}

  bool operator==(const WeightedDestination &O) const {
    return Destination<VertexTy>::operator==(O) && this->weight == O.weight;
  }
  template <typename Direction, typename Itr, typename IDMap>
  static WeightedDestination Create(Itr itr, IDMap &IM) {
    WeightedDestination dst{Direction::Destination(itr, IM), itr->weight};
    return dst;
  }
};

//! \brief The Graph data structure.
//!
//! A graph in CSR format.  The construction method takes care of projecting the
//! vercites in the input edge list into a contiguous space [0; N[ of integers
//! in order to build the CSR representation.  However, the data structure
//! stores a map that allows to project back the IDs into the original space.
//!
//! \tparam VertexTy The integer type representing a vertex of the graph.
//! \tparam DestinationTy The type representing the element of the edge array.
//! \tparam DirectionPolicy The policy encoding the graph direction with repect
//!    of the original data.
template <typename VertexTy,
          typename DestinationTy = WeightedDestination<VertexTy, float>,
          typename DirectionPolicy = ForwardDirection<VertexTy>,
          typename allocator_t = std::allocator<char>>
class Graph {
 public:
  //! The size type.
  using size_type = size_t;
  //! The type of an edge in the graph.
  using edge_type = DestinationTy;
  //! The integer type representing vertices in the graph.
  using vertex_type = VertexTy;

 private:
  // Pointer type for the edges array
  using edge_pointer_t = rebind_alloc_pointer<allocator_t, edge_type>;
  // Pointer type for the indices array
  using index_pointer_t = rebind_alloc_pointer<allocator_t, edge_pointer_t>;

 public:

  //! \brief The neighborhood of a vertex.
  class Neighborhood {
   public:
    //! Construct the neighborhood.
    //!
    //! \param B The begin of the neighbor list.
    //! \param E The end of the neighbor list.
    Neighborhood(edge_pointer_t B, edge_pointer_t E) : begin_(B), end_(E) {}

    //! Begin of the neighborhood.
    //! \return an iterator to the begin of the neighborhood.
    edge_pointer_t begin() const { return begin_; }
    //! End of the neighborhood.
    //! \return an iterator to the begin of the neighborhood.
    edge_pointer_t end() const { return end_; }

   private:
    edge_pointer_t begin_;
    edge_pointer_t end_;
  };

 //! Allocator Graph Constructor.
  Graph(allocator_t allocator = allocator_t())
      : numNodes(0),
        numEdges(0),
        index(nullptr),
        edges(nullptr),
        graph_allocator(allocator),
        idMap(allocator),
        reverseMap(allocator) {}

  Graph(const Graph &O)
      : numNodes(O.numNodes),
        numEdges(O.numEdges),
        idMap(O.idMap),
        reverseMap(O.reverseMap),
        graph_allocator(O.graph_allocator) {
    edges = allocate_edges(numEdges);
    index = allocate_index(numNodes + 1);

#pragma omp parallel for
    for (size_t i = 0; i < numEdges; ++i) {
      edges[i] = O.edges[i];
    }

#pragma omp parallel for
    for (size_t i = 0; i < numNodes + 1; ++i) {
      index[i] = edges + std::distance(O.index[0], O.index[i]);
    }
  }

  //! Copy assignment operator.
  //! \param O The graph to be copied.
  //! \return a reference to the destination graph.
  Graph &operator=(const Graph &O) {
    numNodes = O.numNodes;
    numEdges = O.numEdges;
    idMap = O.idMap;
    reverseMap = O.reverseMap;

    deallocate_index(index, numNodes + 1);
    deallocate_edges(edges, numEdges);

    index = allocate_index(numNodes + 1);
    edges = allocate_edges(numEdges);
#pragma omp parallel for
    for (size_t i = 0; i < numEdges; ++i) {
      edges[i] = O.edges[i];
    }

#pragma omp parallel for
    for (size_t i = 0; i < numNodes + 1; ++i) {
      index[i] = edges + std::distance(O.index[0], O.index[i]);
    }
    return *this;
  }

  //! Move constructor.
  //! \param O The graph to be moved.
  Graph(Graph &&O)
      : numNodes(O.numNodes),
        numEdges(O.numEdges),
        index(O.index),
        edges(O.edges),
        graph_allocator(std::move(O.graph_allocator)),
        idMap(std::move(O.idMap)),
        reverseMap(std::move(O.reverseMap)) {
    O.numNodes = 0;
    O.numEdges = 0;
    O.index = nullptr;
    O.edges = nullptr;
  }

  //! Move assignment operator.
  //! \param O The graph to be moved.
  //! \return a reference to the destination graph.
  Graph &operator=(Graph &&O) {
    if (this == &O) return *this;

    deallocate_index(index, numNodes + 1);
    deallocate_edges(edges, numEdges);

    numNodes = O.numNodes;
    numEdges = O.numEdges;
    index = O.index;
    edges = O.edges;
    idMap = std::move(O.idMap);
    reverseMap = std::move(O.reverseMap);

    O.numNodes = 0;
    O.numEdges = 0;
    O.index = nullptr;
    O.edges = nullptr;

    return *this;
  }

  //! \brief Constructor.
  //!
  //! Build a Graph from a sequence of edges.  The vertex identifiers are
  //! projected over the integer interval [0;N[.  The data structure stores
  //! conversion maps to move fro the internal representation of the vertex IDs
  //! to the original input representation.
  //!
  //! \tparam EdgeIterator The iterator type used to visit the input edge list.
  //!
  //! \param begin The start of the edge list.
  //! \param end The end of the edge list.
  template <typename EdgeIterator>
  Graph(EdgeIterator begin, EdgeIterator end, bool renumbering, allocator_t allocator = allocator_t())
  : graph_allocator(allocator),
    idMap(allocator),
    reverseMap(allocator){

    VertexTy currentID{0};
    VertexTy maxVertexID = 0;
    for (auto itr = begin; itr != end; ++itr) {
      if (idMap.count(itr->source) == 0) {
        idMap[itr->source];
        if (renumbering) {
          reverseMap.push_back(itr->source);
        }
      }

      if (idMap.count(itr->destination) == 0) {
        idMap[itr->destination];
        if (renumbering) {
          reverseMap.push_back(itr->destination);
        }
      }

      maxVertexID = std::max(std::max(itr->source, itr->destination), maxVertexID);
    }

    if (renumbering) {
      // TODO: enable the C++ 17 parallel sort
      std::sort(reverseMap.begin(), reverseMap.end());
      #pragma omp parallel for
      for (size_t i = 0; i < reverseMap.size(); ++i) {
        idMap.at(reverseMap.at(i)) = i;
      }
    } else {
      reverseMap.resize(maxVertexID + 1);
      #pragma omp parallel for
      for (VertexTy id = 0; id <= maxVertexID; ++id) {
        reverseMap.at(id) = id;
        idMap.at(id) = id;
      }
    }

    size_t num_nodes = renumbering ? idMap.size() : maxVertexID + 1;
    size_t num_edges = std::distance(begin, end);

    edges = allocate_edges(num_edges);
    index = allocate_index(num_nodes + 1);

#pragma omp parallel for
    for (size_t i = 0; i < num_nodes + 1; ++i) {
      index[i] = edges;
    }

#pragma omp parallel for
    for (size_t i = 0; i < num_edges; ++i) {
      edges[i] = DestinationTy();
    }

    numNodes = num_nodes;
    numEdges = num_edges;

    for (auto itr = begin; itr != end; ++itr) {
      index[DirectionPolicy::Source(itr, idMap) + 1] += 1;
    }

    for (size_t i = 1; i <= num_nodes; ++i) {
      index[i] += index[i - 1] - edges;
    }

    std::vector<edge_pointer_t> ptrEdge(index, index + num_nodes);
    for (auto itr = begin; itr != end; ++itr) {
      *ptrEdge[DirectionPolicy::Source(itr, idMap)] =
          edge_type::template Create<DirectionPolicy>(itr, idMap);
      ++ptrEdge[DirectionPolicy::Source(itr, idMap)];
    }
  }

  //! \brief Destuctor.
  ~Graph() {
    deallocate_index(index, numNodes + 1);
    deallocate_edges(edges, numEdges);
  }

  //! Returns the out-degree of a vertex.
  //! \param v The input vertex.
  //! \return the in-degree of vertex v in input.
  size_t degree(VertexTy v) const { return index[v + 1] - index[v]; }

  //! Returns the neighborhood of a vertex.
  //! \param v The input vertex.
  //! \return  a range containing the out-neighbors of the vertex v in input.
  Neighborhood neighbors(VertexTy v) const {
    return Neighborhood(index[v], index[v + 1]);
  }

  //! The number of nodes in the Graph.
  //! \return The number of nodes in the Graph.
  size_t num_nodes() const { return numNodes; }

  //! The number of edges in the Graph.
  //! \return The number of edges in the Graph.
  size_t num_edges() const { return numEdges; }

  //! Convert a list of vertices from the interal representation to the original
  //! input representation.
  //!
  //! \tparam Itr The iterator type of the input sequence of vertex IDs.
  //! \tparam OutputItr The iterator type of the output sequence.
  //!
  //! \param b The begin of the input vertex IDs sequence.
  //! \param e The end of the input vertex IDs sequence.
  //! \param o The start of the output vertex IDs sequence.
  template <typename Itr, typename OutputItr>
  void convertID(Itr b, Itr e, OutputItr o) const {
    using value_type = typename Itr::value_type;
    std::transform(b, e, o, [&](const value_type &v) -> value_type {
      return reverseMap.at(v);
    });
  }

  //! Convert a vertex from the interal representation to the original input
  //! representation.
  //!
  //! \param v The input vertex ID.
  //! \return The original vertex ID in the input representation.
  vertex_type convertID(const vertex_type v) const { return reverseMap.at(v); }

  //! Convert a list of vertices from the original input edge list
  //! representation to the internal vertex representation.
  //!
  //! \tparam Itr The iterator type of the input sequence of vertex IDs.
  //! \tparam OutputItr The iterator type of the output sequence.
  //!
  //! \param b The begin of the input vertex IDs sequence.
  //! \param e The end of the input vertex IDs sequence.
  //! \param o The start of the output vertex IDs sequence.
  template <typename Itr, typename OutputItr>
  void transformID(Itr b, Itr e, OutputItr o) const {
    using value_type = typename Itr::value_type;
    std::transform(b, e, o, [this](const value_type &v) -> value_type {
      return transformID(v);
    });
  }

  vertex_type transformID(const vertex_type v) const {
    auto itr = idMap.find(v);
    if (itr != idMap.end())
      return itr->second;
    else
      throw "Bad node";
  }

  //! Dump the internal representation to a binary stream.
  //!
  //! \tparam FStream The type of the output stream
  //!
  //! \param FS The ouput file stream.
  template <typename FStream>
  void dump_binary(FStream &FS) const {
    uint64_t num_nodes = htole64(numNodes);
    uint64_t num_edges = htole64(numEdges);
    FS.write(reinterpret_cast<const char *>(&num_nodes), sizeof(uint64_t));
    FS.write(reinterpret_cast<const char *>(&num_edges), sizeof(uint64_t));

    sequence_of<VertexTy>::dump(FS, reverseMap.begin(), reverseMap.end());

    using relative_index =
        typename std::iterator_traits<edge_pointer_t>::difference_type;
    std::vector<relative_index> relIndex(numNodes + 1, 0);
    std::transform(index, index + numNodes + 1, relIndex.begin(),
                   [=](edge_pointer_t v) -> relative_index {
                     return std::distance(edges, v);
                   });
    sequence_of<relative_index>::dump(FS, relIndex.begin(), relIndex.end());
    sequence_of<edge_type>::dump(FS, edges, edges + numEdges);
  }

 private:
  static constexpr bool isForward =
      std::is_same<DirectionPolicy, ForwardDirection<VertexTy>>::value;
  using transposed_direction =
      typename std::conditional<isForward, BackwardDirection<VertexTy>,
                                ForwardDirection<VertexTy>>::type;
  using transposed_type = Graph<vertex_type, edge_type, transposed_direction,
                                allocator_t>;

  friend transposed_type;

 public:
  //! Get the transposed graph.
  //! \return the transposed graph.
  transposed_type get_transpose() const {
    using out_dest_type = typename transposed_type::edge_type;
    using out_dest_ptr_type = rebind_alloc_pointer<allocator_t, out_dest_type>;
    transposed_type G(graph_allocator);
    G.numEdges = numEdges;
    G.numNodes = numNodes;
    G.reverseMap = reverseMap;
    G.idMap = idMap;
    G.index = G.allocate_index(G.numNodes + 1);
    G.edges = G.allocate_edges(G.numEdges);

    // Initialize with non-null pointers because the increment (++) operation is
    // performed to the values in G.index.
    // Incrementing the null pointer does not work with Boost::offset_ptr and
    // may be an undefined behavior even with the raw pointer.
#pragma omp parallel for
    for (auto itr = G.index; itr < G.index + numNodes + 1; ++itr) {
      *itr = G.edges;
    }

#pragma omp parallel for
    for (auto itr = G.edges; itr < G.edges + numEdges; ++itr) {
      *itr = out_dest_type();
    }

    std::for_each(edges, edges + numEdges,
                  [&](const edge_type &d) { ++G.index[d.vertex + 1]; });

    std::partial_sum(G.index, G.index + numNodes + 1, G.index,
                     [&G](out_dest_ptr_type a, out_dest_ptr_type b) {
                      const auto degree = std::distance(G.edges, b);
                      return a + degree;
                     });

    std::vector<out_dest_ptr_type> destPointers(G.index, G.index + numNodes);
    for (vertex_type v = 0; v < numNodes; ++v) {
      for (auto u : neighbors(v)) {
        *destPointers[u.vertex] = {v, u.weight};
        destPointers[u.vertex]++;
      }
    }

    return G;
  }

  edge_pointer_t *csr_index() const { return index; }

  edge_pointer_t csr_edges() const { return edges; }

  template <typename FStream>
  void load_binary(FStream &FS) {
    #ifdef ENABLE_METALL
    // Static assert 0
    throw 0 && "Not implemented yet, don't use with Metall";
    #endif


    if (!FS.is_open()) throw "Bad things happened!!!";

    FS.read(reinterpret_cast<char *>(&numNodes), sizeof(numNodes));
    FS.read(reinterpret_cast<char *>(&numEdges), sizeof(numEdges));

    numNodes = le64toh(numNodes);
    numEdges = le64toh(numEdges);

    reverseMap.resize(numNodes);
    FS.read(reinterpret_cast<char *>(reverseMap.data()),
            reverseMap.size() * sizeof(VertexTy));

    sequence_of<VertexTy>::load(reverseMap.begin(), reverseMap.end(),
                                reverseMap.begin());

    for (VertexTy i = 0; i < numNodes; ++i) idMap[reverseMap[i]] = i;

    index = allocate_index(numNodes + 1);
    edges = allocate_edges(numEdges);

    #pragma omp parallel for
    for (size_t i = 0; i < numNodes + 1; ++i) {
      index[i] = nullptr;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numEdges; ++i) {
      edges[i] = edge_type();
    }

    FS.read(reinterpret_cast<char *>(pointer_to(index)),
            (numNodes + 1) * sizeof(ptrdiff_t));

    sequence_of<edge_pointer_t>::load(index, index + numNodes + 1, index);

    std::transform(index, index + numNodes + 1, index,
                   [=](edge_pointer_t v) -> edge_pointer_t {
                     return edge_pointer_t(reinterpret_cast<ptrdiff_t>(pointer_to(v)) + edges);
                   });

    FS.read(reinterpret_cast<char *>(pointer_to(edges)), numEdges * sizeof(edge_type));
    sequence_of<edge_type>::load(edges, edges + numEdges, edges);
  }

  private:

  // Obtains a raw pointer from a given pointer. This function is equivalent to
  // std::pointer_to, which is available in C++20.
  template <typename pointer_t>
  static typename std::pointer_traits<pointer_t>::element_type*
    pointer_to(pointer_t p) {
  #ifdef ENABLE_METALL
    return metall::to_raw_pointer(p);
  #else
    return p;
  #endif
  }

  template <typename Alloc, typename T>
  static auto general_allocate(Alloc alloc, const size_t size) {
    using alloc_t = rebind_alloc<Alloc, T>;
    auto ptr = alloc_t(alloc).allocate(size);
    if (size > 0 && !ptr) {
      throw "Bad allocation";
    }
    return ptr;
  }

  template <typename Alloc, typename pointer_t>
  void general_deallocate(Alloc alloc, pointer_t ptr, const size_t size) {
    if (!ptr) return;
    using T = typename std::pointer_traits<pointer_t>::element_type;
    using alloc_t = rebind_alloc<Alloc, T>;
    alloc_t(alloc).deallocate(pointer_to(ptr), size);
  }

  index_pointer_t allocate_index(const std::size_t n) {
    return general_allocate<allocator_t, edge_pointer_t>(graph_allocator, n);
  }

  edge_pointer_t allocate_edges(const std::size_t n) {
    return general_allocate<allocator_t, edge_type>(graph_allocator, n);
  }

  void deallocate_index(index_pointer_t index, const std::size_t n) {
    general_deallocate<allocator_t, index_pointer_t>(graph_allocator, index, n);
  }

  void deallocate_edges(edge_pointer_t edges, const std::size_t n) {
    general_deallocate<allocator_t, edge_pointer_t>(graph_allocator, edges, n);
  }

  index_pointer_t index;
  edge_pointer_t edges;
  allocator_t graph_allocator;

    // Allocator and vector types for the indices array
  using reverse_map_allocator_t = rebind_alloc<allocator_t, VertexTy>;
  #if defined ENABLE_METALL
  using reverse_map_vector_t
    = metall::container::vector<VertexTy, reverse_map_allocator_t>;
  #else
  using reverse_map_vector_t = std::vector<VertexTy, reverse_map_allocator_t>;
  #endif

 using idmap_allocator_t = rebind_alloc<allocator_t,
                                        std::pair<const VertexTy, VertexTy>>;
 #if defined ENABLE_METALL
 using idmap_t = metall::container::unordered_map<VertexTy, VertexTy,
                                                  std::hash<VertexTy>,
                                                  std::equal_to<>,
                                                  idmap_allocator_t>;
 #else
 using idmap_t = std::unordered_map<VertexTy, VertexTy,
                                    std::hash<VertexTy>,
                                    std::equal_to<>,
                                    idmap_allocator_t>;
 #endif

  idmap_t idMap;
  reverse_map_vector_t reverseMap;

  size_t numNodes;
  size_t numEdges;
};

template <typename BwdGraphTy, typename FwdGraphTy>
auto getCommunitiesSubgraphs(
    const FwdGraphTy &Gf,
    const std::vector<typename FwdGraphTy::vertex_type> &communityVector) {
  using vertex_type = typename FwdGraphTy::vertex_type;
  size_t num_communities =
      *std::max_element(communityVector.begin(), communityVector.end()) + 1;
  std::vector<BwdGraphTy> communities(num_communities);

  using EdgeTy = Edge<typename FwdGraphTy::vertex_type, typename FwdGraphTy::edge_type::edge_weight>;
  std::vector<std::vector<EdgeTy>> edge_lists(num_communities);
  for (typename FwdGraphTy::vertex_type src = 0; src < Gf.num_nodes(); ++src) {
    vertex_type original_src = Gf.convertID(src);

    vertex_type community_src = communityVector[original_src - 1];
    for (auto e : Gf.neighbors(src)) {
      vertex_type original_dst = Gf.convertID(e.vertex);

      vertex_type community_dst = communityVector[original_dst - 1];
      if (community_dst == community_src) {
        edge_lists[community_src].push_back(
            {original_src, original_dst, e.weight});
      }
    }
  }

  for (size_t i = 0; i < num_communities; ++i) {
    communities[i] = BwdGraphTy(edge_lists[i].begin(), edge_lists[i].end(), true);
  }

  return communities;
}

}  // namespace ripples

#endif  // RIPPLES_GRAPH_H
