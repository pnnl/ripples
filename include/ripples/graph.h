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

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <ios>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <numeric>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

#include <ripples/utility.h>

#if defined ENABLE_METALL
#include <metall/metall.hpp>
#include <metall/container/vector.hpp>
#include <metall/container/unordered_map.hpp>
#endif

enum class WeightTypeEnum : uint64_t { FLOAT = 32UL, UINT16 = 16UL, UINT8 = 8UL };

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
  using weight_type = void;
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
  using vertex_type = VertexTy;
  using weight_type = WeightTy;

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

//! Iterator for the edges.
template <typename EdgePointerTy, typename WeightPointerTy, typename WeightedDestinationTy>
class WeightedEdgeIterator {
  using edge_pointer_t = EdgePointerTy;
  using weight_pointer_t = WeightPointerTy;
  using weighted_destination_t = WeightedDestinationTy;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = weighted_destination_t;
  using difference_type = std::ptrdiff_t;
  using pointer = weighted_destination_t *;
  using reference = weighted_destination_t &;

  WeightedEdgeIterator(size_t I, edge_pointer_t E, weight_pointer_t W)
      : index(I), edges(E), weights(W) {}

  WeightedEdgeIterator(const WeightedEdgeIterator &O)
      : index(O.index), edges(O.edges), weights(O.weights) {}

  WeightedEdgeIterator &operator=(const WeightedEdgeIterator &O) {
    index = O.index;
    edges = O.edges;
    weights = O.weights;
    return *this;
  }

  WeightedEdgeIterator &operator++() {
    ++index;
    return *this;
  }

  WeightedEdgeIterator operator++(int) {
    WeightedEdgeIterator tmp(*this);
    operator++();
    return tmp;
  }

  WeightedEdgeIterator &operator--() {
    --index;
    return *this;
  }

  WeightedEdgeIterator operator--(int) {
    WeightedEdgeIterator tmp(*this);
    operator--();
    return tmp;
  }

  WeightedEdgeIterator &operator+=(difference_type n) {
    index += n;
    return *this;
  }

  WeightedEdgeIterator &operator-=(difference_type n) {
    index -= n;
    return *this;
  }

  WeightedEdgeIterator operator+(difference_type n) const {
    WeightedEdgeIterator tmp(*this);
    return tmp += n;
  }

  WeightedEdgeIterator operator-(difference_type n) const {
    WeightedEdgeIterator tmp(*this);
    return tmp -= n;
  }

  difference_type operator+(const WeightedEdgeIterator &O) const {
    if (O.edges != this->edges || O.weights != this->weights) throw std::bad_exception();
    return index + O.index;
  }

  difference_type operator-(const WeightedEdgeIterator &O) const {
    if (O.edges != this->edges || O.weights != this->weights) throw std::bad_exception();
    return index - O.index;
  }

  bool operator==(const WeightedEdgeIterator &O) const {
    if (O.edges != this->edges || O.weights != this->weights) throw std::bad_exception();
    return index == O.index;
  }

  bool operator!=(const WeightedEdgeIterator &O) const {
    return index != O.index;
  }

  bool operator<(const WeightedEdgeIterator &O) const { return index < O.index; }

  bool operator>(const WeightedEdgeIterator &O) const { return index > O.index; }

  bool operator<=(const WeightedEdgeIterator &O) const {
    if (O.edges != this->edges || O.weights != this->weights) throw std::bad_exception();
    return index <= O.index;
  }

  bool operator>=(const WeightedEdgeIterator &O) const {
    return index >= O.index;
  }

  weighted_destination_t operator[](difference_type n) const {
    return weighted_destination_t(edges[index + n], weights[index + n]);
  }

  const weighted_destination_t operator*() const {
    return weighted_destination_t(edges[index], weights[index]);
  }

  const weighted_destination_t *operator->() const {
    return &weighted_destination_t(edges[index], weights[index]);
  }

  private:
  size_t index;
  edge_pointer_t edges;
  weight_pointer_t weights;
};

//! \brief The neighborhood of a vertex.
template<typename EdgePointerTy, typename WeightPointerTy, typename WeightedDestinationTy>
class Neighborhood {
 public:
  using iterator_type = WeightedEdgeIterator<EdgePointerTy, WeightPointerTy,
                                             WeightedDestinationTy>;
  //! Construct the neighborhood.
  //!
  //! \param B The begin of the neighbor list.
  //! \param E The end of the neighbor list.
  Neighborhood(size_t B, size_t E, EdgePointerTy edges, WeightPointerTy weights)
    : begin_(B, edges, weights),
      end_(E, edges, weights) {}

  //! Begin of the neighborhood.
  //! \return an iterator to the begin of the neighborhood.
  iterator_type begin() const { return begin_; }
  //! End of the neighborhood.
  //! \return an iterator to the begin of the neighborhood.
  iterator_type end() const { return end_; }

 private:
  iterator_type begin_;
  iterator_type end_;
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
  using weight_type = typename DestinationTy::weight_type;
  using index_type = size_t;

 public:
  // Pointer type for the edges array
  using edge_pointer_t = rebind_alloc_pointer<allocator_t, vertex_type>;
  // Pointer type for the indices array
  using index_pointer_t = rebind_alloc_pointer<allocator_t, index_type>;
  using weight_pointer_t = rebind_alloc_pointer<allocator_t, weight_type>;

  using neighborhood_range = Neighborhood<edge_pointer_t, weight_pointer_t, edge_type>;

  //! Allocator Graph Constructor.
  Graph(allocator_t allocator = allocator_t())
      : numNodes(0),
        numEdges(0),
        index(nullptr),
        edges(nullptr),
        weights(nullptr),
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
    weights = allocate_weights(numEdges);
    index = allocate_index(numNodes + 1);

#pragma omp parallel for
    for (size_t i = 0; i < numEdges; ++i) {
      edges[i] = O.edges[i];
      weights[i] = O.weights[i];
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
    deallocate_weights(weights, numEdges);

    index = allocate_index(numNodes + 1);
    edges = allocate_edges(numEdges);
    weights = allocate_weights(numEdges);
#pragma omp parallel for
    for (size_t i = 0; i < numEdges; ++i) {
      edges[i] = O.edges[i];
      weights[i] = O.weights[i];
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
        weights(O.weights),
        graph_allocator(std::move(O.graph_allocator)),
        idMap(std::move(O.idMap)),
        reverseMap(std::move(O.reverseMap)) {
    O.numNodes = 0;
    O.numEdges = 0;
    O.index = nullptr;
    O.edges = nullptr;
    O.weights = nullptr;
  }

  //! Move assignment operator.
  //! \param O The graph to be moved.
  //! \return a reference to the destination graph.
  Graph &operator=(Graph &&O) {
    if (this == &O) return *this;

    deallocate_index(index, numNodes + 1);
    deallocate_edges(edges, numEdges);
    deallocate_weights(weights, numEdges);

    numNodes = O.numNodes;
    numEdges = O.numEdges;
    index = O.index;
    edges = O.edges;
    weights = O.weights;
    idMap = std::move(O.idMap);
    reverseMap = std::move(O.reverseMap);

    O.numNodes = 0;
    O.numEdges = 0;
    O.index = nullptr;
    O.edges = nullptr;
    O.weights = nullptr;

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


    VertexTy maxVertexID = 0;
    omp_lock_t mapLock;
    omp_init_lock(&mapLock);
    #pragma omp parallel for reduction(max : maxVertexID)
    for (auto itr = begin; itr != end; ++itr) {
      if (idMap.count(itr->source) == 0) {
        omp_set_lock(&mapLock);
        if (idMap.count(itr->source) == 0) {
          idMap[itr->source] = itr->source;
          if (renumbering) {
            reverseMap.push_back(itr->source);
          }
        }
        omp_unset_lock(&mapLock);
      }

      if (idMap.count(itr->destination) == 0) {
        omp_set_lock(&mapLock);
        if (idMap.count(itr->destination) == 0) {
          idMap[itr->destination] = itr->destination;
          if (renumbering) {
            reverseMap.push_back(itr->destination);
          }
        }
        omp_unset_lock(&mapLock);
      }

      maxVertexID = std::max(std::max(itr->source, itr->destination), maxVertexID);
    }


    if (renumbering) {
      // Could utilize the C++ 17 parallel sort
      std::sort(reverseMap.begin(), reverseMap.end());
      #pragma omp parallel for
      for (size_t i = 0; i < reverseMap.size(); ++i) {
        assert(idMap.count(reverseMap.at(i)) > 0);
        idMap.at(reverseMap.at(i)) = i;
      }
      assert(idMap.size() == reverseMap.size());
    } else {
      reverseMap.resize(maxVertexID + 1);
      #pragma omp parallel for
      for (VertexTy id = 0; id <= maxVertexID; ++id) {
        reverseMap.at(id) = id;
      }
    }

    numNodes = reverseMap.size();
    numEdges = std::distance(begin, end);

    edges = allocate_edges(numEdges);
    weights = allocate_weights(numEdges);
    index = allocate_index(numNodes + 1);


#pragma omp parallel for
    for (size_t i = 0; i < numNodes + 1; ++i) {
      index[i] = 0;
    }

    #pragma omp parallel for
    for (auto itr = begin; itr != end; ++itr) {
      #pragma omp atomic
      index[DirectionPolicy::Source(itr, idMap) + 1] += 1;
    }


    for (size_t i = 1; i <= numNodes; ++i) {
      index[i] += index[i - 1];
    }

    std::vector<omp_lock_t> ptrLock(numNodes);
    std::vector<index_type> ptrEdge(numNodes);
#pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
      ptrEdge[i] = index[i];
      omp_init_lock(&ptrLock[i]);
    }
#pragma omp parallel for
    for (auto itr = begin; itr != end; ++itr) {
      omp_set_lock(&ptrLock[DirectionPolicy::Source(itr, idMap)]);

      auto e = edge_type::template Create<DirectionPolicy>(itr, idMap);
      edges[DirectionPolicy::Source(itr, idMap)] = e.vertex;
      weights[DirectionPolicy::Source(itr, idMap)] = e.weight;

      ++ptrEdge[DirectionPolicy::Source(itr, idMap)];

      omp_unset_lock(&ptrLock[DirectionPolicy::Source(itr, idMap)]);
    }
  }

  //! \brief Destuctor.
  ~Graph() {
    deallocate_index(index, numNodes + 1);
    deallocate_edges(edges, numEdges);
    deallocate_weights(weights, numEdges);
  }

  //! Returns the out-degree of a vertex.
  //! \param v The input vertex.
  //! \return the in-degree of vertex v in input.
  size_t degree(VertexTy v) const { return index[v + 1] - index[v]; }

  //! Returns the neighborhood of a vertex.
  //! \param v The input vertex.
  //! \return  a range containing the out-neighbors of the vertex v in input.
  neighborhood_range neighbors(VertexTy v) const {
    return neighborhood_range(index[v], index[v + 1], edges, weights);
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

private:
  size_t total_binary_size() const {
    return 3 * sizeof(uint64_t)
      + sizeof(VertexTy) * numNodes
      + sizeof(index_type) * (numNodes + 1)
      + sizeof(vertex_type) * numEdges
      + sizeof(weight_type) * numEdges;
  }
  void write_chunk(std::ofstream &FS, size_t TotalBytes, char* O) const {
    size_t threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
    size_t low = TotalBytes * threadnum / numthreads,
      high = TotalBytes * (threadnum + 1) / numthreads;
    size_t bytesToRead = high - low;
    O += low;
    FS.seekp(low, std::ios_base::cur);
    FS.write(O, bytesToRead);
    FS.seekp(TotalBytes - high, std::ios_base::cur);
  }

public:
  //! Dump the internal representation to a binary stream.
  //!
  //! \param FilePath The path to the ouput file.
  void dump_binary(const std::string &FilePath) const {
    ssize_t totalFileSize = total_binary_size();
    int mode = S_IRUSR | S_IWUSR | S_IRGRP;
    int file = open(FilePath.c_str(), O_CREAT | O_WRONLY, mode);
#if defined(__gnu_linux__)
    if (fallocate(file, 0, 0, totalFileSize) != 0) {
      // Test posix_fallocate
      std::cout << "Preallocation of " << totalFileSize  << " bytes failed with fallocate. Trying posix_fallocate..." << std::endl;
      if(posix_fallocate(file, 0, totalFileSize) != 0) {
        // Print error message
        std::cout << "Preallocation of " << totalFileSize  << " bytes failed with posix_fallocate. Is the disk too full?" << std::endl;
        close(file);
        exit(-1);
      }
      else{
        std::cout << "Preallocation of " << totalFileSize  << " bytes succeeded with posix_fallocate!" << std::endl;
      }
    }
#else
    fstore_t store = {F_ALLOCATEALL, F_PEOFPOSMODE, 0, totalFileSize};
    fcntl(file, F_PREALLOCATE, &store);

    if (store.fst_bytesalloc < totalFileSize) {
      std::cout << "Preallocation of " << totalFileSize  << " bytes failed with fcntl. Is the disk too full?" << std::endl;
      close(file);
      exit(-1);
    }

    ftruncate(file, totalFileSize);
#endif

    // TODO: fix for 64bit vertices IDs.
    #pragma omp parallel
    {
      std::ofstream FS(FilePath, std::ios::out | std::ios::binary);

      #pragma omp single
      {
        uint64_t endianess_check = 0xc0ffee;
        uint64_t direction_policy = std::is_same<ForwardDirection<VertexTy>, DirectionPolicy>::value ?
          0xf0cacc1a : 0xa1ccac0f;
        uint64_t weight_enum = std::is_same<float, weight_type>::value ?
          static_cast<uint64_t>(WeightTypeEnum::FLOAT) :
          std::is_same<uint16_t, weight_type>::value ?
          static_cast<uint64_t>(WeightTypeEnum::UINT16) :
          static_cast<uint64_t>(WeightTypeEnum::UINT8);
        FS.write(reinterpret_cast<const char *>(&endianess_check), sizeof(uint64_t));
        FS.write(reinterpret_cast<const char *>(&direction_policy), sizeof(uint64_t));
        FS.write(reinterpret_cast<const char *>(&weight_enum), sizeof(uint64_t));
        FS.write(reinterpret_cast<const char *>(&numNodes), sizeof(uint64_t));
        FS.write(reinterpret_cast<const char *>(&numEdges), sizeof(uint64_t));
      }

      constexpr size_t num_metadata_elements = 3;

      FS.seekp(num_metadata_elements * sizeof(uint64_t) + sizeof(numNodes) + sizeof(numEdges), std::ios_base::beg);

      write_chunk(FS, numNodes * sizeof(VertexTy),
                  const_cast<char *>(
                      reinterpret_cast<const char *>(reverseMap.data())));

      write_chunk(FS, (numNodes + 1)  * sizeof(index_type),
                  reinterpret_cast<char *>(pointer_to(index)));

      write_chunk(FS, numEdges * sizeof(vertex_type),
                  reinterpret_cast<char *>(pointer_to(edges)));

      write_chunk(FS, numEdges * sizeof(weight_type),
                  reinterpret_cast<char *>(pointer_to(weights)));
    }
    close(file);
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
    G.weights = G.allocate_weights(G.numEdges);

#pragma omp parallel for
    for (auto itr = G.index; itr < G.index + numNodes + 1; ++itr) {
      *itr = 0;
    }

    std::for_each(edges, edges + numEdges,
                  [&](const vertex_type &d) { ++G.index[d + 1]; });

    std::partial_sum(G.index, G.index + numNodes + 1, G.index,
                     std::plus<index_type>());

    std::vector<index_type> destPointers(numNodes + 1);
#pragma omp parallel for
    for (index_type i = 0; i < destPointers.size(); ++i) {
      destPointers[i] = index[i];
    }
    for (vertex_type v = 0; v < numNodes; ++v) {
      for (auto u : neighbors(v)) {
        edges[destPointers[u.vertex]] = v;
        weights[destPointers[u.vertex]] = u.weight;
        destPointers[u.vertex]++;
      }
    }

    return G;
  }

  index_pointer_t csr_index() const { return index; }

  edge_pointer_t csr_edges() const { return edges; }

  weight_pointer_t csr_weights() const { return weights; }

private:
  template<typename FStreamTy>
  void read_chunk(FStreamTy &FS, size_t TotalBytes, char* O) {
    size_t threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
    size_t low = TotalBytes * threadnum / numthreads,
      high = TotalBytes * (threadnum + 1) / numthreads;
    size_t bytesToRead = high - low;
    O += low;
    FS.seekg(low, std::ios_base::cur);
    FS.read(O, bytesToRead);
    FS.seekg(TotalBytes - high, std::ios_base::cur);
  }

public:
  void load_binary(const std::string & FileName) {
    #ifdef ENABLE_METALL
    // Static assert 0
    throw 0 && "Not implemented yet, don't use with Metall";
    #endif

    {
      std::ifstream FS(FileName, std::ios::binary);
      if (!FS.is_open()) throw "Bad things happened!!!";
      uint64_t endianess_check;
      FS.read(reinterpret_cast<char *>(&endianess_check), sizeof(uint64_t));

      if (endianess_check != 0xc0ffee) {
        std::cout <<
          "The endianess check failed when reloading the input binary.\nLikely,"
          "the input file was generated on a different architecture.\nPlease,"
          "used the dump-graph tool to produce a new binary."
                  << std::endl;

        FS.close();
        exit(-1);
      }
      uint64_t direction_check;
      FS.read(reinterpret_cast<char *>(&direction_check), sizeof(uint64_t));

      if (std::is_same<ForwardDirection<VertexTy>, DirectionPolicy>::value &&
          direction_check != 0xf0cacc1a) {
        std::cout <<
          "You are loading a binary that is not compatible with the algorithm you are trying to run.\n"
          "Please, try to regenerate the binary passing --transpose."
                  << std::endl;
        FS.close();
        exit(-1);
      } else if (std::is_same<BackwardDirection<VertexTy>, DirectionPolicy>::value &&
          direction_check != 0xa1ccac0f) {
        std::cout
            << "You are loading a binary that is not compatible with the "
               "algorithm you are trying to run.\n"
               "Please, try to regenerate the binary passing --transpose."
            << std::endl;
        FS.close();
        exit(-1);
      }
      uint64_t weight_type_check;
      FS.read(reinterpret_cast<char *>(&weight_type_check), sizeof(uint64_t));
      const std::unordered_map<uint64_t, std::string> weight_type_map = {
        {static_cast<uint64_t>(WeightTypeEnum::FLOAT), "float"},
        {static_cast<uint64_t>(WeightTypeEnum::UINT16), "uint16"},
        {static_cast<uint64_t>(WeightTypeEnum::UINT8), "uint8"}};
      uint64_t weight_type_confirm;
      if constexpr(std::is_same<weight_type, float>::value) {
        weight_type_confirm = static_cast<uint64_t>(WeightTypeEnum::FLOAT);
      } else if constexpr(std::is_same<weight_type, uint16_t>::value) {
        weight_type_confirm = static_cast<uint64_t>(WeightTypeEnum::UINT16);
      } else if constexpr(std::is_same<weight_type, uint8_t>::value) {
        weight_type_confirm = static_cast<uint64_t>(WeightTypeEnum::UINT8);
      } else {
        std::cout << "The specified weight type for ripples not supported.\n"
                     "Please recompile to a supported weight type of float,"
                     "uint16, or uint8" << std::endl;
        FS.close();
        exit(-1);
      }
      if (weight_type_check != weight_type_confirm) {
        std::string weight_type_str;
        if (weight_type_map.count(weight_type_check)){
          weight_type_str = weight_type_map.at(weight_type_check);
        }
        else{
          // Convert weight_type_check to a string
          std::string weight_val  = std::to_string(weight_type_check);
          weight_type_str = "unknown (uint64_t val = " + weight_val + ")";
        }
        std::string weight_type_confirm_str = weight_type_map.at(weight_type_confirm);
        std::cout << "The weight type in the binary file is " << weight_type_str
                  << " but the weight type specified at compile time is "
                  << weight_type_confirm_str << ".\nPlease recompile ripples to"
                  << " the correct weight type or regenerate a new binary file."
                  << std::endl;
        FS.close();
        exit(-1);
      }
    }

#pragma omp parallel
    {
      std::ifstream FS(FileName, std::ios::binary);
      constexpr size_t num_metadata_elements = 3;
      #pragma omp single
      {
        FS.seekg(num_metadata_elements * sizeof(uint64_t), std::ios_base::beg);
        FS.read(reinterpret_cast<char *>(&numNodes), sizeof(numNodes));
        FS.read(reinterpret_cast<char *>(&numEdges), sizeof(numEdges));

        numNodes = le64toh(numNodes);
        numEdges = le64toh(numEdges);

        reverseMap.resize(numNodes);
      }

      FS.seekg(num_metadata_elements * sizeof(uint64_t) + sizeof(numNodes) +
                   sizeof(numEdges),
               std::ios_base::beg);
      read_chunk(FS, reverseMap.size() * sizeof(VertexTy),
                 reinterpret_cast<char *>(reverseMap.data()));

      #pragma omp single
      {
        index = allocate_index(numNodes + 1);
        edges = allocate_edges(numEdges);
        weights = allocate_weights(numEdges);
      }

      read_chunk(FS, (numNodes + 1) * sizeof(index_type),
                 reinterpret_cast<char *>(pointer_to(index)));

      read_chunk(FS, numEdges * sizeof(vertex_type), reinterpret_cast<char *>(pointer_to(edges)));
      read_chunk(FS, numEdges * sizeof(weight_type), reinterpret_cast<char *>(pointer_to(weights)));

      decltype(idMap) localMap;
      #pragma omp for
      for (VertexTy i = 0; i < numNodes; ++i) localMap[reverseMap[i]] = i;

      #pragma omp critical
      {
#if (__cplusplus >= 201703L)
        idMap.merge(localMap);
#else
        idMap.insert(localMap.begin(), localMap.end());
#endif
      }
    }
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
    return general_allocate<allocator_t, index_type>(graph_allocator, n);
  }

  edge_pointer_t allocate_edges(const std::size_t n) {
    return general_allocate<allocator_t, vertex_type>(graph_allocator, n);
  }

  weight_pointer_t allocate_weights(const std::size_t n) {
    return general_allocate<allocator_t, weight_type>(graph_allocator, n);
  }

  void deallocate_index(index_pointer_t index, const std::size_t n) {
    general_deallocate<allocator_t, index_pointer_t>(graph_allocator, index, n);
  }

  void deallocate_edges(edge_pointer_t edges, const std::size_t n) {
    general_deallocate<allocator_t, edge_pointer_t>(graph_allocator, edges, n);
  }

  void deallocate_weights(weight_pointer_t edges, const std::size_t n) {
    general_deallocate<allocator_t, weight_pointer_t>(graph_allocator, weights, n);
  }

  index_pointer_t index;
  edge_pointer_t edges;
  weight_pointer_t weights;
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

  using EdgeTy = Edge<typename FwdGraphTy::vertex_type, typename FwdGraphTy::edge_type::weight_type>;
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
