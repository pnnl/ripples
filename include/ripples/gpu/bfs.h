//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_GPU_HIP_BFS_H
#define RIPPLES_GPU_HIP_BFS_H

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"
#include "thrust/count.h"
#include "thrust/device_vector.h"
#include "thrust/for_each.h"
#include "thrust/host_vector.h"
#include "thrust/iterator/constant_iterator.h"
#include "thrust/inner_product.h"
#include "thrust/partition.h"
#include "thrust/random.h"
#include "thrust/reduce.h"
#include "thrust/sort.h"
#include "thrust/transform_scan.h"

#if defined(RIPPLES_ENABLE_CUDA)
#define RUNTIME CUDA
#define MAX_COLOR_WIDTH 32
#elif defined(RIPPLES_ENABLE_HIP)
#define RUNTIME HIP
#define MAX_COLOR_WIDTH 64
#endif

#include "ripples/gpu/bfs_kernels.h"

#include <cassert>

#define EXPERIMENTAL_SCAN_BFS

#define HIERARCHICAL

#define REORDERING

#define SORTING

#define FUSED_COLOR_SET

// #define PAUSE_AND_RESUME

// #define PROFILE_OVERHEAD

// #define FULL_COLORS_MOTIVATION

// #define FRONTIER_PROFILE

// #define UTILIZATION_PROFILE

#ifdef FRONTIER_PROFILE
#include <chrono>
#include <iostream>
#include <fstream>
// Create struct storing the frontier size, time, and number of colors
struct FrontierProfile {
  size_t frontier_size;
  long frontier_time;
  size_t frontier_colors;
  size_t old_frontier_size;
  long scatter_time;
  size_t max_outdegree;
  size_t iteration;
  size_t edge_colors;
  size_t unique_colors;
};
std::vector<FrontierProfile> profile_vector;
#endif


#endif

namespace ripples {

//! \brief A structure of arrays storing the BFS frontier.
//!
//! The frontier stores three piece of information:
//!  - The vertex (v)
//!  - The color representing the single BFS
//!  - The edge weight with which we reached v;
template <typename GraphTy, typename ColorTy = typename GraphTy::vertex_type>
struct Frontier {
  thrust::device_vector<typename GraphTy::vertex_type> v;
  thrust::device_vector<ColorTy> color;
  thrust::device_vector<typename GraphTy::weight_type> weight;
};

//! \brief A structure of arrays storing the Hierarchical BFS frontier.
//!
//! The frontier stores three piece of information:
//!  - The vertex (v)
//!  - The color representing the single BFS
//!  - The edge weight with which we reached v;
template <typename GraphTy, typename ColorTy = typename GraphTy::vertex_type>
struct FrontierHierarch {
  FrontierHierarch(size_t num_nodes) : v(num_nodes), color(num_nodes), offset(num_nodes) {}
  thrust::device_vector<typename GraphTy::vertex_type> v;
  thrust::device_vector<ColorTy> color;
  thrust::device_vector<typename GraphTy::vertex_type> offset;
};

template <typename GraphTy, typename ColorTy = typename GraphTy::vertex_type>
struct BFSContext{
  BFSContext(size_t num_nodes = 0, size_t small = 0, size_t medium = 0, size_t large = 0, size_t extreme = 0, size_t gpu_id = 0) :
    small_frontier(small), medium_frontier(medium), large_frontier(large), extreme_frontier(extreme),
    frontier_matrix(num_nodes), visited_matrix(num_nodes), host_visited_matrix(num_nodes) {
      GPU<RUNTIME>::set_device(gpu_id);
      streams = std::vector<typename GPU<RUNTIME>::stream_type>(4, GPU<RUNTIME>::create_stream());
    }
  Frontier<GraphTy, ColorTy> frontier, new_frontier;
  FrontierHierarch<GraphTy, ColorTy> small_frontier;
  FrontierHierarch<GraphTy, ColorTy> medium_frontier;
  FrontierHierarch<GraphTy, ColorTy> large_frontier;
  FrontierHierarch<GraphTy, ColorTy> extreme_frontier;
  thrust::device_vector<typename GraphTy::vertex_type> numNeighbors;
  thrust::device_vector<typename GraphTy::vertex_type> numNeighborsSeparate;
  thrust::device_vector<ColorTy> frontier_matrix;
  thrust::device_vector<ColorTy> visited_matrix;
  thrust::host_vector<ColorTy> host_visited_matrix;
  std::vector<typename GPU<RUNTIME>::stream_type> streams;
};

template <typename GraphTy, typename ColorTy = typename GraphTy::vertex_type, size_t num_sets = MAX_COLOR_WIDTH,
          typename OutputTy = std::vector<uint32_t>>
struct BFSMultiContext{
  BFSMultiContext(size_t num_nodes = 0, size_t small = 0, size_t medium = 0, size_t large = 0, size_t extreme = 0, size_t gpu_id = 0) :
    small_vertices(small), medium_vertices(medium + small), large_vertices(large + medium + small), extreme_vertices(extreme + large + medium + small),
    small_colors(small * num_sets), medium_colors((medium + small) * num_sets), large_colors((large + medium + small) * num_sets), extreme_colors((extreme + large + medium + small) * num_sets),
    frontier_matrix(num_nodes*num_sets), reduced_frontier(num_sets), visited_matrix(num_nodes*num_sets), host_visited_matrix(num_nodes*num_sets), workloads(4), host_workloads(4),
    host_reduced_frontier(num_sets), output_mappings(num_sets * sizeof(ColorTy) * 8) {
      GPU<RUNTIME>::set_device(gpu_id);
      streams = std::vector<typename GPU<RUNTIME>::stream_type>(4, GPU<RUNTIME>::create_stream());
    }
  using color_type = ColorTy;
  thrust::device_vector<typename GraphTy::vertex_type> small_vertices;
  thrust::device_vector<typename GraphTy::vertex_type> medium_vertices;
  thrust::device_vector<typename GraphTy::vertex_type> large_vertices;
  thrust::device_vector<typename GraphTy::vertex_type> extreme_vertices;
  thrust::device_vector<ColorTy> small_colors;
  thrust::device_vector<ColorTy> medium_colors;
  thrust::device_vector<ColorTy> large_colors;
  thrust::device_vector<ColorTy> extreme_colors;
  thrust::device_vector<typename GraphTy::vertex_type> workloads;
  thrust::host_vector<typename GraphTy::vertex_type> host_workloads;
  thrust::device_vector<ColorTy> frontier_matrix;
  thrust::device_vector<ColorTy> reduced_frontier;
  thrust::device_vector<ColorTy> visited_matrix;
  thrust::host_vector<ColorTy> host_visited_matrix;
  thrust::host_vector<ColorTy> host_reduced_frontier;
  std::vector<typename GPU<RUNTIME>::stream_type> streams;
  std::vector<OutputTy> output_mappings;
  #ifdef PROFILE_OVERHEAD
  std::vector<size_t> setup_time;
  std::vector<size_t> traversal_time;
  std::vector<size_t> end_time;
  std::vector<size_t> num_colors;
  std::vector<size_t> remaining_count;
  std::vector<size_t> num_traversals;
  void print_to_file(std::string filename){
    std::ofstream file;
    file.open(filename);
    file << "Setup Time,Traversal Time,End Time,Num Colors,Remaining Count,Num Traversals" << std::endl;
    for(size_t i = 0; i < setup_time.size(); i++){
      file << setup_time[i] << "," << traversal_time[i] << "," << end_time[i] << "," << num_colors[i] << "," << remaining_count[i] << "," << num_traversals[i] << std::endl;
    }
    file.close();
  }
  #endif
  #ifdef UTILIZATION_PROFILE
  std::vector<size_t> small_queue;
  std::vector<size_t> medium_queue;
  std::vector<size_t> large_queue;
  std::vector<size_t> extreme_queue;
  std::vector<size_t> small_warps;
  std::vector<size_t> medium_warps;
  std::vector<size_t> large_warps;
  std::vector<size_t> extreme_warps;
  void print_utilization_to_file(std::string filename){
    std::ofstream file;
    file.open(filename);
    file << "Small Queue,Medium Queue,Large Queue,Extreme Queue,Small Warps,Medium Warps,Large Warps,Extreme Warps" << std::endl;
    for(size_t i = 0; i < small_queue.size(); i++){
      file << small_queue[i] << "," << medium_queue[i] << "," << large_queue[i] << "," << extreme_queue[i] << "," << small_warps[i] << "," << medium_warps[i] << "," << large_warps[i] << "," << extreme_warps[i] << std::endl;
    }
    file.close();
  }
  #endif // UTILIZATION_PROFILE
};

//! \brief Get the next color ID from the color mask.
template <typename T>
__host__ __device__ __inline__ T getNextColor(T color);

template <>
__host__ __device__ __inline__ uint32_t getNextColor(uint32_t color) {
  return __builtin_clz(color);
}

template <>
__host__ __device__ __inline__ uint64_t getNextColor(uint64_t color) {
  return __builtin_clzl(color);
}

template <typename T>
__host__ __device__ __inline__ T numColors(T color);

template <>
__host__ __device__ __inline__ uint32_t numColors(uint32_t color) {
  return __builtin_popcount(color);
}

template <>
__host__ __device__ __inline__ uint64_t numColors(uint64_t color) {
  return __builtin_popcountl(color);
}

//! \brief Get a color mask from a color ID.
template <typename T>
__host__ __device__ __inline__ T getMaskFromColor(T color) {
  return (T)1 << ((sizeof(T) * 8 - 1) - color);
}

//! \brief Remove a color from the mask of colors.
template <typename T>
__host__ __device__ __inline__ T clearColor(T colors, T color) {
  return colors & (~getMaskFromColor(color));
}

template <typename VertexTy, typename WeightTy, typename GraphTy>
struct GPUEdgeScatter{
  VertexTy d_index;
  VertexTy d_edge;
  WeightTy d_weight;
  template <typename Tuple>
  __device__ void operator()(Tuple &T) {
    auto vertex = thrust::get<0>(T);
    auto color = thrust::get<1>(T);
    auto startO = thrust::get<2>(T);
    auto O = thrust::get<3>(T);

    // std::cout << "Compute + Zip" << std::endl;

    auto startJ = d_edge + d_index[vertex];
    auto endJ = d_edge + d_index[vertex + 1];

    auto startW = d_weight + d_index[vertex];
    auto endW = d_weight + d_index[vertex + 1];

    auto adjB = thrust::make_zip_iterator(thrust::make_tuple(
        startJ, thrust::constant_iterator<typename GraphTy::vertex_type>(color), startW));
    auto adjE = thrust::make_zip_iterator(thrust::make_tuple(
        endJ, thrust::constant_iterator<typename GraphTy::vertex_type>(color), endW));

    // std::cout << "Copy" << std::endl;

    thrust::copy(thrust::device, adjB, adjE, O + startO);
  }
};

template <typename VertexTy>
struct GPUIndependentCascade {
  template <typename Tuple>
  __device__ void operator()(Tuple &T) {
    auto vertex = thrust::get<0>(T);
    auto colors = thrust::get<1>(T);
    auto weight = thrust::get<2>(T);
    auto seed = thrust::get<3>(T);

    thrust::minstd_rand generator(seed);
    thrust::uniform_real_distribution<float> value;
    decltype(colors) newColors = 0;

    while (colors != 0) {
      decltype(colors) color = getNextColor(colors);
      decltype(colors) mask = getMaskFromColor(color);

      if (!(visited_matrix[vertex] & mask) && value(generator) <= weight) {
        newColors |= mask;
      }

      colors = clearColor(colors, color);
    }
    if(newColors != 0){
      VertexTy *addr = visited_matrix + vertex;
      atomicOr(addr, newColors);
    }
    thrust::get<1>(T) = newColors;
  }

  VertexTy *visited_matrix;
};

template <typename GraphTy, typename DeviceContextTy, typename SItrTy,
          typename OItrTy, typename diff_model_tag>
void GPUBatchedBFS(GraphTy &G, const DeviceContextTy &Context, SItrTy B,
                   SItrTy E, OItrTy O, diff_model_tag &&tag) {
  using DeviceGraphTy = typename DeviceContextTy::device_graph_type;
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;

  constexpr unsigned int NumColors = 8 * sizeof(vertex_type);

  assert(std::distance(B, E) <= NumColors &&
         "Only up to sizeof(vertex_type) BFS are supported.");

  GPU<RUNTIME>::set_device(Context.gpu_id);

  thrust::device_vector<vertex_type> visited_matrix(G.num_nodes());
  thrust::host_vector<vertex_type> host_visited_matrix(G.num_nodes(), 0);

  // std::cout << "Setup Step" << std::endl;

  Frontier<GraphTy> frontier, new_frontier;
  uint32_t color = 1 << (NumColors - 1);
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    frontier.v.push_back(*itr);
    frontier.color.push_back(color);
    host_visited_matrix[*itr] |= color;
  }
  visited_matrix = host_visited_matrix;

  // std::cout << "Reduce Frontier" << std::endl;

  // Reduce frontier:
  // -1 sort the frontier by vertex
  // -2 count the unique vertices
  // -2 reduce the frontier by fusing colors
  thrust::sort_by_key(std::begin(frontier.v), std::end(frontier.v),
                      std::begin(frontier.color));
  size_t FrontierSize = thrust::inner_product(
      frontier.v.begin(), frontier.v.end() - 1, frontier.v.begin() + 1, 1,
      thrust::plus<int>(), thrust::not_equal_to<vertex_type>());
  new_frontier.v.resize(FrontierSize);
  new_frontier.color.resize(FrontierSize);
  thrust::reduce_by_key(frontier.v.begin(), frontier.v.end(),
                        frontier.color.begin(), new_frontier.v.begin(),
                        new_frontier.color.begin(), thrust::equal_to<vertex_type>(),
                        thrust::bit_or<vertex_type>());

  thrust::swap(frontier, new_frontier);

  thrust::device_vector<vertex_type> numNeighbors(frontier.v.size() + 1, 0);
  GPUIndependentCascade<vertex_type> simStep;
  simStep.visited_matrix = visited_matrix.data().get();

  // std::cout << "Process Frontier" << std::endl;

  #ifdef FRONTIER_PROFILE
  size_t iteration = 0;
  #endif

  while (frontier.v.size() != 0) {
    // std::cout << "Size = " << frontier.v.size() << std::endl;
    auto d_graph = Context.d_graph;
    auto d_index = d_graph->d_index_;
    auto d_edge = d_graph->d_edges_;
    auto d_weight = d_graph->d_weights_;
    // std::cout << "Inclusive Scan" << std::endl;
    thrust::transform_inclusive_scan(
        thrust::device, std::begin(frontier.v), std::end(frontier.v),
        std::begin(numNeighbors) + 1,
        [d_index](const vertex_type &FE) {
          return d_index[FE + 1] - d_index[FE];
        },
        thrust::plus<vertex_type>{});

    new_frontier.v.resize(numNeighbors.back());
    new_frontier.color.resize(numNeighbors.back() + 1);
    new_frontier.weight.resize(numNeighbors.back() + 1);

    #if 0
      Frontier<GraphTy> test_frontier;
      test_frontier.v.resize(numNeighbors.back());
      test_frontier.color.resize(numNeighbors.back() + 1);
      test_frontier.weight.resize(numNeighbors.back() + 1);
      {
      auto O = thrust::make_zip_iterator(
          thrust::make_tuple(test_frontier.v.begin(), test_frontier.color.begin(),
                            test_frontier.weight.begin()));
      auto B = thrust::make_zip_iterator(thrust::make_tuple(
          frontier.v.begin(), frontier.color.begin(), numNeighbors.begin(),
          thrust::constant_iterator<decltype(O)>(O),
          thrust::constant_iterator<decltype(d_graph)>(d_graph)));
      auto E = thrust::make_zip_iterator(thrust::make_tuple(
          frontier.v.end(), frontier.color.end(), numNeighbors.end() - 1,
          thrust::constant_iterator<decltype(O)>(O),
          thrust::constant_iterator<decltype(d_graph)>(d_graph)));
      // std::cout << "Process Each Edge" << std::endl;
      for (auto itr = B; itr < E; ++itr) {
        const auto &T = *itr;
        auto vertex = thrust::get<0>(T);
        auto color = thrust::get<1>(T);
        auto startO = thrust::get<2>(T);

        // std::cout << "Compute + Zip" << std::endl;

        auto startJ = d_graph->d_edges_ + d_graph->d_index_[vertex];
        auto endJ = d_graph->d_edges_ + d_graph->d_index_[vertex + 1];

        auto startW = d_graph->d_weights_ + d_graph->d_index_[vertex];
        auto endW = d_graph->d_weights_ + d_graph->d_index_[vertex + 1];

        auto adjB = thrust::make_zip_iterator(thrust::make_tuple(
            startJ, thrust::constant_iterator<vertex_type>(color), startW));
        auto adjE = thrust::make_zip_iterator(thrust::make_tuple(
            endJ, thrust::constant_iterator<vertex_type>(color), endW));

        // std::cout << "Copy" << std::endl;

        thrust::copy(thrust::device, adjB, adjE, O + startO);
      }
      }
    #else
    auto O = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.begin(), new_frontier.color.begin(),
                           new_frontier.weight.begin()));
    auto B = thrust::make_zip_iterator(thrust::make_tuple(
        frontier.v.begin(), frontier.color.begin(), numNeighbors.begin(),
        thrust::constant_iterator<decltype(O)>(O)));
    auto E = thrust::make_zip_iterator(thrust::make_tuple(
        frontier.v.end(), frontier.color.end(), numNeighbors.end() - 1,
        thrust::constant_iterator<decltype(O)>(O)));
    GPUEdgeScatter<decltype(d_index), decltype(d_weight), GraphTy> scatEdge{d_index, d_edge, d_weight};
    thrust::for_each(thrust::device, B, E, scatEdge);
    #endif

    #if 0
    auto OE = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.end(), new_frontier.color.end(),
                           new_frontier.weight.end()));
    auto compare = thrust::make_zip_iterator(
        thrust::make_tuple(test_frontier.v.begin(), test_frontier.color.begin(),
                           test_frontier.weight.begin()));
    assert(thrust::equal(new_frontier.v.begin(), new_frontier.v.end(), test_frontier.v.begin()));
    std::cout << "Equal!" << std::endl;
    #endif

    #ifdef FRONTIER_PROFILE
    size_t edge_size = new_frontier.v.size();
    // Determine number of 1 bits in color
    size_t num_colors = thrust::transform_reduce(
        thrust::device, new_frontier.color.begin(), new_frontier.color.end(),
        [](const vertex_type &c) { return numColors(c); }, 0,
        thrust::plus<uint32_t>());
    GPU<RUNTIME>::device_sync();
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // auto seedItrB = thrust::counting_iterator<uint64_t>(clock());
    auto seedItrB = thrust::counting_iterator<uint64_t>(0);
    auto seedItrE = seedItrB + thrust::distance(new_frontier.v.begin(),
                                                new_frontier.v.end());
    auto edgeB = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.begin(), new_frontier.color.begin(),
                           new_frontier.weight.begin(), seedItrB));
    auto edgeE = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.end(), new_frontier.color.end(),
                           new_frontier.weight.end(), seedItrE));
    // std::cout << "Sim Step" << std::endl;
    thrust::for_each(thrust::device, edgeB, edgeE, simStep);

    if (new_frontier.v.size() == 0) break;

    // std::cout << "Recombine" << std::endl;

    // - Recombine vertices that might be appearing multiple times.
    thrust::sort_by_key(thrust::device, std::begin(new_frontier.v),
                        std::end(new_frontier.v),
                        std::begin(new_frontier.color));
    size_t FrontierSize = thrust::inner_product(
        new_frontier.v.begin(), new_frontier.v.end() - 1,
        new_frontier.v.begin() + 1, 1, thrust::plus<int>(),
        thrust::not_equal_to<vertex_type>());

    frontier.v.resize(FrontierSize);
    frontier.color.resize(FrontierSize);
    thrust::reduce_by_key(new_frontier.v.begin(), new_frontier.v.end(),
                          new_frontier.color.begin(), frontier.v.begin(),
                          frontier.color.begin(), thrust::equal_to<vertex_type>(),
                          thrust::bit_or<vertex_type>());

    // std::cout << "Remove Empty" << std::endl;
    // - Remove vertices that might have empty label.
    auto cleanUpB = thrust::make_zip_iterator(
        thrust::make_tuple(frontier.v.begin(), frontier.color.begin()));
    auto itr =
        thrust::partition(thrust::device, cleanUpB,
                          thrust::make_zip_iterator(thrust::make_tuple(
                              frontier.v.end(), frontier.color.end())),
                          [](const auto &T) { return thrust::get<1>(T) != 0; });
    frontier.v.resize(thrust::distance(cleanUpB, itr));
    frontier.color.resize(thrust::distance(cleanUpB, itr));
    frontier.weight.resize(thrust::distance(cleanUpB, itr));

    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    iteration++;
    profile_vector.push_back({edge_size, time.count(), num_colors});
    #endif

    numNeighbors.resize(frontier.v.size() + 1);
  }

  // std::cout << "Clean Frontier" << std::endl;

  host_visited_matrix = visited_matrix;

  for (vertex_type v = 0; v < host_visited_matrix.size(); ++v) {
    if (host_visited_matrix[v] == 0) continue;

    vertex_type colors = host_visited_matrix[v];

    while (colors != 0) {
      vertex_type color = getNextColor(colors);

      (O + color)->push_back(v);

      colors = clearColor(colors, color);
    }
  }
}

template <typename ColorTy>
struct GPUIndependentCascadeScan {
  template <typename Tuple>
  __device__ void operator()(Tuple &T) {
    auto vertex = thrust::get<0>(T);
    auto colors = thrust::get<1>(T);
    auto weight = thrust::get<2>(T);
    auto seed = thrust::get<3>(T);

    thrust::minstd_rand generator(seed*seed+19283);
    thrust::uniform_real_distribution<float> value;
    decltype(colors) newColors = 0;

    while (colors != 0) {
      decltype(colors) color = getNextColor(colors);
      decltype(colors) mask = getMaskFromColor(color);

      if (!(visited_matrix[vertex] & mask) && value(generator) <= weight) {
        newColors |= mask;
      }
      colors = clearColor(colors, color);
    }
    if(newColors != 0){
      ColorTy *addr = visited_matrix + vertex;
      atomicOr(addr, newColors);
      ColorTy *frontier_addr = frontier_matrix + vertex;
      atomicOr(frontier_addr, newColors);
    }
  }

  ColorTy *visited_matrix;
  ColorTy *frontier_matrix;
};

template<typename VertexTy, typename ColorTy = VertexTy>
struct notZero
{
	__host__ __device__
    bool operator()(thrust::tuple<VertexTy, ColorTy> T)
	{
		ColorTy v = thrust::get<1>(T);
		return (v != 0);
	}
	
};

template <typename GraphTy, typename DeviceContextTy, typename SItrTy,
          typename OItrTy, typename diff_model_tag>
void GPUBatchedScanBFS(GraphTy &G, const DeviceContextTy &Context, SItrTy B,
                   SItrTy E, OItrTy O, diff_model_tag &&tag) {
  using DeviceGraphTy = typename DeviceContextTy::device_graph_type;
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;

  constexpr unsigned int NumColors = 8 * sizeof(vertex_type);

  assert(std::distance(B, E) <= NumColors &&
         "Only up to sizeof(vertex_type) BFS are supported.");

  GPU<RUNTIME>::set_device(Context.gpu_id);

  thrust::device_vector<vertex_type> frontier_matrix(G.num_nodes(), 0);
  thrust::device_vector<vertex_type> visited_matrix(G.num_nodes());
  thrust::host_vector<vertex_type> host_visited_matrix(G.num_nodes(), 0);

  // std::cout << "Setup Step" << std::endl;

  // Perform setup, initialize first set of visited vertices
  Frontier<GraphTy> frontier, new_frontier;
  #ifdef FULL_COLORS_MOTIVATION
  uint32_t color = 1 << (NumColors - 1);
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    frontier.v.push_back(*B);
    frontier.color.push_back(color);
    host_visited_matrix[*B] |= color;
  }
  #else
  uint32_t color = 1 << (NumColors - 1);
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    frontier.v.push_back(*itr);
    frontier.color.push_back(color);
    host_visited_matrix[*itr] |= color;
  }
  #endif
  visited_matrix = host_visited_matrix;

  // Reduce frontier:
  // -1 sort the frontier by vertex
  // -2 count the unique vertices
  // -2 reduce the frontier by fusing colors
  thrust::sort_by_key(std::begin(frontier.v), std::end(frontier.v),
                      std::begin(frontier.color));
  size_t FrontierSize = thrust::inner_product(
      frontier.v.begin(), frontier.v.end() - 1, frontier.v.begin() + 1, 1,
      thrust::plus<int>(), thrust::not_equal_to<vertex_type>());
  new_frontier.v.resize(FrontierSize);
  new_frontier.color.resize(FrontierSize);
  thrust::reduce_by_key(frontier.v.begin(), frontier.v.end(),
                        frontier.color.begin(), new_frontier.v.begin(),
                        new_frontier.color.begin(), thrust::equal_to<vertex_type>(),
                        thrust::bit_or<vertex_type>());

  thrust::swap(frontier, new_frontier);

  auto d_graph = Context.d_graph;
  auto d_index = d_graph->d_index_;
  auto d_edge = d_graph->d_edges_;
  auto d_weight = d_graph->d_weights_;

  thrust::device_vector<vertex_type> numNeighbors(frontier.v.size() + 1, 0);
  GPUIndependentCascadeScan<vertex_type> simStep;
  GPUEdgeScatter<decltype(d_index), decltype(d_weight), GraphTy> scatEdge{d_index, d_edge, d_weight};
  simStep.visited_matrix = visited_matrix.data().get();
  simStep.frontier_matrix = frontier_matrix.data().get();

  // std::cout << "Process Frontier" << std::endl;

  #ifdef FRONTIER_PROFILE
  size_t iteration = 0;
  #endif

  while (frontier.v.size() != 0) {
    #ifdef FRONTIER_PROFILE
    // size_t num_colors = thrust::transform_reduce(
    //     thrust::device, frontier.color.begin(), frontier.color.end(),
    //     [](const vertex_type &c) { return numColors(c); }, 0,
    //     thrust::plus<vertex_type>());
    // Find largest outdegree node in frontier
    size_t max_outdegree = thrust::transform_reduce(
        thrust::device, std::begin(frontier.v), std::end(frontier.v),
        [d_index](const vertex_type &FE) {
          return d_index[FE + 1] - d_index[FE];
        },
        0, thrust::maximum<size_t>());
    size_t vertex_size = frontier.v.size();
    GPU<RUNTIME>::device_sync();
    auto start_scatter = std::chrono::high_resolution_clock::now();
    #endif
    // std::cout << "Size = " << frontier.v.size() << std::endl;
    // std::cout << "Inclusive Scan" << std::endl;
    thrust::transform_inclusive_scan(
        thrust::device, std::begin(frontier.v), std::end(frontier.v),
        std::begin(numNeighbors) + 1,
        [d_index](const vertex_type &FE) {
          return d_index[FE + 1] - d_index[FE];
        },
        thrust::plus<vertex_type>{});
    
    // #ifdef FRONTIER_PROFILE
    // GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    // #endif

    // #ifdef FRONTIER_PROFILE
    // size_t vertex_size = frontier.v.size();
    // GPU<RUNTIME>::device_sync();
    // auto start_scatter = std::chrono::high_resolution_clock::now();
    // #endif 

    new_frontier.v.resize(numNeighbors.back());
    new_frontier.color.resize(numNeighbors.back() + 1);
    new_frontier.weight.resize(numNeighbors.back() + 1);

    // std::cout << "Edges = " << new_frontier.v.size() << std::endl;

    // #ifdef FRONTIER_PROFILE
    // GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    // #endif

    #ifdef FRONTIER_PROFILE
    // Find largest outdegree node in frontier
    // size_t max_outdegree = thrust::transform_reduce(
    //     thrust::device, std::begin(frontier.v), std::end(frontier.v),
    //     [d_index](const vertex_type &FE) {
    //       return d_index[FE + 1] - d_index[FE];
    //     },
    //     0, thrust::maximum<size_t>());
    // size_t vertex_size = frontier.v.size();
    // GPU<RUNTIME>::device_sync();
    // auto start_scatter = std::chrono::high_resolution_clock::now();
    #endif 

    auto O = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.begin(), new_frontier.color.begin(),
                           new_frontier.weight.begin()));
    auto B = thrust::make_zip_iterator(thrust::make_tuple(
        frontier.v.begin(), frontier.color.begin(), numNeighbors.begin(),
        thrust::constant_iterator<decltype(O)>(O)));
    auto E = thrust::make_zip_iterator(thrust::make_tuple(
        frontier.v.end(), frontier.color.end(), numNeighbors.end() - 1,
        thrust::constant_iterator<decltype(O)>(O)));
    thrust::for_each(thrust::device, B, E, scatEdge);

    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    auto end_scatter = std::chrono::high_resolution_clock::now();
    size_t edge_size = new_frontier.v.size();
    // Determine number of 1 bits in color
    size_t num_colors = thrust::transform_reduce(
        thrust::device, new_frontier.color.begin(), new_frontier.color.end(),
        [](const vertex_type &c) { return numColors(c); }, 0,
        thrust::plus<uint32_t>());
    GPU<RUNTIME>::device_sync();
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // auto seedItrB = thrust::counting_iterator<uint64_t>(clock());
    auto seedItrB = thrust::counting_iterator<uint64_t>(clock());
    auto seedItrE = seedItrB + thrust::distance(new_frontier.v.begin(),
                                                new_frontier.v.end());
    auto edgeB = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.begin(), new_frontier.color.begin(),
                           new_frontier.weight.begin(), seedItrB));
    auto edgeE = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.end(), new_frontier.color.end(),
                           new_frontier.weight.end(), seedItrE));
    // std::cout << "Sim Step" << std::endl;
    thrust::for_each(thrust::device, edgeB, edgeE, simStep);

    if (new_frontier.v.size() == 0) break;

    // std::cout << "Rebuild" << std::endl;

    frontier.v.resize(frontier_matrix.size());
    frontier.color.resize(frontier_matrix.size());

    // Rebuild new frontier queue
    auto vertexBegin = thrust::counting_iterator<vertex_type>(0);
    auto vertexEnd = vertexBegin + frontier_matrix.size();
    auto frontierB = thrust::make_zip_iterator(
      thrust::make_tuple(frontier.v.begin(), frontier.color.begin()));
    auto copyB = thrust::make_zip_iterator(
      thrust::make_tuple(vertexBegin, frontier_matrix.begin()));
    auto copyE = thrust::make_zip_iterator(
      thrust::make_tuple(vertexEnd, frontier_matrix.end()));
    
    auto frontierE = thrust::copy_if(copyB, copyE, frontierB, notZero<vertex_type>());

    frontier.v.resize(thrust::distance(frontierB, frontierE));
    frontier.color.resize(thrust::distance(frontierB, frontierE));
    thrust::fill(frontier_matrix.begin(), frontier_matrix.end(),  0);

    numNeighbors.resize(frontier.v.size() + 1);

    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto time_scatter = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_scatter - start_scatter);
    // profile_vector.push_back({vertex_size, time_scatter.count(), num_colors, vertex_size, time_scatter.count(), iteration});
    iteration++;
    profile_vector.push_back({edge_size, time.count(), num_colors, vertex_size, time_scatter.count(), max_outdegree, iteration});
    #endif
  }

  // std::cout << "Clean Frontier" << std::endl;

  host_visited_matrix = visited_matrix;

  for (vertex_type v = 0; v < host_visited_matrix.size(); ++v) {
    if (host_visited_matrix[v] == 0) continue;

    vertex_type colors = host_visited_matrix[v];

    while (colors != 0) {
      vertex_type color = getNextColor(colors);

      (O + color)->push_back(v);

      colors = clearColor(colors, color);
    }
  }
}

#if defined(RIPPLES_ENABLE_CUDA)
#define SMALL_THRESHOLD 32
#elif defined(RIPPLES_ENABLE_HIP)
#define SMALL_THRESHOLD 64
#else
// Unsupported GPU runtime
#error "Unsupported GPU runtime"
#endif
#define MEDIUM_THRESHOLD 256
#define LARGE_THRESHOLD 65536
#define NUM_LEVELS 4

template <typename GraphTy, typename DeviceContextTy, typename SItrTy,
          typename OItrTy, typename diff_model_tag, typename ColorTy = typename GraphTy::vertex_type>
void GPUBatchedTieredQueueBFS(const GraphTy &G, const DeviceContextTy &Context, SItrTy B,
                   SItrTy E, OItrTy O, diff_model_tag &&tag, BFSContext<GraphTy, ColorTy> &bfs_ctx,
                   ColorTy NumColors = 32) {
  using DeviceGraphTy = typename DeviceContextTy::device_graph_type;
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = ColorTy;

  // std::cout << "Initialization: " << Context.gpu_id << std::endl;

  // constexpr unsigned int NumColors = 8 * sizeof(color_type);

  assert(std::distance(B, E) <= NumColors &&
         "Only up to sizeof(color_type) BFS are supported.");

  GPU<RUNTIME>::set_device(Context.gpu_id);

  auto &streams = bfs_ctx.streams;
  auto &frontier_matrix = bfs_ctx.frontier_matrix;
  thrust::fill(frontier_matrix.begin(), frontier_matrix.end(), 0);
  auto &visited_matrix = bfs_ctx.visited_matrix;
  auto &host_visited_matrix = bfs_ctx.host_visited_matrix;
  thrust::fill(host_visited_matrix.begin(), host_visited_matrix.end(), 0);

  auto d_graph = Context.d_graph;
  auto d_index = d_graph->d_index_;
  auto d_edge = d_graph->d_edges_;
  auto d_weight = d_graph->d_weights_;

  // GPU<RUNTIME>::device_sync();
  // std::cout << "Device Vectors: " << Context.gpu_id << std::endl;

  // Perform setup, initialize first set of visited vertices
  auto &frontier = bfs_ctx.frontier;
  auto &new_frontier = bfs_ctx.new_frontier;
  auto &small_frontier = bfs_ctx.small_frontier;
  auto &medium_frontier = bfs_ctx.medium_frontier;
  auto &large_frontier = bfs_ctx.large_frontier;
  auto &extreme_frontier = bfs_ctx.extreme_frontier;
  #ifdef FULL_COLORS_MOTIVATION
  color_type color = (color_type)1 << (NumColors - 1);
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    frontier.v.push_back(*B);
    frontier.color.push_back(color);
    host_visited_matrix[*B] |= color;
  }
  #else
  color_type color = (color_type)1 << (NumColors - 1);
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    frontier.v.push_back(*itr);
    frontier.color.push_back(color);
    host_visited_matrix[*itr] |= color;
  }
  #endif
  visited_matrix = host_visited_matrix;

  // GPU<RUNTIME>::device_sync();
  // std::cout << "Color Setup: " << Context.gpu_id << std::endl;

  // Reduce frontier:
  // -1 sort the frontier by vertex
  // -2 count the unique vertices
  // -2 reduce the frontier by fusing colors
  thrust::sort_by_key(std::begin(frontier.v), std::end(frontier.v),
                      std::begin(frontier.color));
  // GPU<RUNTIME>::device_sync();
  // std::cout << "Inner Prod: " << Context.gpu_id << std::endl;
  size_t FrontierSize = thrust::inner_product(
      frontier.v.begin(), frontier.v.end() - 1, frontier.v.begin() + 1, 1,
      thrust::plus<vertex_type>(), thrust::not_equal_to<vertex_type>());
  // GPU<RUNTIME>::device_sync();
  // std::cout << "Resize + Reduce: " << Context.gpu_id << std::endl;
  new_frontier.v.resize(FrontierSize);
  new_frontier.color.resize(FrontierSize);
  thrust::reduce_by_key(frontier.v.begin(), frontier.v.end(),
                        frontier.color.begin(), new_frontier.v.begin(),
                        new_frontier.color.begin(), thrust::equal_to<vertex_type>(),
                        thrust::bit_or<color_type>());
  // GPU<RUNTIME>::device_sync();
  // std::cout << "Swap: " << Context.gpu_id << std::endl;
  // thrust::swap(thrust::device, frontier, new_frontier);
  frontier.v.swap(new_frontier.v);
  frontier.color.swap(new_frontier.color);

  // GPU<RUNTIME>::device_sync();
  // std::cout << "Num Neighbors: " << Context.gpu_id << std::endl;

  // thrust::device_vector<vertex_type> numNeighbors(frontier.v.size() + 1, 0);
  auto &numNeighbors = bfs_ctx.numNeighbors;
  numNeighbors.resize(frontier.v.size() + 1);
  // thrust::fill(numNeighbors.begin(), numNeighbors.end(), 0);
  auto &numNeighborsSeparate = bfs_ctx.numNeighborsSeparate;
  numNeighborsSeparate.resize(frontier.v.size() + 1);
  numNeighborsSeparate[0] = 0;
  // thrust::fill(numNeighborsSeparate.begin(), numNeighborsSeparate.end(), 0);
  // thrust::device_vector<vertex_type> numNeighborsSeparate(frontier.v.size() + 1, 0);
  GPUIndependentCascadeScan<color_type> simStep;
  // GPUEdgeScatter<decltype(d_index), decltype(d_weight), GraphTy> scatEdge{d_index, d_edge, d_weight};
  simStep.visited_matrix = visited_matrix.data().get();
  simStep.frontier_matrix = frontier_matrix.data().get();

  // GPU<RUNTIME>::device_sync();
  // std::cout << "Process Frontier: " << Context.gpu_id << std::endl;

  #ifdef FRONTIER_PROFILE
  size_t iteration = 0;
  #endif

  while (frontier.v.size() != 0) {
    #ifdef FRONTIER_PROFILE
    size_t num_colors = thrust::transform_reduce(
        thrust::device, frontier.color.begin(), frontier.color.end(),
        [](const color_type &c) { return numColors(c); }, 0,
        thrust::plus<uint32_t>());
    size_t unique_colors = numColors(thrust::reduce(
        thrust::device, frontier.color.begin(), frontier.color.end(),
        (color_type)0, thrust::bit_or<color_type>()));
    size_t vertex_size = frontier.v.size();
    // GPU<RUNTIME>::device_sync();
    // Find largest outdegree node in frontier
    size_t max_outdegree = thrust::transform_reduce(
        thrust::device, std::begin(frontier.v), std::end(frontier.v),
        [d_index](const vertex_type &FE) {
          return d_index[FE + 1] - d_index[FE];
        },
        0, thrust::maximum<size_t>());
    GPU<RUNTIME>::device_sync();
    auto start_scatter = std::chrono::high_resolution_clock::now();
    #endif
    // GPU<RUNTIME>::device_sync();
    // std::cout << "Size = " << frontier.v.size() << std::endl;
    // std::cout << "Inclusive Scan" << std::endl;
    thrust::transform(
        thrust::device, std::begin(frontier.v), std::end(frontier.v),
        std::begin(numNeighborsSeparate) + 1,
        [d_index](const vertex_type &FE) {
          return d_index[FE + 1] - d_index[FE];
        });
    
    thrust::inclusive_scan(thrust::device, std::begin(numNeighborsSeparate), std::end(numNeighborsSeparate),
                        std::begin(numNeighbors), thrust::plus<vertex_type>{});
    
    auto toClassifyB = thrust::make_zip_iterator(thrust::make_tuple(frontier.v.begin(), frontier.color.begin(), numNeighbors.begin()));
    auto toClassifyE = thrust::make_zip_iterator(thrust::make_tuple(frontier.v.end(), frontier.color.end(), numNeighbors.end() - 1));

    // Distribute the frontier to the appropriate queues
    auto smallCopyB = thrust::make_zip_iterator(thrust::make_tuple(small_frontier.v.begin(), small_frontier.color.begin(), small_frontier.offset.begin()));
    auto smallsize = thrust::copy_if(thrust::device, toClassifyB, toClassifyE, std::begin(numNeighborsSeparate) + 1,
                    smallCopyB, [](const vertex_type &degree) {
                      return degree < SMALL_THRESHOLD;
                    });
    auto smallWork = thrust::distance(smallCopyB, smallsize);
    auto mediumCopyB = thrust::make_zip_iterator(thrust::make_tuple(medium_frontier.v.begin(), medium_frontier.color.begin(), medium_frontier.offset.begin()));
    auto mediumsize = thrust::copy_if(thrust::device, toClassifyB, toClassifyE, std::begin(numNeighborsSeparate) + 1,
                    mediumCopyB, [](const vertex_type &degree) {
                      return degree >= SMALL_THRESHOLD && degree < MEDIUM_THRESHOLD;
                    });
    auto mediumWork = thrust::distance(mediumCopyB, mediumsize);
    auto largeCopyB = thrust::make_zip_iterator(thrust::make_tuple(large_frontier.v.begin(), large_frontier.color.begin(), large_frontier.offset.begin()));
    auto largesize = thrust::copy_if(thrust::device, toClassifyB, toClassifyE, std::begin(numNeighborsSeparate) + 1,
                    largeCopyB, [](const vertex_type &degree) {
                      return degree >= MEDIUM_THRESHOLD && degree < LARGE_THRESHOLD;
                    });
    auto largeWork = thrust::distance(largeCopyB, largesize);
    auto extremeCopyB = thrust::make_zip_iterator(thrust::make_tuple(extreme_frontier.v.begin(), extreme_frontier.color.begin(), extreme_frontier.offset.begin()));
    auto extremesize = thrust::copy_if(thrust::device, toClassifyB, toClassifyE, std::begin(numNeighborsSeparate) + 1,
                    extremeCopyB, [](const vertex_type &degree) {
                      return degree >= LARGE_THRESHOLD;
                    });
    auto extremeWork = thrust::distance(extremeCopyB, extremesize);
    
    
    
    // #ifdef FRONTIER_PROFILE
    // GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    // #endif

    // #ifdef FRONTIER_PROFILE
    // size_t vertex_size = frontier.v.size();
    // GPU<RUNTIME>::device_sync();
    // auto start_scatter = std::chrono::high_resolution_clock::now();
    // #endif 

    new_frontier.v.resize(numNeighbors.back());
    new_frontier.color.resize(numNeighbors.back() + 1);
    new_frontier.weight.resize(numNeighbors.back() + 1);

    // #ifdef FRONTIER_PROFILE
    // GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    // #endif

    // auto O = thrust::make_zip_iterator(
    //     thrust::make_tuple(new_frontier.v.begin(), new_frontier.color.begin(),
    //                        new_frontier.weight.begin()));
    // auto B = thrust::make_zip_iterator(thrust::make_tuple(
    //     frontier.v.begin(), frontier.color.begin(), numNeighbors.begin(),
    //     thrust::constant_iterator<decltype(O)>(O)));
    // auto E = thrust::make_zip_iterator(thrust::make_tuple(
    //     frontier.v.end(), frontier.color.end(), numNeighbors.end() - 1,
    //     thrust::constant_iterator<decltype(O)>(O)));
    // thrust::for_each(thrust::device, B, E, scatEdge);

    auto new_frontier_v_ptr = thrust::raw_pointer_cast(new_frontier.v.data());
    auto new_frontier_color_ptr = thrust::raw_pointer_cast(new_frontier.color.data());
    auto new_frontier_weight_ptr = thrust::raw_pointer_cast(new_frontier.weight.data());
    auto small_frontier_v_ptr = thrust::raw_pointer_cast(small_frontier.v.data());
    auto small_frontier_color_ptr = thrust::raw_pointer_cast(small_frontier.color.data());
    auto small_frontier_offset_ptr = thrust::raw_pointer_cast(small_frontier.offset.data());
    auto medium_frontier_v_ptr = thrust::raw_pointer_cast(medium_frontier.v.data());
    auto medium_frontier_color_ptr = thrust::raw_pointer_cast(medium_frontier.color.data());
    auto medium_frontier_offset_ptr = thrust::raw_pointer_cast(medium_frontier.offset.data());
    auto large_frontier_v_ptr = thrust::raw_pointer_cast(large_frontier.v.data());
    auto large_frontier_color_ptr = thrust::raw_pointer_cast(large_frontier.color.data());
    auto large_frontier_offset_ptr = thrust::raw_pointer_cast(large_frontier.offset.data());
    auto extreme_frontier_v_ptr = thrust::raw_pointer_cast(extreme_frontier.v.data());
    auto extreme_frontier_color_ptr = thrust::raw_pointer_cast(extreme_frontier.color.data());
    auto extreme_frontier_offset_ptr = thrust::raw_pointer_cast(extreme_frontier.offset.data());
    GPU<RUNTIME>::device_sync();
    // std::cout << "smallWork: " << smallWork << " mediumWork: " << mediumWork << " largeWork: " << largeWork << " extremeWork: " << extremeWork << std::endl;
    #if defined(RIPPLES_ENABLE_CUDA)
    if(smallWork > 0)
      thread_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<(smallWork+31)/32, 32, 0, streams[0]>>>(d_index,
          d_edge, d_weight,
          small_frontier_v_ptr, small_frontier_color_ptr, small_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          smallWork);
    if(mediumWork > 0)
      warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<mediumWork, 32, 0, streams[1]>>>(d_index,
          d_edge, d_weight,
          medium_frontier_v_ptr, medium_frontier_color_ptr, medium_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          mediumWork);
    if(largeWork > 0)
      warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<largeWork, 256, 0, streams[2]>>>(d_index,
          d_edge, d_weight,
          large_frontier_v_ptr, large_frontier_color_ptr, large_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          largeWork);
    if(extremeWork > 0)
      warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<extremeWork, 1024, 0, streams[3]>>>(d_index,
          d_edge, d_weight,
          extreme_frontier_v_ptr, extreme_frontier_color_ptr, extreme_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          extremeWork);
    #elif defined(RIPPLES_ENABLE_HIP)
    // std::cout << "smallWork" << std::endl;
    if(smallWork > 0)
      hipLaunchKernelGGL((thread_scatter_kernel<RUNTIME, GraphTy, ColorTy>), (smallWork+63)/64, 64,
                        0, streams[0], d_index,
                        d_edge, d_weight,
                        small_frontier_v_ptr, small_frontier_color_ptr, small_frontier_offset_ptr,
                        new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
                        smallWork);
    // GPU<RUNTIME>::device_sync();
    // std::cout << "mediumWork" << std::endl;
    if(mediumWork > 0)
      hipLaunchKernelGGL((warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>), mediumWork, 64,
                        0, streams[1], d_index,
                        d_edge, d_weight,
                        medium_frontier_v_ptr, medium_frontier_color_ptr, medium_frontier_offset_ptr,
                        new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
                        mediumWork);
    // GPU<RUNTIME>::device_sync();
    // std::cout << "largeWork" << std::endl;
    if(largeWork > 0)
      hipLaunchKernelGGL((warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>), largeWork, 256,
                        0, streams[2], d_index,
                        d_edge, d_weight,
                        large_frontier_v_ptr, large_frontier_color_ptr, large_frontier_offset_ptr,
                        new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
                        largeWork);
    // GPU<RUNTIME>::device_sync();
    // std::cout << "extremeWork" << std::endl;
    if(extremeWork > 0)
      hipLaunchKernelGGL((warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>), extremeWork, 1024,
                        0, streams[3], d_index,
                        d_edge, d_weight,
                        extreme_frontier_v_ptr, extreme_frontier_color_ptr, extreme_frontier_offset_ptr,
                        new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
                        extremeWork);
    #else
    #error "Unsupported GPU runtime"
    #endif
    
    GPU<RUNTIME>::device_sync();
    // std::cout << "Done With Streams: " << Context.gpu_id << std::endl;
    // std::cout << "Edges = " << new_frontier.v.size() << std::endl;

    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    auto end_scatter = std::chrono::high_resolution_clock::now();
    size_t edge_size = new_frontier.v.size();
    // Determine number of 1 bits in color
    size_t edge_colors = thrust::transform_reduce(
        thrust::device, new_frontier.color.begin(), new_frontier.color.end(),
        [](const color_type &c) { return numColors(c); }, 0,
        thrust::plus<uint32_t>());
    GPU<RUNTIME>::device_sync();
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // auto seedItrB = thrust::counting_iterator<uint64_t>(clock());
    auto seedItrB = thrust::counting_iterator<uint64_t>(clock());
    auto seedItrE = seedItrB + thrust::distance(new_frontier.v.begin(),
                                                new_frontier.v.end());
    auto edgeB = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.begin(), new_frontier.color.begin(),
                           new_frontier.weight.begin(), seedItrB));
    auto edgeE = thrust::make_zip_iterator(
        thrust::make_tuple(new_frontier.v.end(), new_frontier.color.end(),
                           new_frontier.weight.end(), seedItrE));
    // std::cout << "Sim Step" << std::endl;
    thrust::for_each(thrust::device, edgeB, edgeE, simStep);

    if (new_frontier.v.size() == 0) break;

    // std::cout << "Rebuild" << std::endl;

    frontier.v.resize(frontier_matrix.size());
    frontier.color.resize(frontier_matrix.size());

    // Rebuild new frontier queue
    auto vertexBegin = thrust::counting_iterator<vertex_type>(0);
    auto vertexEnd = vertexBegin + frontier_matrix.size();
    auto frontierB = thrust::make_zip_iterator(
      thrust::make_tuple(frontier.v.begin(), frontier.color.begin()));
    auto copyB = thrust::make_zip_iterator(
      thrust::make_tuple(vertexBegin, frontier_matrix.begin()));
    auto copyE = thrust::make_zip_iterator(
      thrust::make_tuple(vertexEnd, frontier_matrix.end()));
    
    auto frontierE = thrust::copy_if(copyB, copyE, frontierB, notZero<vertex_type, color_type>());

    frontier.v.resize(thrust::distance(frontierB, frontierE));
    frontier.color.resize(thrust::distance(frontierB, frontierE));
    thrust::fill(frontier_matrix.begin(), frontier_matrix.end(),  0);

    numNeighbors.resize(frontier.v.size() + 1);
    numNeighborsSeparate.resize(frontier.v.size() + 1);

    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto time_scatter = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_scatter - start_scatter);
    // profile_vector.push_back({vertex_size, time_scatter.count(), num_colors, vertex_size, time_scatter.count(), iteration});
    iteration++;
    profile_vector.push_back({edge_size, time.count(), num_colors, vertex_size, time_scatter.count(), max_outdegree, iteration, edge_colors, unique_colors});
    #endif
  }

  // std::cout << "Clean Frontier" << std::endl;

  host_visited_matrix = visited_matrix;

  for (vertex_type v = 0; v < host_visited_matrix.size(); ++v) {
    if (host_visited_matrix[v] == 0) continue;

    color_type colors = host_visited_matrix[v];
    // std::cout << "Color = " << colors << "\n";

    while (colors != 0) {
      color_type color = getNextColor(colors);

      (O + color)->push_back(v);

      colors = clearColor(colors, color);
    }
    // std::cout << "Done" << "\n";
  }
  // std::cout << "Done Removing" << "\n";
}

template <typename GraphTy, typename DeviceContextTy, typename SItrTy,
          typename OItrTy, typename diff_model_tag,
          typename BFSCtxTy>
void GPUBatchedBFSMultiColorFused(
    const GraphTy &G, const DeviceContextTy &Context, SItrTy B, SItrTy E,
    OItrTy O, diff_model_tag &&tag,
    BFSCtxTy &bfs_ctx) {
  using DeviceGraphTy = typename DeviceContextTy::device_graph_type;
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = typename BFSCtxTy::color_type;

  #ifdef PROFILE_OVERHEAD
  auto start = std::chrono::high_resolution_clock::now();
  #endif // PROFILE_OVERHEAD

  // std::cout << "Initialization: " << Context.gpu_id << std::endl;

  #if defined(RIPPLES_ENABLE_CUDA)
    constexpr size_t thread_size = 32;
    typedef uint32_t WarpMaskTy;
  #elif defined(RIPPLES_ENABLE_HIP)
    constexpr size_t thread_size = 64;
    typedef uint64_t WarpMaskTy;
  #else
    #error "Unsupported GPU runtime"
  #endif

  const size_t num_colors = std::distance(B, E);
  constexpr size_t color_size = sizeof(color_type) * 8;
  const size_t unrounded_color_dim = (num_colors + color_size - 1) / color_size;
  // Round color_dim to a power of 2
  const size_t color_dim = (size_t)1
                           << (size_t)std::ceil(std::log2(unrounded_color_dim));
  // // Print out unrounded color_dim and color_dim
  // std::cout << "unrounded_color_dim = " << unrounded_color_dim << std::endl;
  // std::cout << "color_dim = " << color_dim << std::endl;
  // // Print num_colors
  // std::cout << "num_colors = " << num_colors << std::endl;

  // Assert color_dim is less than or equal to warp size
#if defined(RIPPLES_ENABLE_CUDA)
  assert(color_dim <= 32 &&
         "color_dim must be less than or equal to warp size.");
#elif defined(RIPPLES_ENABLE_HIP)
  assert(color_dim <= 64 &&
         "color_dim must be less than or equal to warp size.");
#endif

  GPU<RUNTIME>::set_device(Context.gpu_id);

  auto &streams = bfs_ctx.streams;
  auto &frontier_matrix = bfs_ctx.frontier_matrix;
  thrust::fill(thrust::device.on(streams[0]), frontier_matrix.begin(), frontier_matrix.end(), 0);
  auto &visited_matrix = bfs_ctx.visited_matrix;
  thrust::fill(thrust::device.on(streams[0]), visited_matrix.begin(), visited_matrix.end(), 0);
  auto &host_visited_matrix = bfs_ctx.host_visited_matrix;

  auto d_graph = Context.d_graph;
  auto d_index = d_graph->d_index_;
  auto d_edge = d_graph->d_edges_;
  auto d_weight = d_graph->d_weights_;  

  // GPU<RUNTIME>::device_sync();
  // std::cout << "Device Vectors: " << Context.gpu_id << std::endl;

  // Perform setup, initialize first set of visited vertices

  auto &workloads = bfs_ctx.workloads;
  auto &host_workloads = bfs_ctx.host_workloads;
  color_type color = (color_type)1 << (color_size - 1);
  size_t count = 0;
#ifdef FULL_COLORS_MOTIVATION
  for (auto itr = B; itr < E; ++itr, ++count) {
    const size_t color_block_id = count / color_size;
    frontier_matrix[(*B) * color_dim + color_block_id] |= color;
    color =
        color == (color_type)1 ? (color_type)1 << (color_size - 1) : (color_type)color >> 1;
  }
#else
  for (auto itr = B; itr < E; ++itr, ++count) {
    assert(color != 0);
    const size_t color_block_id = count / color_size;
    frontier_matrix[(*itr) * color_dim + color_block_id] |= color;
    color =
        color == (color_type)1 ? (color_type)1 << (color_size - 1) : (color_type)color >> 1;
  }
#endif

#ifdef FRONTIER_PROFILE
  size_t iteration = 0;
#endif

  bool finished = false;
  auto small_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.small_vertices.data());
  auto small_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.small_colors.data());
  auto medium_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.medium_vertices.data());
  auto medium_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.medium_colors.data());
  auto large_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.large_vertices.data());
  auto large_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.large_colors.data());
  auto extreme_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.extreme_vertices.data());
  auto extreme_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.extreme_colors.data());
  auto workloads_ptr = thrust::raw_pointer_cast(workloads.data());
  auto visited_matrix_ptr = thrust::raw_pointer_cast(visited_matrix.data());
  auto frontier_matrix_ptr = thrust::raw_pointer_cast(frontier_matrix.data());
  constexpr size_t num_threads = 256;
  const size_t num_nodes_per_block = num_threads / color_dim;
  const size_t num_build_blocks =
      (G.num_nodes() + num_nodes_per_block - 1) / num_nodes_per_block;

  #ifdef PROFILE_OVERHEAD
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  start = std::chrono::high_resolution_clock::now();
  size_t num_traversals = 0;
  size_t remaining_count = 0;
  #endif // PROFILE_OVERHEAD
  
  while (!finished) {
    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    // XOR frontier matrix with visited matrix
    thrust::device_vector<color_type> frontier_matrix_xor(frontier_matrix.size());
    thrust::transform(thrust::device, frontier_matrix.begin(), frontier_matrix.end(), visited_matrix.begin(), frontier_matrix_xor.begin(), thrust::bit_xor<color_type>());
    size_t p_num_colors = thrust::transform_reduce(
        thrust::device, frontier_matrix_xor.begin(), frontier_matrix_xor.end(),
        [](const color_type &c) { return numColors(c); }, 0,
        thrust::plus<uint32_t>());
    // size_t unique_colors = numColors(thrust::reduce(
    //     thrust::device, frontier.color.begin(), frontier.color.end(),
    //     (color_type)0, thrust::bit_or<color_type>()));
    size_t unique_colors = 0;
    // Count number of values that aren't 0 in frontier_matrix_xor
    // size_t vertex_size = frontier.v.size();
    // GPU<RUNTIME>::device_sync();
    // Find largest outdegree node in frontier
    // size_t max_outdegree = thrust::transform_reduce(
    //     thrust::device, std::begin(frontier.v), std::end(frontier.v),
    //     [d_index](const vertex_type &FE) {
    //       return d_index[FE + 1] - d_index[FE];
    //     },
    //     0, thrust::maximum<size_t>());
    size_t max_outdegree = 0;
    GPU<RUNTIME>::device_sync();
    auto start_scatter = std::chrono::high_resolution_clock::now();
    #endif
    finished = true;
    // Set workloads to 0
    thrust::fill(thrust::device.on(streams[0]), workloads.begin(), workloads.end(), 0);
    GPU<RUNTIME>::stream_sync(streams[0]);
    build_frontier_queues_kernel<RUNTIME, GraphTy, color_type, WarpMaskTy>
        <<<num_build_blocks, num_threads, 0, streams[0]>>>(
            d_index, small_vertices_ptr, medium_vertices_ptr,
            large_vertices_ptr, extreme_vertices_ptr, 
            small_colors_ptr, medium_colors_ptr, large_colors_ptr, extreme_colors_ptr,
            visited_matrix_ptr,
            frontier_matrix_ptr, workloads_ptr, G.num_nodes(), num_colors,
            color_dim);
    GPU<RUNTIME>::stream_sync(streams[0]);

    const size_t block_size = thread_size / color_dim;
    // Retrieve workload sizes
    host_workloads = workloads;

    // // print workloads from host_workloads
    // for (size_t i = 0; i < host_workloads.size(); i++) {
    //   std::cout << "Workload " << i << " = " << host_workloads[i] << std::endl;
    // }

    size_t threshold = 0;
    const size_t num_blocks =
        (host_workloads[threshold] + block_size - 1) / block_size;
    // Enqueue binned kernels
    #ifdef UTILIZATION_PROFILE
    bfs_ctx.small_queue.push_back(host_workloads[threshold]);
    bfs_ctx.small_warps.push_back(num_blocks * thread_size);
    #endif // UTILIZATION_PROFILE
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_thread_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<num_blocks, thread_size, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, small_vertices_ptr,
              small_colors_ptr, frontier_matrix_ptr,
              host_workloads[threshold], color_dim);
    }
    threshold++;
    #ifdef UTILIZATION_PROFILE
    bfs_ctx.medium_queue.push_back(host_workloads[threshold]);
    bfs_ctx.medium_warps.push_back((host_workloads[threshold] * thread_size));
    #endif // UTILIZATION_PROFILE
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_set_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<host_workloads[threshold], thread_size, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, medium_vertices_ptr,
              medium_colors_ptr,
              frontier_matrix_ptr, host_workloads[threshold], color_dim);
    }
    threshold++;
    #ifdef UTILIZATION_PROFILE
    bfs_ctx.large_queue.push_back(host_workloads[threshold]);
    bfs_ctx.large_warps.push_back((host_workloads[threshold] * 256));
    #endif // UTILIZATION_PROFILE
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_set_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<host_workloads[threshold], 256, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, large_vertices_ptr,
              large_colors_ptr, frontier_matrix_ptr,
              host_workloads[threshold], color_dim);
    }
    threshold++;
    #ifdef UTILIZATION_PROFILE
    bfs_ctx.extreme_queue.push_back(host_workloads[threshold]);
    bfs_ctx.extreme_warps.push_back((host_workloads[threshold] * 1024));
    #endif // UTILIZATION_PROFILE
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_set_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<host_workloads[threshold], 1024, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, extreme_vertices_ptr,
              extreme_colors_ptr,
              frontier_matrix_ptr, host_workloads[threshold], color_dim);
    }
    for (size_t i = 0; i < streams.size(); i++){
      GPU<RUNTIME>::stream_sync(streams[i]); 
    }
    #ifdef FRONTIER_PROFILE
    // Add up host_workloads 0 through 3
    size_t vertex_size = 0;
    for (size_t i = 0; i < host_workloads.size(); i++) {
      vertex_size += host_workloads[i];
    }
    auto end_scatter = std::chrono::high_resolution_clock::now();
    // size_t edge_size = new_frontier.v.size();
    size_t edge_size = 0;
    // Determine number of 1 bits in color
    // size_t edge_colors = thrust::transform_reduce(
    //     thrust::device, new_frontier.color.begin(), new_frontier.color.end(),
    //     [](const color_type &c) { return numColors(c); }, 0,
    //     thrust::plus<uint32_t>());
    size_t edge_colors = 0;
    GPU<RUNTIME>::device_sync();
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto time_scatter = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_scatter - start_scatter);
    // profile_vector.push_back({vertex_size, time_scatter.count(), num_colors, vertex_size, time_scatter.count(), iteration});
    iteration++;
    profile_vector.push_back({edge_size, time.count(), p_num_colors, vertex_size, time_scatter.count(), max_outdegree, iteration, edge_colors, unique_colors});
    #endif
    #ifdef PROFILE_OVERHEAD
    num_traversals++;
    remaining_count += num_colors;
    #endif // PROFILE_OVERHEAD
  }

  #ifdef PROFILE_OVERHEAD
  end = std::chrono::high_resolution_clock::now();
  auto traversal_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  start = std::chrono::high_resolution_clock::now();
  #endif // PROFILE_OVERHEAD

  // std::cout << "Free! " << Context.gpu_id << std::endl;

  host_visited_matrix = visited_matrix;
  // size_t num_vertices_check = 0;
  for (vertex_type v = 0; v < G.num_nodes(); v++) {
    for(size_t color_set = 0; color_set < unrounded_color_dim; color_set++){
      if(host_visited_matrix[(v * color_dim) + color_set] == 0) continue;

      color_type colors = host_visited_matrix[(v * color_dim) + color_set];
      // num_vertices_check += numColors(colors);
      size_t offset = color_set * color_size;
      while (colors != 0) {
        color_type color = getNextColor(colors);
        // if(offset >= num_colors){
        //   std::cout << "Color = " << color << "\n";
        //   assert(color < num_colors && "Color out of bounds");
        // }

        (O + (offset + color))->push_back(v);

        colors = clearColor(colors, color);
      }
    }
    // std::cout << "Done" << "\n";
  }
  // Add up out number of vertices in O + color for each color
  // size_t num_vertices = 0;
  // for (size_t color = 0; color < num_colors; color++) {
  //   num_vertices += (O + color)->size();
  // }
  // std::cout << "Number of vertices = " << num_vertices << " num_check = " << num_vertices_check << std::endl;
  // assert(num_vertices >= num_colors && "Number of vertices is incorrect");
  // std::cout << "Done Removing" << "\n";

  #ifdef PROFILE_OVERHEAD
  end = std::chrono::high_resolution_clock::now();
  auto end_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // setup_time, traversal_time, end_time
  bfs_ctx.setup_time.push_back(setup_time.count());
  bfs_ctx.traversal_time.push_back(traversal_time.count());
  bfs_ctx.end_time.push_back(end_time.count());
  bfs_ctx.num_traversals.push_back(num_traversals);
  bfs_ctx.remaining_count.push_back(remaining_count);
  bfs_ctx.num_colors.push_back(num_colors);
  #endif // PROFILE_OVERHEAD
}

#ifdef PAUSE_AND_RESUME

template <typename GraphTy, typename DeviceContextTy, typename SItrTy,
          typename OItrTy, typename diff_model_tag,
          typename ThresholdTy, typename BFSCtxTy>
size_t GPUBatchedBFSMultiColorFusedReload(
    const GraphTy &G, const DeviceContextTy &Context, SItrTy B, SItrTy E,
    OItrTy O, diff_model_tag &&tag,
    BFSCtxTy &bfs_ctx,
    const ThresholdTy pause_threshold, const size_t num_colors, const bool reset) {
  using DeviceGraphTy = typename DeviceContextTy::device_graph_type;
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = typename BFSCtxTy::color_type;

  #ifdef PROFILE_OVERHEAD
  auto start = std::chrono::high_resolution_clock::now();
  #endif // PROFILE_OVERHEAD

  // std::cout << "Initialization: " << Context.gpu_id << std::endl;

  #if defined(RIPPLES_ENABLE_CUDA)
    constexpr size_t thread_size = 32;
    typedef uint32_t WarpMaskTy;
  #elif defined(RIPPLES_ENABLE_HIP)
    constexpr size_t thread_size = 64;
    typedef uint64_t WarpMaskTy;
  #else
    #error "Unsupported GPU runtime"
  #endif

  size_t num_remaining = num_colors;
  const size_t num_colors_threshold = (ThresholdTy)num_colors * pause_threshold;
  constexpr size_t color_size = sizeof(color_type) * 8;
  const size_t unrounded_color_dim = (num_colors + color_size - 1) / color_size;
  // Round color_dim to a power of 2
  const size_t color_dim = (size_t)1
                           << (size_t)std::ceil(std::log2(unrounded_color_dim));
  // // Print out unrounded color_dim and color_dim
  // std::cout << "unrounded_color_dim = " << unrounded_color_dim << std::endl;
  // std::cout << "color_dim = " << color_dim << std::endl;
  // // Print num_colors
  // std::cout << "num_colors = " << num_colors << std::endl;

  // Assert color_dim is less than or equal to warp size
#if defined(RIPPLES_ENABLE_CUDA)
  assert(color_dim <= 32 &&
         "color_dim must be less than or equal to warp size.");
#elif defined(RIPPLES_ENABLE_HIP)
  assert(color_dim <= 64 &&
         "color_dim must be less than or equal to warp size.");
#endif

  GPU<RUNTIME>::set_device(Context.gpu_id);
  constexpr size_t num_threads = 256;
  const size_t num_nodes_per_block = num_threads / color_dim;
  const size_t num_build_blocks =
      (G.num_nodes() + num_nodes_per_block - 1) / num_nodes_per_block;

  auto &streams = bfs_ctx.streams;
  auto &frontier_matrix = bfs_ctx.frontier_matrix;
  auto &visited_matrix = bfs_ctx.visited_matrix;
  auto &host_visited_matrix = bfs_ctx.host_visited_matrix;
  auto &workloads = bfs_ctx.workloads;
  auto &reduced_frontier = bfs_ctx.reduced_frontier;
  auto &host_workloads = bfs_ctx.host_workloads;
  auto &host_reduced_frontier = bfs_ctx.host_reduced_frontier;
  auto &output_mappings = bfs_ctx.output_mappings;
  auto small_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.small_vertices.data());
  auto small_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.small_colors.data());
  auto medium_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.medium_vertices.data());
  auto medium_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.medium_colors.data());
  auto large_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.large_vertices.data());
  auto large_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.large_colors.data());
  auto extreme_vertices_ptr = thrust::raw_pointer_cast(bfs_ctx.extreme_vertices.data());
  auto extreme_colors_ptr = thrust::raw_pointer_cast(bfs_ctx.extreme_colors.data());
  auto workloads_ptr = thrust::raw_pointer_cast(workloads.data());
  auto visited_matrix_ptr = thrust::raw_pointer_cast(visited_matrix.data());
  auto frontier_matrix_ptr = thrust::raw_pointer_cast(frontier_matrix.data());
  auto reduced_frontier_ptr = thrust::raw_pointer_cast(bfs_ctx.reduced_frontier.data());
  auto d_graph = Context.d_graph;
  auto d_index = d_graph->d_index_;
  auto d_edge = d_graph->d_edges_;
  auto d_weight = d_graph->d_weights_;
  if(reset){
    thrust::fill(thrust::device.on(streams[0]), frontier_matrix.begin(), frontier_matrix.end(), 0);
    thrust::fill(thrust::device.on(streams[0]), visited_matrix.begin(), visited_matrix.end(), 0);
    thrust::fill(host_reduced_frontier.begin(), host_reduced_frontier.end(), 0);
  }
  else{
    filter_finished<RUNTIME, GraphTy, color_type>
        <<<num_build_blocks, num_threads, 0, streams[0]>>>(
            frontier_matrix_ptr, visited_matrix_ptr, reduced_frontier_ptr, G.num_nodes(), color_dim);
    // Copy reduced_frontier to host_reduced_frontier
  }

  // GPU<RUNTIME>::stream_sync(streams[0]);
  // std::cout << "Device Vectors: " << Context.gpu_id << std::endl;

  // Perform setup, initialize first set of visited vertices
  color_type color = (color_type)1 << (color_size - 1);
  assert(color != 0);
  size_t count = 0;
  size_t offset_count = 0;
  GPU<RUNTIME>::stream_sync(streams[0]);
#ifdef FULL_COLORS_MOTIVATION
  for (auto itr = B; itr < E; ++itr, ++count) {
    size_t color_block_id = count / color_size;
    while(color & host_reduced_frontier[color_block_id] != 0){
      count++;
      color_block_id = count / color_size;
      color =
        color == (color_type)1 ? (color_type)1 << (color_size - 1) : (color_type)color >> 1;
    }
    frontier_matrix[(*B) * color_dim + color_block_id] |= color;
    color =
        color == (color_type)1 ? (color_type)1 << (color_size - 1) : (color_type)color >> 1;
  }
#else // !FULL_COLORS_MOTIVATION
  for (auto itr = B; itr < E; ++itr, ++count) {
    size_t color_block_id = count / color_size;
    while((color & host_reduced_frontier[color_block_id]) != 0){
      count++;
      color_block_id = count / color_size;
      color =
        color == (color_type)1 ? (color_type)1 << (color_size - 1) : (color_type)color >> 1;
    }
    // assert(count < 256);
    frontier_matrix[(*itr) * color_dim + color_block_id] |= color;
    output_mappings[count] = O + std::distance(B, itr);
    color =
        color == (color_type)1 ? (color_type)1 << (color_size - 1) : (color_type)color >> 1;
  }
#endif // !FULL_COLORS_MOTIVATION

#ifdef FRONTIER_PROFILE
  size_t iteration = 0;
#endif

  bool finished = false;

  #ifdef PROFILE_OVERHEAD
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  start = std::chrono::high_resolution_clock::now();
  size_t num_traversals = 0;
  size_t remaining_count = 0;
  #endif // PROFILE_OVERHEAD

  bool resume = !reset;

  while (!finished) {
    // Set workloads to 0
    // Skip building frontier queues if we are resuming
    if(resume){
      resume = false;
      num_remaining = num_colors;
    }
    else{
      thrust::fill(thrust::device.on(streams[0]), workloads.begin(), workloads.end(), 0);
      thrust::fill(thrust::device.on(streams[0]), reduced_frontier.begin(), reduced_frontier.end(), 0);
      build_frontier_queues_kernel_check<RUNTIME, GraphTy, color_type, WarpMaskTy>
          <<<num_build_blocks, num_threads, sizeof(color_type) * (num_threads / thread_size) * color_dim, streams[0]>>>(
              d_index, small_vertices_ptr, medium_vertices_ptr,
              large_vertices_ptr, extreme_vertices_ptr, 
              small_colors_ptr, medium_colors_ptr, large_colors_ptr, extreme_colors_ptr,
              visited_matrix_ptr,
              frontier_matrix_ptr, workloads_ptr, reduced_frontier_ptr, G.num_nodes(), num_colors,
              color_dim);
      GPU<RUNTIME>::stream_sync(streams[0]);
      host_reduced_frontier = reduced_frontier;
      num_remaining = thrust::transform_reduce(
          thrust::host, host_reduced_frontier.begin(), host_reduced_frontier.begin() + color_dim,
          [](const color_type &c) { return numColors(c); }, 0,
          thrust::plus<uint32_t>());
      
      if((num_colors - num_remaining) >= num_colors_threshold){
        finished = true;
        break;
      }
    }

    const size_t block_size = thread_size / color_dim;
    // Retrieve workload sizes
    host_workloads = workloads;

    // // print workloads from host_workloads
    // for (size_t i = 0; i < host_workloads.size(); i++) {
    //   std::cout << "Workload " << i << " = " << host_workloads[i] << std::endl;
    // }

    size_t threshold = 0;
    const size_t num_blocks =
        (host_workloads[threshold] + block_size - 1) / block_size;
    // Enqueue binned kernels
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_thread_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<num_blocks, thread_size, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, small_vertices_ptr,
              small_colors_ptr, frontier_matrix_ptr,
              host_workloads[threshold], color_dim);
    }
    threshold++;
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_set_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<host_workloads[threshold], thread_size, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, medium_vertices_ptr,
              medium_colors_ptr,
              frontier_matrix_ptr, host_workloads[threshold], color_dim);
    }
    threshold++;
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_set_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<host_workloads[threshold], 256, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, large_vertices_ptr,
              large_colors_ptr, frontier_matrix_ptr,
              host_workloads[threshold], color_dim);
    }
    threshold++;
    if (host_workloads[threshold] > 0){
      finished = false;
      fused_color_set_scatter_kernel<RUNTIME, GraphTy, color_type>
          <<<host_workloads[threshold], 1024, 0, streams[threshold]>>>(
              d_index, d_edge, d_weight, extreme_vertices_ptr,
              extreme_colors_ptr,
              frontier_matrix_ptr, host_workloads[threshold], color_dim);
    }
    for (size_t i = 0; i < streams.size(); i++){
      GPU<RUNTIME>::stream_sync(streams[i]); 
    }
    #ifdef PROFILE_OVERHEAD
    num_traversals++;
    remaining_count += num_remaining;
    #endif // PROFILE_OVERHEAD
  }

  #ifdef PROFILE_OVERHEAD
  end = std::chrono::high_resolution_clock::now();
  auto traversal_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  start = std::chrono::high_resolution_clock::now();
  #endif // PROFILE_OVERHEAD

  // std::cout << "Free! " << Context.gpu_id << std::endl;

  host_visited_matrix = visited_matrix;
  // size_t num_vertices_check = 0;
  for (vertex_type v = 0; v < G.num_nodes(); v++) {
    for(size_t color_set = 0; color_set < unrounded_color_dim; color_set++){
      // Filter colors that are not in the reduced frontier
      color_type colors = host_visited_matrix[(v * color_dim) + color_set] & (~host_reduced_frontier[color_set]);
      if(colors){
        // num_vertices_check += numColors(colors);
        size_t offset = color_set * color_size;
        while (colors != 0) {
          color_type color = getNextColor(colors);
          // if(offset >= num_colors){
          //   std::cout << "Color = " << color << "\n";
          //   assert(color < num_colors && "Color out of bounds");
          // }

          // O = map color to appropriate output vector
          output_mappings[offset + color]->push_back(v);

          colors = clearColor(colors, color);
        }
      }
    }
    // std::cout << "Done" << "\n";
  }
  // Add up out number of vertices in O + color for each color
  // size_t num_vertices = 0;
  // for (size_t color = 0; color < num_colors; color++) {
  //   num_vertices += (O + color)->size();
  // }
  // std::cout << "Number of vertices = " << num_vertices << " num_check = " << num_vertices_check << std::endl;
  // assert(num_vertices >= num_colors && "Number of vertices is incorrect");
  // std::cout << "Done Removing" << "\n";

  #ifdef PROFILE_OVERHEAD
  end = std::chrono::high_resolution_clock::now();
  auto end_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  // setup_time, traversal_time, end_time
  bfs_ctx.setup_time.push_back(setup_time.count());
  bfs_ctx.traversal_time.push_back(traversal_time.count());
  bfs_ctx.end_time.push_back(end_time.count());
  bfs_ctx.num_traversals.push_back(num_traversals);
  bfs_ctx.remaining_count.push_back(remaining_count);
  bfs_ctx.num_colors.push_back(num_colors);
  #endif // PROFILE_OVERHEAD

  return num_remaining;
}
#endif // PAUSE_AND_RESUME

template <typename GraphTy, typename DeviceContextTy, typename diff_model_tag>
void GPUCalculateDegrees(GraphTy &G, const DeviceContextTy &Context, diff_model_tag &&tag,
                         int &small_neighbors, int &medium_neighbors, int &large_neighbors,
                         int &extreme_neighbors) {
  using DeviceGraphTy = typename DeviceContextTy::device_graph_type;
  using vertex_type = typename GraphTy::vertex_type;
  
  GPU<RUNTIME>::set_device(Context.gpu_id);

  auto d_graph = Context.d_graph;
  auto d_index = d_graph->d_index_;
  // Count number of neighbors for each vertex
  thrust::device_vector<vertex_type> numTotalNeighbors(G.num_nodes());
  thrust::transform(thrust::device, thrust::make_counting_iterator<vertex_type>(0),
                    thrust::make_counting_iterator<vertex_type>(G.num_nodes()),
                    numTotalNeighbors.begin(),
                    [d_index] __device__  (const vertex_type &FE) {
                      return d_index[FE + 1] - d_index[FE];
                    });
  small_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [small_threshold = SMALL_THRESHOLD] __device__ (const vertex_type &FE) {
        return FE < small_threshold;
      });
  medium_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [small_threshold = SMALL_THRESHOLD,
       medium_threshold = MEDIUM_THRESHOLD] __device__ (const vertex_type &FE) {
        return FE >= small_threshold && FE < medium_threshold;
      });
  large_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [medium_threshold = MEDIUM_THRESHOLD,
       large_threshold = LARGE_THRESHOLD] __device__ (const vertex_type &FE) {
        return FE >= medium_threshold && FE < large_threshold;
      });
  extreme_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [large_threshold = LARGE_THRESHOLD] __device__ (const vertex_type &FE) {
        return FE >= large_threshold;
      });
  // Print out the number of vertices with each number of neighbors
  // std::cout << "Small neighbors: " << small_neighbors << std::endl;
  // std::cout << "Medium neighbors: " << medium_neighbors << std::endl;
  // std::cout << "Large neighbors: " << large_neighbors << std::endl;
  // std::cout << "Extreme neighbors: " << extreme_neighbors << std::endl;
}
}  // namespace ripples

#endif
