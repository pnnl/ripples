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
#include "thrust/device_vector.h"
#include "thrust/for_each.h"
#include "thrust/host_vector.h"
#include "thrust/inner_product.h"
#include "thrust/random.h"
#include "thrust/reduce.h"
#include "thrust/transform_scan.h"

#if defined(RIPPLES_ENABLE_CUDA)
#define RUNTIME CUDA
#elif defined(RIPPLES_ENABLE_HIP)
#define RUNTIME HIP
#endif

#include "ripples/gpu/bfs_kernels.h"

#define EXPERIMENTAL_SCAN_BFS

#define HIERARCHICAL

#define REORDERING

#define SORTING

// #define FULL_COLORS_MOTIVATION

// #define FRONTIER_PROFILE

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
  thrust::device_vector<typename GraphTy::vertex_type> v;
  thrust::device_vector<ColorTy> color;
  thrust::device_vector<typename GraphTy::vertex_type> offset;
};

//! \brief Get the next color ID from the color mask.
template <typename T>
__host__ __device__ T getNextColor(T color);

template <>
__host__ __device__ uint32_t getNextColor(uint32_t color) {
  return __builtin_clz(color);
}

template <>
__host__ __device__ uint64_t getNextColor(uint64_t color) {
  return __builtin_clzl(color);
}

template <typename T>
__host__ __device__ T numColors(T color);

template <>
__host__ __device__ uint32_t numColors(uint32_t color) {
  return __builtin_popcount(color);
}

template <>
__host__ __device__ uint64_t numColors(uint64_t color) {
  return __builtin_popcountl(color);
}

//! \brief Get a color mask from a color ID.
template <typename T>
__host__ __device__ T getMaskFromColor(T color) {
  return (T)1 << ((sizeof(T) * 8 - 1) - color);
}

//! \brief Remove a color from the mask of colors.
template <typename T>
__host__ __device__ T clearColor(T colors, T color) {
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
#else
#define SMALL_THRESHOLD 64
#endif
#define MEDIUM_THRESHOLD 256
#define LARGE_THRESHOLD 65536
#define NUM_LEVELS 4

template <typename GraphTy, typename DeviceContextTy, typename SItrTy,
          typename OItrTy, typename diff_model_tag, typename ColorTy = typename GraphTy::vertex_type>
void GPUBatchedTieredQueueBFS(GraphTy &G, const DeviceContextTy &Context, SItrTy B,
                   SItrTy E, OItrTy O, diff_model_tag &&tag, int small_neighbors,
                   int medium_neighbors, int large_neighbors, int extreme_neighbors,
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

  std::vector<typename GPU<RUNTIME>::stream_type> streams(NUM_LEVELS, GPU<RUNTIME>::create_stream());

  thrust::device_vector<color_type> frontier_matrix(G.num_nodes(), 0);
  thrust::device_vector<color_type> visited_matrix(G.num_nodes());
  thrust::host_vector<color_type> host_visited_matrix(G.num_nodes(), 0);

  auto d_graph = Context.d_graph;
  auto d_index = d_graph->d_index_;
  auto d_edge = d_graph->d_edges_;
  auto d_weight = d_graph->d_weights_;

  // GPU<RUNTIME>::device_sync();
  // std::cout << "Device Vectors: " << Context.gpu_id << std::endl;

  // Perform setup, initialize first set of visited vertices
  Frontier<GraphTy, ColorTy> frontier, new_frontier;
  FrontierHierarch<GraphTy, ColorTy> small_frontier, medium_frontier, large_frontier, extreme_frontier;
  // Resize all frontier queues
  small_frontier.v.resize(small_neighbors);
  small_frontier.color.resize(small_neighbors);
  small_frontier.offset.resize(small_neighbors);
  medium_frontier.v.resize(medium_neighbors);
  medium_frontier.color.resize(medium_neighbors);
  medium_frontier.offset.resize(medium_neighbors);
  large_frontier.v.resize(large_neighbors);
  large_frontier.color.resize(large_neighbors);
  large_frontier.offset.resize(large_neighbors);
  extreme_frontier.v.resize(extreme_neighbors);
  extreme_frontier.color.resize(extreme_neighbors);
  extreme_frontier.offset.resize(extreme_neighbors);
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
      thrust::plus<int>(), thrust::not_equal_to<vertex_type>());
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

  thrust::device_vector<vertex_type> numNeighbors(frontier.v.size() + 1, 0);
  thrust::device_vector<vertex_type> numNeighborsSeparate(frontier.v.size() + 1, 0);
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
          G.num_nodes());
    if(mediumWork > 0)
      warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<mediumWork, 32, 0, streams[1]>>>(d_index,
          d_edge, d_weight,
          medium_frontier_v_ptr, medium_frontier_color_ptr, medium_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          G.num_nodes());
    if(largeWork > 0)
      warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<largeWork, 256, 0, streams[2]>>>(d_index,
          d_edge, d_weight,
          large_frontier_v_ptr, large_frontier_color_ptr, large_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          G.num_nodes());
    if(extremeWork > 0)
      warp_block_scatter_kernel<RUNTIME, GraphTy, ColorTy>
        <<<extremeWork, 1024, 0, streams[3]>>>(d_index,
          d_edge, d_weight,
          extreme_frontier_v_ptr, extreme_frontier_color_ptr, extreme_frontier_offset_ptr,
          new_frontier_v_ptr, new_frontier_color_ptr, new_frontier_weight_ptr,
          G.num_nodes());
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
                    [d_index](const vertex_type &FE) {
                      return d_index[FE + 1] - d_index[FE];
                    });
  small_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [small_threshold = SMALL_THRESHOLD](const vertex_type &FE) {
        return FE < small_threshold;
      });
  medium_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [small_threshold = SMALL_THRESHOLD,
       medium_threshold = MEDIUM_THRESHOLD](const vertex_type &FE) {
        return FE >= small_threshold && FE < medium_threshold;
      });
  large_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [medium_threshold = MEDIUM_THRESHOLD,
       large_threshold = LARGE_THRESHOLD](const vertex_type &FE) {
        return FE >= medium_threshold && FE < large_threshold;
      });
  extreme_neighbors = thrust::count_if(
      thrust::device, numTotalNeighbors.begin(), numTotalNeighbors.end(),
      [large_threshold = LARGE_THRESHOLD](const vertex_type &FE) {
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
