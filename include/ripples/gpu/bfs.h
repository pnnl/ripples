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

#define EXPERIMENTAL_SCAN_BFS

#define FRONTIER_PROFILE

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
template <typename GraphTy>
struct Frontier {
  thrust::device_vector<typename GraphTy::vertex_type> v;
  thrust::device_vector<typename GraphTy::vertex_type> color;
  thrust::device_vector<typename GraphTy::weight_type> weight;
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

//! \brief Get a color mask from a color ID.
template <typename T>
__host__ __device__ T getMaskFromColor(T color) {
  return 1ul << ((sizeof(T) * 8 - 1) - color);
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
        [](const vertex_type &c) { return __popc(c); }, 0,
        thrust::plus<vertex_type>());
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

template <typename VertexTy>
struct GPUIndependentCascadeScan {
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
      VertexTy *frontier_addr = frontier_matrix + vertex;
      atomicOr(frontier_addr, newColors);
    }
  }

  VertexTy *visited_matrix;
  VertexTy *frontier_matrix;
};

template<typename VertexTy>
struct notZero
{
	__host__ __device__
    bool operator()(thrust::tuple<VertexTy, VertexTy> T)
	{
		VertexTy v = thrust::get<1>(T);
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
  uint32_t color = 1 << (NumColors - 1);
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    frontier.v.push_back(*itr);
    frontier.color.push_back(color);
    host_visited_matrix[*itr] |= color;
  }
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

  while (frontier.v.size() != 0) {
    // #ifdef FRONTIER_PROFILE
    // size_t vertex_size = frontier.v.size();
    // GPU<RUNTIME>::device_sync();
    // auto start_scatter = std::chrono::high_resolution_clock::now();
    // #endif
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

    // #ifdef FRONTIER_PROFILE
    // GPU<RUNTIME>::device_sync();
    // auto end_scatter = std::chrono::high_resolution_clock::now();
    // #endif

    #ifdef FRONTIER_PROFILE
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
        [](const vertex_type &c) { return __popc(c); }, 0,
        thrust::plus<vertex_type>());
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

    #ifdef FRONTIER_PROFILE
    GPU<RUNTIME>::device_sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto time_scatter = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_scatter - start_scatter);
    profile_vector.push_back({edge_size, time.count(), num_colors, vertex_size, time_scatter.count(), max_outdegree});
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

}  // namespace ripples

#endif
