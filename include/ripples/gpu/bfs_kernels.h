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

#ifndef RIPPLES_GPU_LT_KERNEL_H
#define RIPPLES_GPU_LT_KERNEL_H

#include "ripples/gpu/generate_rrr_sets.h"
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"
#include "ripples/gpu/bfs.h"

#if defined(RIPPLES_ENABLE_CUDA)
#include "ripples/cuda/cuda_utils.h"
#endif

namespace ripples {

  // Override popcount for 32 or 64-bit integers
__device__ __forceinline__ int popcount(uint32_t x) {
  return __popc(x);
}

__device__ __forceinline__ int popcount(uint64_t x) {
  return __popcll(x);
}


__device__ __forceinline__ uint32_t bit_reverse(uint32_t x) {
  return __brev(x);
}

__device__ __forceinline__ uint64_t bit_reverse(uint64_t x) {
  return __brevll(x);
}

__device__ __forceinline__ int clz(uint32_t x) {
  return __clz(x);
}

__device__ __forceinline__ int clz(uint64_t x) {
  return __clzll(x);
}

__device__ __forceinline__ int ffs(uint32_t x) {
  return __ffs((uint32_t)x);
}

__device__ __forceinline__ unsigned int ffs(uint64_t x) {
  return __ffsll((unsigned long long int)x);
}

//! \brief Get a color mask from a color ID.
template <typename T>
__device__ __forceinline__ T getColorMask(T color) {
  return (T)1 << ((sizeof(T) * 8 - 1) - color);
}

//! \brief Remove a color from the mask of colors.
template <typename T>
__device__ __forceinline__ T removeColor(T colors, T color) {
  return colors & (~getColorMask(color));
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy =  typename GraphTy::vertex_type>
__global__ void thread_scatter_kernel(
    typename GraphTy::vertex_type *index,
    typename GraphTy::vertex_type *edges,
    typename GraphTy::weight_type *weights,
    typename GraphTy::vertex_type *vertex,
    ColorTy *colors,
    typename GraphTy::vertex_type *output_location,
    typename GraphTy::vertex_type *output_vertex,
    ColorTy *output_colors,
    typename GraphTy::weight_type *output_weights,
    const size_t num_nodes) {
  using vertex_type = typename GraphTy::vertex_type;
  using color_type = ColorTy;

  // Thread-parallel scatter
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // tid = vertex id
  if (tid < num_nodes) {
    vertex_type vertex_id = vertex[tid];
    color_type color = colors[tid];
    vertex_type output_offset = output_location[tid];
    vertex_type start = index[vertex_id];
    vertex_type end = index[vertex_id + 1];
    for (vertex_type i = 0; i < end-start; i++) {
      output_vertex[output_offset + i] = edges[start + i];
      output_colors[output_offset + i] = color;
      output_weights[output_offset + i] = weights[start + i];
    }
  }
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy =  typename GraphTy::vertex_type>
__global__ void warp_block_scatter_kernel(
    typename GraphTy::vertex_type *index,
    typename GraphTy::vertex_type *edges,
    typename GraphTy::weight_type *weights,
    typename GraphTy::vertex_type *vertex,
    ColorTy *colors,
    typename GraphTy::vertex_type *output_location,
    typename GraphTy::vertex_type *output_vertex,
    ColorTy *output_colors,
    typename GraphTy::weight_type *output_weights,
    const size_t num_nodes) {
  using vertex_type = typename GraphTy::vertex_type;
  using color_type = ColorTy;

  // Warp/block-parallel scatter  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (bid < num_nodes) {
    vertex_type vertex_id = vertex[bid];
    color_type color = colors[bid];
    vertex_type output_offset = output_location[bid];
    vertex_type start = index[vertex_id];
    vertex_type end = index[vertex_id + 1];
    // Write the edges and colors to the output array
    for (vertex_type i = tid; i < end-start; i += blockDim.x) {
      output_vertex[output_offset + i] = edges[start + i];
      output_colors[output_offset + i] = color;
      output_weights[output_offset + i] = weights[start + i];
    }
  }
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy,
          size_t NumBlocks>
__global__ void color_thread_scatter_kernel(
    const typename GraphTy::vertex_type * __restrict__ index,
    const typename GraphTy::vertex_type * __restrict__ edges,
    const typename GraphTy::weight_type * __restrict__ weights,
    const typename GraphTy::vertex_type * __restrict__ vertex,
    const ColorTy * __restrict__ colors,
    const typename GraphTy::vertex_type * __restrict__ output_location,
    typename GraphTy::vertex_type * __restrict__ output_vertex,
    ColorTy * __restrict__ output_colors,
    typename GraphTy::weight_type * __restrict__ output_weights,
    const size_t num_nodes) {
  using vertex_type = typename GraphTy::vertex_type;
  using color_type = ColorTy;

  // Chunk-parallel scatter
  // Each group of NumBlocks threads will work on a different vertex to scatter
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  // Figure out which color set we are working on
  const int input_id = global_id / NumBlocks; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int color_set_id = global_id % NumBlocks; // 0, 1, 2, 3, ..., 0, 1, 2, 3, ...
  const int edges_per_block = (blockDim.x + NumBlocks - 1) / NumBlocks;
  if (input_id < num_nodes) {
    vertex_type vertex_id = vertex[input_id];
    color_type color = colors[input_id * NumBlocks + color_set_id];
    vertex_type output_offset = output_location[input_id];
    vertex_type start = index[vertex_id];
    vertex_type end = index[vertex_id + 1];
    // Write the edges and colors to the output array
    for (vertex_type i = color_set_id; i < end-start; i+=edges_per_block) {
      output_vertex[output_offset + i] = edges[start + i];
      output_weights[output_offset + i] = weights[start + i];
    }
    for (vertex_type i = 0; i < end-start; ++i) {
      output_colors[(output_offset + i) * NumBlocks + color_set_id] = color;
    }
  }
}


template <GPURuntime R,
          typename GraphTy,
          typename ColorTy,
          size_t NumBlocks>
__global__ void color_set_scatter_kernel(
    const typename GraphTy::vertex_type * __restrict__ index,
    const typename GraphTy::vertex_type * __restrict__ edges,
    const typename GraphTy::weight_type * __restrict__ weights,
    const typename GraphTy::vertex_type * __restrict__ vertex,
    const ColorTy * __restrict__ colors,
    const typename GraphTy::vertex_type * __restrict__ output_location,
    typename GraphTy::vertex_type * __restrict__ output_vertex,
    ColorTy * __restrict__ output_colors,
    typename GraphTy::weight_type * __restrict__ output_weights,
    const size_t num_nodes) {
  using vertex_type = typename GraphTy::vertex_type;
  using color_type = ColorTy;

  // Warp/block-parallel scatter  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  // Figure out which color set we are working on
  const int input_id = global_id / NumBlocks; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int color_set_id = global_id % NumBlocks; // 0, 1, 2, 3, ..., 0, 1, 2, 3, ...
  const int edges_per_block = (blockDim.x + NumBlocks - 1) / NumBlocks;
  if (bid < num_nodes) {
    vertex_type vertex_id = vertex[bid];
    color_type color = colors[bid * NumBlocks + color_set_id];
    vertex_type output_offset = output_location[bid];
    vertex_type start = index[vertex_id];
    vertex_type end = index[vertex_id + 1];
    // Write the weights and edges to the output array
    for (vertex_type i = tid; i < end-start; i += blockDim.x) {
      output_vertex[output_offset + i] = edges[start + i];
      output_weights[output_offset + i] = weights[start + i];
    }
    for (vertex_type i = tid / NumBlocks; i < end-start; i += edges_per_block) {
      output_colors[(output_offset + i) * NumBlocks + color_set_id] = color;
    }
  }
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy,
          size_t NumBlocks>
__global__ void sim_step_thread_kernel(
    const typename GraphTy::vertex_type * __restrict__ vertex_ptr,
    const ColorTy * __restrict__ color_ptr,
    const typename GraphTy::weight_type * __restrict__ weight_ptr,
    ColorTy * __restrict__ visited_matrix,
    ColorTy * __restrict__ frontier_matrix,
    const size_t num_edges) {
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = ColorTy;

  // Thread-parallel scatter  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  uint64_t seed = clock() + global_id;
  thrust::minstd_rand generator(seed*seed+19283);
  thrust::uniform_real_distribution<float> value;
  color_type newColors = 0;
  // Figure out which color set we are working on
  const int input_id = global_id / NumBlocks; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int color_set_id = global_id % NumBlocks; // 0, 1, 2, 3, ..., 0, 1, 2, 3, ...
  const int edges_per_block = (blockDim.x + NumBlocks - 1) / NumBlocks;
  if (input_id < num_edges) {
    vertex_type vertex = vertex_ptr[input_id];
    weight_type weight = weight_ptr[input_id];
    color_type colors = color_ptr[global_id];
    // Write the weights and edges to the output array
    while(colors != 0) {
      color_type color = clz(colors);
      color_type mask = getColorMask(color);
      if (value(generator) <= weight) {
        newColors |= mask;
      }
      colors = removeColor(colors, color);
    }
    if(newColors != 0) {
      const vertex_type offset = vertex * NumBlocks + color_set_id;
      ColorTy *visited_addr = visited_matrix + offset;
      color_type old = atomicOr(visited_addr, newColors);
      color_type frontierColors = old ^ newColors;
      // If new, unvisited values were added to the visited matrix, add them to the frontier matrix
      if(frontierColors != 0) {
        ColorTy *frontier_addr = frontier_matrix + offset;
        atomicOr(frontier_addr, newColors);
      }
    }
  }
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy,
          size_t NumBlocks>
__global__ void sim_step_block_kernel(
    const typename GraphTy::vertex_type * __restrict__ vertex_ptr,
    const ColorTy * __restrict__ color_ptr,
    const typename GraphTy::weight_type * __restrict__ weight_ptr,
    ColorTy * __restrict__ visited_matrix,
    ColorTy * __restrict__ frontier_matrix,
    const size_t num_edges) {
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = ColorTy;


  // Warp/block-parallel scatter  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  uint64_t seed = clock() + global_id;
  thrust::minstd_rand generator(seed*seed+19283);
  thrust::uniform_real_distribution<float> value;
  bool color_mask = 0;
  // Number of colors per color_type
  const int num_colors = sizeof(color_type) * 8;
  // Calculate number of edges
  const int threads_per_color_set = sizeof(color_type) * 8 * NumBlocks;
  // Figure out which color set we are working on
  const int input_id = global_id / threads_per_color_set; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int color_bit = global_id % num_colors; // 0, 1, 2, 3, ..., 0, 1, 2, 3, ...
  const int color_set_id = global_id / num_colors; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int edges_per_block = (blockDim.x + NumBlocks - 1) / NumBlocks;
  if (input_id < num_edges) {
    vertex_type vertex = vertex_ptr[input_id];
    weight_type weight = weight_ptr[input_id];
    color_type colors = color_ptr[color_set_id];
    // Write the weights and edges to the output array
    if (colors & (1 << color_bit)) {
      if (value(generator) <= weight) {
        color_mask = true;
      }
    }
    color_type newColors = __ballot(color_mask);
    if(tid % warpSize == 0){
      if(newColors != 0) {
        const vertex_type offset = vertex * NumBlocks + color_set_id;
        ColorTy *visited_addr = visited_matrix + offset;
        color_type old = atomicOr(visited_addr, newColors);
        color_type frontierColors = old ^ newColors;
        // If new, unvisited values were added to the visited matrix, add them to the frontier matrix
        if(frontierColors != 0) {
          ColorTy *frontier_addr = frontier_matrix + offset;
          atomicOr(frontier_addr, newColors);
        }
      }
    }
  }
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy>
__global__ void fused_color_thread_scatter_kernel(
    const typename GraphTy::vertex_type * __restrict__ index,
    const typename GraphTy::vertex_type * __restrict__ edges,
    const typename GraphTy::weight_type * __restrict__ weights,
    const typename GraphTy::vertex_type * __restrict__ vertex,
    const ColorTy * __restrict__ colors,
    ColorTy * __restrict__ frontier_matrix,
    const size_t num_nodes,
    const size_t color_dim) {
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = ColorTy;

  // Chunk-parallel scatter
  // Each group of color_dim threads will work on a different vertex to scatter
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  uint64_t seed = clock64() + global_id;
  // uint64_t seed = 0xFFFFFFFFFFFFFFFF;
  thrust::minstd_rand generator(seed*seed+19283);
  thrust::uniform_real_distribution<float> value;
  // Figure out which color set we are working on
  const int input_id = global_id / color_dim; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int color_set_id = global_id % color_dim; // 0, 1, 2, 3, ..., 0, 1, 2, 3, ...
  if (input_id < num_nodes) {
    vertex_type vertex_id = vertex[input_id];
    const color_type full_colors = colors[global_id];
    vertex_type start_edge = index[vertex_id];
    vertex_type end_edge = index[vertex_id + 1];
    // Write the edges and colors to the output array
    for (vertex_type i = 0; i < end_edge - start_edge; ++i) {
      color_type full_colors_temp = full_colors;
      vertex_type edge = edges[start_edge + i];
      weight_type weight = weights[start_edge + i];
      color_type color_mask = 0;
      while(full_colors_temp != 0) {
        color_type color = clz(full_colors_temp);
        color_type mask = getColorMask(color);
        if (value(generator) <= weight) {
          color_mask |= mask;
        }
        full_colors_temp = removeColor(full_colors_temp, color);
      }
      if(color_mask != 0) {
        const vertex_type offset = edge * color_dim + color_set_id;
        ColorTy *frontier_addr = frontier_matrix + offset;
        atomicOr(frontier_addr, color_mask);
      }
    }
  }
}

template <GPURuntime R,
          typename GraphTy,
          typename ColorTy>
__global__ void fused_color_set_scatter_kernel(
    const typename GraphTy::vertex_type * __restrict__ index,
    const typename GraphTy::vertex_type * __restrict__ edges,
    const typename GraphTy::weight_type * __restrict__ weights,
    const typename GraphTy::vertex_type * __restrict__ vertex,
    const ColorTy * __restrict__ colors,
    ColorTy * __restrict__ frontier_matrix,
    const size_t num_nodes,
    const size_t color_dim) {
  using vertex_type = typename GraphTy::vertex_type;
  using weight_type = typename GraphTy::weight_type;
  using color_type = ColorTy;

  // Warp/block-parallel scatter  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  uint64_t seed = clock64() + global_id;
  // uint64_t seed = 0xFFFFFFFFFFFFFFFF;
  thrust::minstd_rand generator(seed*seed+19283);
  thrust::uniform_real_distribution<float> value;
  // Figure out which color set we are working on
  const int color_id = tid % color_dim; // 0, 1, 2, 3, ..., 0, 1, 2, 3, ...
  const int color_set = tid / color_dim; // 0, 0, 0, 0, ..., 1, 1, 1, 1, ...
  const int edges_per_block = blockDim.x / color_dim;
  if (bid < num_nodes) {
    vertex_type vertex_id = vertex[bid];
    color_type full_colors = colors[bid * color_dim + color_id];
    vertex_type start_edge = index[vertex_id];
    vertex_type end_edge = index[vertex_id + 1];
    // Write the weights and edges to the output array
    for (vertex_type i = color_set; i < end_edge - start_edge; i += edges_per_block) {
      color_type full_colors_temp = full_colors;
      vertex_type edge = edges[start_edge + i];
      weight_type weight = weights[start_edge + i];
      color_type color_mask = 0;
      while(full_colors_temp != 0) {
        color_type color = clz(full_colors_temp);
        color_type mask = getColorMask(color);
        if (value(generator) <= weight) {
          color_mask |= mask;
        }
        full_colors_temp = removeColor(full_colors_temp, color);
      }
      if(color_mask != 0) {
        const vertex_type offset = edge * color_dim + color_id;
        ColorTy *frontier_addr = frontier_matrix + offset;
        atomicOr(frontier_addr, color_mask);
      }
    }
  }
}

// #if defined(RIPPLES_ENABLE_HIP) && (HIP_VERSION <= 5 * 10000000 + 1 * 100000 + 0)
#if defined(RIPPLES_ENABLE_HIP)
// Workaround for HIP 5.1.0 divergent warp lanes
template <GPURuntime R,
          typename GraphTy,
          typename ColorTy,
          typename WarpMaskTy>
__global__ void build_frontier_queues_kernel(
    const typename GraphTy::vertex_type * __restrict__ index,
    typename GraphTy::vertex_type * __restrict__ small_frontier,
    typename GraphTy::vertex_type * __restrict__ medium_frontier,
    typename GraphTy::vertex_type * __restrict__ large_frontier,
    typename GraphTy::vertex_type * __restrict__ extreme_frontier,
    ColorTy * __restrict__ small_color,
    ColorTy * __restrict__ medium_color,
    ColorTy * __restrict__ large_color,
    ColorTy * __restrict__ extreme_color,
    ColorTy * __restrict__ visited_matrix,
    const ColorTy * __restrict__ frontier_matrix,
    typename GraphTy::vertex_type * __restrict__ frontier_offsets,
    const size_t num_nodes,
    const size_t num_colors,
    const size_t color_dim) {
  using vertex_type = typename GraphTy::vertex_type;
  using color_type = ColorTy;

  constexpr size_t color_size = sizeof(color_type) * 8;
  constexpr int frontier_bins = 4;

  // Warp/block-parallel scatter  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  const int warp_tid = tid % warpSize;
  const int color_set = warp_tid / color_dim;
  const int color_id = warp_tid % color_dim;
  const vertex_type vertex_id = global_id / color_dim;
  int frontier_bin = -1;
  color_type frontier_colors = 0;
  if (vertex_id < num_nodes){
    color_type old_visited = visited_matrix[global_id];
    color_type new_visited = frontier_matrix[global_id];
    frontier_colors = old_visited ^ new_visited;
    visited_matrix[global_id] = new_visited;
  }
  // __warp tid in __ballot returns least significant bit as 0th lane
  WarpMaskTy frontier_mask = __ballot(frontier_colors != 0);
  // Create bitmask of 1 values for each color in the warp
  WarpMaskTy warp_color_mask = (((WarpMaskTy)1 << color_dim) - 1) << (color_set * color_dim);
  // Create bitmask of threads in warp that have a frontier vertex
  WarpMaskTy frontier_mask_filtered = frontier_mask & warp_color_mask;
  WarpMaskTy warp_offset_mask = ((WarpMaskTy)1 << color_set * color_dim) - 1;
  vertex_type *vertex_frontier;
  ColorTy *color_frontier;

  if(frontier_mask_filtered){
    // Retrieve outdegree
    vertex_type start = index[vertex_id];
    vertex_type end = index[vertex_id + 1];
    vertex_type outdegree = end - start;
    // Find block-wide offsets for each frontier queue
    if(outdegree < 64/color_dim){
      vertex_frontier = small_frontier;
      color_frontier = small_color;
      frontier_bin = 0;
    } else if(outdegree < 256/color_dim){
      vertex_frontier = medium_frontier;
      color_frontier = medium_color;
      frontier_bin = 1;
    } else if(outdegree < 65536/color_dim){
      vertex_frontier = large_frontier;
      color_frontier = large_color;
      frontier_bin = 2;
    } else {
      vertex_frontier = extreme_frontier;
      color_frontier = extreme_color;
      frontier_bin = 3;
    }
    // Share outdegree with warps in warp_color_mask
  }
  #pragma unroll
  for(int i = 0; i < frontier_bins; i++){
    vertex_type offset;
    WarpMaskTy active_threads = __ballot(color_id == 0 && frontier_bin == i);
    uint32_t thread_leader = ffs(active_threads) - 1;
    WarpMaskTy offset_mask = active_threads & warp_offset_mask;
    if(warp_tid == thread_leader){
      offset = atomicAdd(frontier_offsets + i, popcount(active_threads));
    }
    offset = __shfl(offset, thread_leader);
    offset += popcount(offset_mask);
    if(frontier_bin == i){
      vertex_frontier[offset] = vertex_id;
      offset *= color_dim;
      color_frontier[offset + color_id] = frontier_colors;
    }
  }
}
#else
template <GPURuntime R,
          typename GraphTy,
          typename ColorTy,
          typename WarpMaskTy>
__global__ void build_frontier_queues_kernel(
    const typename GraphTy::vertex_type * __restrict__ index,
    const typename GraphTy::vertex_type * __restrict__ edges,
    typename GraphTy::vertex_type * __restrict__ small_frontier,
    typename GraphTy::vertex_type * __restrict__ medium_frontier,
    typename GraphTy::vertex_type * __restrict__ large_frontier,
    typename GraphTy::vertex_type * __restrict__ extreme_frontier,
    ColorTy * __restrict__ small_color,
    ColorTy * __restrict__ medium_color,
    ColorTy * __restrict__ large_color,
    ColorTy * __restrict__ extreme_color,
    ColorTy * __restrict__ visited_matrix,
    const ColorTy * __restrict__ frontier_matrix,
    typename GraphTy::vertex_type * __restrict__ frontier_offsets,
    const size_t num_nodes,
    const size_t num_colors,
    const size_t color_dim) {
  using vertex_type = typename GraphTy::vertex_type;
  using color_type = ColorTy;

  constexpr size_t color_size = sizeof(color_type) * 8;

  // Warp/block-parallel scatter  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int global_id = tid + bid * blockDim.x;
  const int warp_tid = tid % warpSize;
  const int color_set = warp_tid / color_dim;
  const int color_id = warp_tid % color_dim;
  const vertex_type vertex_id = global_id / color_dim;
  if (vertex_id < num_nodes){
    color_type old_visited = visited_matrix[global_id];
    color_type new_visited = frontier_matrix[global_id];
    color_type frontier_colors = old_visited ^ new_visited;
    visited_matrix[global_id] = new_visited | old_visited;
    // __warp tid in __ballot returns least significant bit as 0th lane
    WarpMaskTy frontier_mask = __ballot(frontier_colors != 0);
    // Create bitmask of 1 values for each color in the warp
    WarpMaskTy warp_color_mask = (((WarpMaskTy)1 << color_dim) - 1) << (color_set * color_dim);
    // Create bitmask of threads in warp that have a frontier vertex
    WarpMaskTy frontier_mask_filtered = frontier_mask & warp_color_mask;
    WarpMaskTy warp_offset_mask = ((WarpMaskTy)1 << color_set * color_dim) - 1;

    if(frontier_mask_filtered){
      // Retrieve outdegree
      vertex_type start = index[vertex_id];
      vertex_type end = index[vertex_id + 1];
      vertex_type outdegree = end - start;
      // Find block-wide offsets for each frontier queue
      vertex_type offset;
      if(outdegree < 64/color_dim){
        if(color_id == 0){
          WarpMaskTy active_threads = __ballot(1);
          uint32_t thread_leader = ffs(active_threads) - 1;
          WarpMaskTy offset_mask = active_threads & warp_offset_mask;
          if(warp_tid == thread_leader){
            offset = atomicAdd(frontier_offsets, popcount(active_threads));
          }
          offset = __shfl(offset, thread_leader);
          offset += popcount(offset_mask);
          small_frontier[offset] = vertex_id;
        }
        offset = __shfl(offset, color_set * color_dim) * color_dim;
        small_color[offset + color_id] = frontier_colors;
      } else if(outdegree < 256/color_dim){
        if(color_id == 0){
          WarpMaskTy active_threads = __ballot(1);
          uint32_t thread_leader = ffs(active_threads) - 1;
          WarpMaskTy offset_mask = active_threads & warp_offset_mask;
          if(warp_tid == thread_leader){
            offset = atomicAdd(frontier_offsets + 1, popcount(active_threads));
          }
          offset = __shfl(offset, thread_leader);
          offset += popcount(offset_mask);
          medium_frontier[offset] = vertex_id;
        }
        offset = __shfl(offset, color_set * color_dim) * color_dim;
        medium_color[offset + color_id] = frontier_colors;
      } else if(outdegree < 65536/color_dim){
        if(color_id == 0){
          WarpMaskTy active_threads = __ballot(1);
          uint32_t thread_leader = ffs(active_threads) - 1;
          WarpMaskTy offset_mask = active_threads & warp_offset_mask;
          if(warp_tid == thread_leader){
            offset = atomicAdd(frontier_offsets + 2, popcount(active_threads));
          }
          offset = __shfl(offset, thread_leader);
          offset += popcount(offset_mask);
          large_frontier[offset] = vertex_id;
        }
        offset = __shfl(offset, color_set * color_dim) * color_dim;
        large_color[offset + color_id] = frontier_colors;
      } else {
        if(color_id == 0){
          WarpMaskTy active_threads = __ballot(1);
          uint32_t thread_leader = ffs(active_threads) - 1;
          WarpMaskTy offset_mask = active_threads & warp_offset_mask;
          if(warp_tid == thread_leader){
            offset = atomicAdd(frontier_offsets + 3, popcount(active_threads));
          }
          offset = __shfl(offset, thread_leader);
          offset += popcount(offset_mask);
          extreme_frontier[offset] = vertex_id;
        }
        offset = __shfl(offset, color_set * color_dim) * color_dim;
        extreme_color[offset + color_id] = frontier_colors;
      }
      // Share outdegree with warps in warp_color_mask
    }
  }
}
#endif // HIP_VERSION Check

}  // namespace ripples
#endif