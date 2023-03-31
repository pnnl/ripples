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

#if defined(RIPPLES_ENABLE_CUDA)
#include "ripples/cuda/cuda_utils.h"
#endif

namespace ripples {

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
}  // namespace ripples
#endif
