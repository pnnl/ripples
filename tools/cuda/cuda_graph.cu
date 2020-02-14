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

#include <cstddef>
#include <cassert>

#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_supported_graphs.h"

namespace ripples {

template <typename GraphTy>
__global__ void build_graph_kernel(
                                   typename cuda_device_graph<GraphTy>::vertex_t *d_edges,
                                   typename cuda_device_graph<GraphTy>::weight_t *d_weights,
                                   typename cuda_device_graph<GraphTy>::vertex_t *d_index,
                                   typename GraphTy::edge_type *d_src_weighted_edges,
                                   typename GraphTy::edge_type **d_src_index, size_t num_nodes) {
  using vertex_t = typename cuda_device_graph<GraphTy>::vertex_t;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    vertex_t first = d_src_index[tid] - d_src_index[0];
    vertex_t last = d_src_index[tid + 1] - d_src_index[0];
    if(tid == 0)
      d_index[0] = 0;
    d_index[tid + 1] = last;
    for(; first < last; ++first) {
      d_edges[first] = d_src_weighted_edges[first].vertex;
      d_weights[first] = d_src_weighted_edges[first].weight;
    }
  }
}

//! \brief Construct a device-side CUDA Graph from a host-side Graph.
//!
//! \param hg The host-side Graph to be mirrored.
template<typename GraphTy>
cuda_device_graph<GraphTy> *make_cuda_graph(const GraphTy &hg){

  // allocate
  auto res = new cuda_device_graph<GraphTy>();
  cudaMalloc(&res->d_edges_,
             hg.num_edges() * sizeof(typename cuda_device_graph<GraphTy>::vertex_t));
  cudaMalloc(&res->d_weights_,
             hg.num_edges() * sizeof(typename cuda_device_graph<GraphTy>::weight_t));
  cudaMalloc(
      &res->d_index_,
      (hg.num_nodes() + 1) * sizeof(typename cuda_device_graph<GraphTy>::vertex_t));
  cuda_check(__FILE__, __LINE__);

  // copy graph to device
  using destination_type = typename GraphTy::edge_type;
  destination_type *d_weighted_edges;
  cudaMalloc(&d_weighted_edges, hg.num_edges() * sizeof(destination_type));
  cudaMemcpy(d_weighted_edges, hg.csr_edges(),
             hg.num_edges() * sizeof(destination_type), cudaMemcpyHostToDevice);
  destination_type **d_index;
  cudaMalloc(&d_index, (hg.num_nodes() + 1) * sizeof(destination_type *));
  cudaMemcpy(d_index, hg.csr_index(),
             (hg.num_nodes() + 1) * sizeof(destination_type *),
             cudaMemcpyHostToDevice);
  cuda_check(__FILE__, __LINE__);

  // build
  constexpr int block_size = 512;
  auto n_blocks = (hg.num_nodes() + block_size - 1) / block_size;
  build_graph_kernel<GraphTy><<<n_blocks, block_size>>>(res->d_edges_, res->d_weights_,
                                               res->d_index_, d_weighted_edges,
                                               d_index, hg.num_nodes());
  cuda_check(__FILE__, __LINE__);

  cudaFree(d_weighted_edges);
  cudaFree(d_index);
  cuda_check(__FILE__, __LINE__);

  return res;
}

template
cuda_device_graph<IMMGraphTy> *make_cuda_graph<IMMGraphTy>(const IMMGraphTy &);
template
cuda_device_graph<HCGraphTy> *make_cuda_graph<HCGraphTy>(const HCGraphTy &);

}
