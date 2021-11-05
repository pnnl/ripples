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

#include "ripples/gpu/gpu_graph.h"
#include "ripples/cuda/cuda_supported_graphs.h"
#include "ripples/gpu/gpu_runtime_trait.h"

namespace ripples {
template <GPURuntime R, typename GraphTy>
__global__ void build_graph_kernel(
    typename gpu_graph<R, GraphTy>::vertex_t *d_edges,
    typename gpu_graph<R, GraphTy>::weight_t *d_weights,
    typename gpu_graph<R, GraphTy>::vertex_t *d_index,
    typename GraphTy::edge_type *d_src_weighted_edges,
    typename GraphTy::edge_type **d_src_index, size_t num_nodes) {
  using vertex_t = typename gpu_graph<R, GraphTy>::vertex_t;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    vertex_t first = d_src_index[tid] - d_src_index[0];
    vertex_t last = d_src_index[tid + 1] - d_src_index[0];
    if (tid == 0) d_index[0] = 0;
    d_index[tid + 1] = last;
    for (; first < last; ++first) {
      d_edges[first] = d_src_weighted_edges[first].vertex;
      d_weights[first] = d_src_weighted_edges[first].weight;
    }
  }
}

//! \brief Construct a device-side CUDA Graph from a host-side Graph.
//!
//! \param hg The host-side Graph to be mirrored.
template <GPURuntime R, typename GraphTy>
gpu_graph<R, GraphTy> *make_gpu_graph(const GraphTy &hg) {
  // allocate
  auto res = new gpu_graph<R, GraphTy>();
  GPU<R>::device_malloc(
      &res->d_edges_,
      hg.num_edges() * sizeof(typename gpu_graph<R, GraphTy>::vertex_t));
  GPU<R>::device_malloc(
      &res->d_weights_,
      hg.num_edges() * sizeof(typename gpu_graph<R, GraphTy>::weight_t));
  GPU<R>::device_malloc(
      &res->d_index_,
      (hg.num_nodes() + 1) * sizeof(typename gpu_graph<R, GraphTy>::vertex_t));

  // copy graph to device
  using destination_type = typename GraphTy::edge_type;
  destination_type *d_weighted_edges;
  GPU<R>::device_malloc(&d_weighted_edges,
                        hg.num_edges() * sizeof(destination_type));
  GPU<R>::h2d(d_weighted_edges, hg.csr_edges(),
              hg.num_edges() * sizeof(destination_type));
  destination_type **d_index;
  GPU<R>::device_malloc(&d_index,
                        (hg.num_nodes() + 1) * sizeof(destination_type *));
  GPU<R>::h2d(d_index, hg.csr_index(),
              (hg.num_nodes() + 1) * sizeof(destination_type *));

  // build
#if defined(RIPPLES_ENABLE_CUDA)
  constexpr int block_size = 512;
  auto n_blocks = (hg.num_nodes() + block_size - 1) / block_size;
  build_graph_kernel<R, GraphTy>
      <<<n_blocks, block_size>>>(res->d_edges_, res->d_weights_, res->d_index_,
                                 d_weighted_edges, d_index, hg.num_nodes());
#elif defined(RIPPLES_ENABLE_HIP)
  constexpr int block_size = 256;
  dim3 threads(block_size, 1, 1);
  dim3 blocks((hg.num_nodes() + block_size - 1) / block_size, 1, 1);
  hipLaunchKernelGGL((build_graph_kernel<R, GraphTy>), blocks, threads, 0, 0,
                     res->d_edges_, res->d_weights_, res->d_index_,
                     d_weighted_edges, d_index, hg.num_nodes());
#else
#error "Unsupported GPU runtime"
#endif

  GPU<R>::device_free(d_weighted_edges);
  GPU<R>::device_free(d_index);

  return res;
}

template <>
gpu_graph<CUDA, IMMGraphTy> *make_gpu_graph<CUDA, IMMGraphTy>(
    const IMMGraphTy &);
template <>
gpu_graph<CUDA, HCGraphTy> *make_gpu_graph<CUDA, HCGraphTy>(const HCGraphTy &);
template <>
gpu_graph<HIP, IMMGraphTy> *make_gpu_graph<HIP, IMMGraphTy>(const IMMGraphTy &);
template <>
gpu_graph<HIP, HCGraphTy> *make_gpu_graph<HIP, HCGraphTy>(const HCGraphTy &);
}  // namespace ripples
