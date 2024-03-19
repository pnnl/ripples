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
#include "ripples/gpu/gpu_supported_graphs.h"
#include "ripples/gpu/gpu_runtime_trait.h"

namespace ripples {
// Unused for host-side transformation
template <GPURuntime R, typename GraphTy>
__global__ void build_graph_kernel(
    typename gpu_graph<R, GraphTy>::vertex_t *d_edges,
    typename gpu_graph<R, GraphTy>::weight_t *d_weights,
    typename gpu_graph<R, GraphTy>::index_t *d_index,
    typename GraphTy::edge_type *d_src_weighted_edges,
    typename GraphTy::index_pointer_t *d_src_index, size_t num_nodes) {
  using index_t = typename gpu_graph<R, GraphTy>::index_t;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    index_t first = d_src_index[tid];
    index_t last = d_src_index[tid + 1];
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
      reinterpret_cast<void **>(&res->d_edges_),
      hg.num_edges() * sizeof(typename gpu_graph<R, GraphTy>::vertex_t));
  GPU<R>::device_malloc(
      reinterpret_cast<void **>(&res->d_weights_),
      hg.num_edges() * sizeof(typename gpu_graph<R, GraphTy>::weight_t));
  GPU<R>::device_malloc(
      reinterpret_cast<void **>(&res->d_index_),
      (hg.num_nodes() + 1) * sizeof(typename gpu_graph<R, GraphTy>::index_t));
#ifndef HOST_SIDE_TRANSFORMATION
  // copy graph to device
  using destination_type = typename GraphTy::edge_type;
  destination_type *d_weighted_edges;
  GPU<R>::device_malloc(reinterpret_cast<void **>(&d_weighted_edges),
                        hg.num_edges() * sizeof(destination_type));
  GPU<R>::h2d(d_weighted_edges, hg.csr_edges(),
              hg.num_edges() * sizeof(destination_type));
  destination_type **d_index;
  GPU<R>::device_malloc(reinterpret_cast<void **>(&d_index),
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

  GPU<R>::device_sync();
  GPU<R>::device_free(d_weighted_edges);
  GPU<R>::device_free(d_index);
#else // HOST_SIDE_TRANSFORMATION
  using index_t = typename gpu_graph<R, GraphTy>::index_t;
  using vertex_t = typename gpu_graph<R, GraphTy>::vertex_t;
  using weight_t = typename gpu_graph<R, GraphTy>::weight_t;
  vertex_t *h_edges = new vertex_t[hg.num_edges()];
  weight_t *h_weights = new weight_t[hg.num_edges()];
  transform_to_gpu_graph<R, GraphTy>(h_edges, h_weights, hg.csr_edges(), hg.csr_index(), hg.num_nodes());
  GPU<R>::h2d(res->d_index_, hg.csr_index(), (hg.num_nodes() + 1) * sizeof(index_t));
  GPU<R>::h2d(res->d_edges_, h_edges, hg.num_edges() * sizeof(vertex_t));
  GPU<R>::h2d(res->d_weights_, h_weights, hg.num_edges() * sizeof(weight_t));
  delete[] h_edges;
  delete[] h_weights;
#endif // !HOST_SIDE_TRANSFORMATION

  return res;
}

#if defined(RIPPLES_ENABLE_CUDA)
template gpu_graph<CUDA, IMMGraphTy> *make_gpu_graph<CUDA, IMMGraphTy>(
    const IMMGraphTy &);
template void transform_to_gpu_graph<CUDA, IMMGraphTy>(
    typename gpu_graph<CUDA, IMMGraphTy>::vertex_t *h_edges,
    typename gpu_graph<CUDA, IMMGraphTy>::weight_t *h_weights,
    typename IMMGraphTy::edge_type *h_src_weighted_edges,
    typename IMMGraphTy::index_type *h_src_index, size_t num_nodes);
template gpu_graph<CUDA, HCGraphTy> *make_gpu_graph<CUDA, HCGraphTy>(
    const HCGraphTy &);
template void transform_to_gpu_graph<CUDA, HCGraphTy>(
    typename gpu_graph<CUDA, IMMGraphTy>::vertex_t *h_edges,
    typename gpu_graph<CUDA, IMMGraphTy>::weight_t *h_weights,
    typename IMMGraphTy::edge_type *h_src_weighted_edges,
    typename IMMGraphTy::index_type *h_src_index, size_t num_nodes);
#elif defined(RIPPLES_ENABLE_HIP)
template gpu_graph<HIP, IMMGraphTy> *make_gpu_graph<HIP, IMMGraphTy>(
    const IMMGraphTy &);
template void transform_to_gpu_graph<HIP, IMMGraphTy>(
    typename gpu_graph<HIP, IMMGraphTy>::vertex_t *h_edges,
    typename gpu_graph<HIP, IMMGraphTy>::weight_t *h_weights,
    typename IMMGraphTy::edge_type *h_src_weighted_edges,
    typename IMMGraphTy::index_type *h_src_index, size_t num_nodes);
template gpu_graph<HIP, HCGraphTy> *make_gpu_graph<HIP, HCGraphTy>(
    const HCGraphTy &);
template void transform_to_gpu_graph<HIP, HCGraphTy>(
    typename gpu_graph<HIP, IMMGraphTy>::vertex_t *h_edges,
    typename gpu_graph<HIP, IMMGraphTy>::weight_t *h_weights,
    typename IMMGraphTy::edge_type *h_src_weighted_edges,
    typename IMMGraphTy::index_type *h_src_index, size_t num_nodes);
#endif
}  // namespace ripples
