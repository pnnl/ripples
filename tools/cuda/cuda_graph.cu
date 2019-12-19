//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iterator>

#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/cuda_graph.cuh"

namespace ripples {

__global__ void build_graph_kernel(
    typename cuda_device_graph::vertex_t *d_edges,
    typename cuda_device_graph::weight_t *d_weights,
    typename cuda_device_graph::vertex_t *d_index,
    typename cuda_GraphTy::edge_type *d_src_weighted_edges,
    typename cuda_GraphTy::edge_type **d_src_index, size_t num_nodes) {
  using vertex_t = typename cuda_device_graph::vertex_t;

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

//! \brief Construct a CUDA Graph from a host-side Graph.
//!
//! \param hg The host-side Graph to be mirrored.
cuda_device_graph *make_cuda_graph(const cuda_GraphTy &hg) {

  // allocate
  auto res = new cuda_device_graph();
  cudaMalloc(&res->d_edges_,
             hg.num_edges() * sizeof(typename cuda_device_graph::vertex_t));
  cudaMalloc(&res->d_weights_,
             hg.num_edges() * sizeof(typename cuda_device_graph::weight_t));
  cudaMalloc(
      &res->d_index_,
      (hg.num_nodes() + 1) * sizeof(typename cuda_device_graph::vertex_t));
  cuda_check(__FILE__, __LINE__);

  // copy graph to device
  using destination_type = typename cuda_GraphTy::edge_type;
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
  build_graph_kernel<<<n_blocks, block_size>>>(res->d_edges_, res->d_weights_,
                                               res->d_index_, d_weighted_edges,
                                               d_index, hg.num_nodes());
  cuda_check(__FILE__, __LINE__);

  cudaFree(d_weighted_edges);
  cudaFree(d_index);
  cuda_check(__FILE__, __LINE__);

  return res;
}

void destroy_cuda_graph(cuda_device_graph *g) {
  assert(g);
  assert(g->d_index_);
  assert(g->d_edges_);
  cudaFree(g->d_edges_);
  cudaFree(g->d_index_);
  delete g;
}

}  // namespace ripples
