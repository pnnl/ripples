//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_GRAPH_CUH
#define IM_CUDA_CUDA_GRAPH_CUH

#include <cassert>

#include "ripples/graph.h"

template <typename host_GraphTy>
struct cuda_graph {
  typename host_GraphTy::DestinationTy **d_index_ = nullptr,
                                       *d_edges_ = nullptr;
};

//! \brief Construct a CUDA Graph from a host-side Graph.
//!
//! \param hg The host-side Graph to be mirrored.
template <typename host_GraphTy>
cuda_graph<host_GraphTy> *make_cuda_graph(const host_GraphTy &hg) {
  using destination_type = typename host_GraphTy::DestinationTy;

  auto res = new cuda_graph<host_GraphTy>();

  // prepare edges
  cudaMalloc(&res->d_edges_, hg.num_edges() * sizeof(destination_type));

  // prepare index
  cudaMalloc(&res->d_index_, (hg.num_nodes() + 1) * sizeof(destination_type *));

  // copy index
  auto ds = res->d_edges_;
  cudaMemcpy(res->d_index_, &res->d_edges_, sizeof(destination_type *),
             cudaMemcpyHostToDevice);
  for (size_t i = 1; i <= hg.num_nodes(); ++i) {
    ds += hg.degree(i - 1);
    cudaMemcpy(res->d_index_ + i, &ds, sizeof(destination_type *),
               cudaMemcpyHostToDevice);
  }

  // copy edges
  cudaMemcpy(res->d_edges_, hg.neighbors(0).begin(),
             hg.num_edges() * sizeof(destination_type), cudaMemcpyHostToDevice);

  return res;
}

template <typename cuda_graph>
void destroy_cuda_graph(cuda_graph *g) {
  assert(g);
  assert(g->d_index_);
  assert(g->d_edges_);
  cudaFree(g->d_edges_);
  cudaFree(g->d_index_);
  delete g;
}

#endif  // IM_CUDA_CUDA_GRAPH_CUH
