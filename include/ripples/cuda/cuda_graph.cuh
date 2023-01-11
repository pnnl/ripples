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

#ifndef RIPPLES_CUDA_CUDA_GRAPH_CUH
#define RIPPLES_CUDA_CUDA_GRAPH_CUH

#include "ripples/cuda/cuda_utils.h"

namespace ripples {

template<typename GraphTy>
struct cuda_device_graph {
  using vertex_t = int; // TODO vertex type hard-coded in nvgraph
  using edge_t = typename GraphTy::edge_type;
  using weight_t = typename edge_t::edge_weight;
  vertex_t *d_index_ = nullptr, *d_edges_ = nullptr;
  weight_t *d_weights_ = nullptr;
};


template <typename GraphTy>
extern cuda_device_graph<GraphTy> *make_cuda_graph(const GraphTy &);


//! \brief Destroy a device-side CUDA Graph.
//!
//! \param hg The device-side CUDA Graph to be destroyed.
template <typename GraphTy>
void destroy_cuda_graph(cuda_device_graph<GraphTy> *g) {
  assert(g);
  assert(g->d_index_);
  assert(g->d_edges_);
  cudaFree(g->d_edges_);
  cudaFree(g->d_index_);
  delete g;
}

template <typename GraphTy>
struct cuda_ctx {
  size_t gpu_id;
  cuda_device_graph<GraphTy> * d_graph;

  ~cuda_ctx() {
    cuda_set_device(gpu_id);
    destroy_cuda_graph(d_graph);
  }
};

template<typename GraphTy>
cuda_ctx<GraphTy> *cuda_make_ctx(const GraphTy &G, size_t gpu_id) {
  auto res = new cuda_ctx<GraphTy>();
  res->gpu_id = gpu_id;
  cuda_set_device(gpu_id);
  res->d_graph = make_cuda_graph(G);
  return res;
}

template<typename GraphTy>
void cuda_destroy_ctx(cuda_ctx<GraphTy> *ctx) {
  cuda_set_device(ctx->gpu_id);
  //destroy_cuda_graph(ctx->d_graph); //Bug Fix: calling this will lead to double-free or corruption since device_graph will get deletd twice
}


template<typename GraphTy>
typename cuda_device_graph<GraphTy>::vertex_t *cuda_graph_index(cuda_ctx<GraphTy> *ctx) {
  return ctx->d_graph->d_index_;
}
template <typename GraphTy>
typename cuda_device_graph<GraphTy>::vertex_t *cuda_graph_edges(cuda_ctx<GraphTy> *ctx) {
  return ctx->d_graph->d_edges_;
}
template <typename GraphTy>
typename cuda_device_graph<GraphTy>::weight_t *cuda_graph_weights(cuda_ctx<GraphTy> *ctx) {
  return ctx->d_graph->d_weights_;
}

}  // namespace ripples

#endif  // RIPPLES_CUDA_CUDA_GRAPH_CUH
