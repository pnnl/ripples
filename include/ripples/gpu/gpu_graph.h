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

#ifndef RIPPLES_GPU_GPU_GRAPH_H
#define RIPPLES_GPU_GPU_GRAPH_H

#include "ripples/gpu/gpu_runtime_trait.h"

namespace ripples {

template <GPURuntime R, typename GraphTy>
struct gpu_graph {
  using vertex_t = typename GraphTy::vertex_type;  // TODO vertex type hard-coded in nvgraph
  using index_t = typename GraphTy::index_type;
  using edge_t = typename GraphTy::edge_type;
  using weight_t = typename edge_t::edge_weight;

  index_t *d_index_ = nullptr;
  vertex_t *d_edges_ = nullptr;
  weight_t *d_weights_ = nullptr;
};

template <GPURuntime R, typename GraphTy>
extern gpu_graph<R, GraphTy> * make_gpu_graph(const GraphTy &);

template <GPURuntime R, typename GraphTy>
void destroy_gpu_graph(gpu_graph<R, GraphTy> *g) {
  assert(g);
  assert(g->d_index_);
  assert(g->d_edges_);
  GPU<R>::device_free(g->d_index_);
  GPU<R>::device_free(g->d_edges_);
  if (g->d_weights_ != nullptr) GPU<R>::device_free(g->d_weights_);
  delete g;
}

template <GPURuntime R, typename GraphTy>
struct gpu_ctx {
  using device_id_type = typename GPU<R>::device_id_type;
  using device_graph_type = gpu_graph<R, GraphTy>;

  device_id_type gpu_id;
  device_graph_type *d_graph;
};

template <GPURuntime R, typename GraphTy>
gpu_ctx<R, GraphTy> *make_gpu_context(const GraphTy &G,
                                      typename GPU<R>::device_id_type gpu_id) {
  auto res = new gpu_ctx<R, GraphTy>();
  res->gpu_id = gpu_id;
  GPU<R>::set_device(gpu_id);
  res->d_graph = make_gpu_graph<R>(G);
  return res;
}

template <GPURuntime R, typename GraphTy>
void destroy_gpu_context(gpu_ctx<R, GraphTy> *ctx) {
  GPU<R>::set_device(ctx->gpu_id);
  destroy_gpu_graph(ctx->d_graph);
}

}  // namespace ripples

#endif
