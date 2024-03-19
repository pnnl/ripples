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

  GPU<R>::h2d(res->d_edges_, hg.csr_edges(),
              hg.num_edges() * sizeof(typename gpu_graph<R, GraphTy>::vertex_t));
  GPU<R>::h2d(res->d_weights_, hg.csr_weights(),
              hg.num_edges() * sizeof(typename gpu_graph<R, GraphTy>::weight_t));
  GPU<R>::h2d(res->d_index_, hg.csr_index(),
              (hg.num_nodes() + 1) * sizeof(typename gpu_graph<R, GraphTy>::index_t));

  return res;
}
}  // namespace ripples
