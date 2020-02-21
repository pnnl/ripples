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

#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_hc_engine.h"
#include "ripples/cuda/cuda_supported_graphs.h"

#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"

namespace ripples {
template <typename GraphTy, typename PRNGTy>
__global__ void generate_sample_ic_kernel(
    size_t batch_size, size_t num_edges,
    typename cuda_device_graph<GraphTy>::weight_t *weights,
    PRNGTy *d_trng_states, char *d_flag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int pos = tid;

  trng::uniform01_dist<float> u;
  auto &r(d_trng_states[tid]);

  while (pos < num_edges * batch_size) {
    int idx = pos % num_edges;
    typename cuda_device_graph<GraphTy>::weight_t w = weights[idx];
    d_flag[pos] = u(r) > w ? 1 : 0;

    pos += blockDim.x * gridDim.x;
  }
}

template <typename GraphTy, typename PRNGTy>
void cuda_generate_samples_ic(size_t n_blocks, size_t block_size,
                              size_t batch_size, size_t num_edges,
                              PRNGTy *d_trng_states, cuda_ctx<GraphTy> *ctx,
                              char *d_flags, cudaStream_t stream) {
  generate_sample_ic_kernel<GraphTy, PRNGTy>
      <<<n_blocks, block_size, 0, stream>>>(batch_size, num_edges,
                                            ctx->d_graph->d_weights_,
                                            d_trng_states, d_flags);
  cuda_check(__FILE__, __LINE__);
}

template <typename GraphTy, typename PRNGTy>
void cuda_generate_samples_lt(size_t n_blocks, size_t block_size,
                              size_t batch_size, size_t num_edges,
                              PRNGTy *d_trng_states, cuda_ctx<GraphTy> *ctx,
                              char *d_flags, cudaStream_t stream) {}

template void cuda_generate_samples_lt<HCGraphTy, trng::lcg64>(
    size_t n_blocks, size_t block_size, size_t batch_size, size_t num_edges,
    trng::lcg64 *d_trng_states, cuda_ctx<HCGraphTy> *ctx, char *d_flags,
    cudaStream_t stream);
template void cuda_generate_samples_ic<HCGraphTy, trng::lcg64>(
    size_t n_blocks, size_t block_size, size_t batch_size, size_t num_edges,
    trng::lcg64 *d_trng_states, cuda_ctx<HCGraphTy> *ctx, char *d_flags,
    cudaStream_t stream);
}  // namespace ripples
