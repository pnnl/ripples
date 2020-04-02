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

#include <unordered_map>

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_utils.h"

namespace ripples {

__global__ void kernel_lt_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                     cuda_PRNGeneratorTy r, size_t num_seqs,
                                     size_t first_seq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  d_trng_states[tid] = r;
  d_trng_states[tid].split(num_seqs, first_seq + tid);
}

__global__ void kernel_ic_trng_setup(cuda_PRNGeneratorTy *d_trng_states,
                                     cuda_PRNGeneratorTy r, size_t num_seqs,
                                     size_t first_seq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  d_trng_states[tid] = r;
  d_trng_states[tid].split(num_seqs, first_seq + tid);
}

void cuda_lt_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size) {
  kernel_lt_trng_setup<<<n_blocks, block_size>>>(d_trng_state, r, num_seqs,
                                                 first_seq);
  cuda_check(__FILE__, __LINE__);
}

void cuda_ic_rng_setup(cuda_PRNGeneratorTy *d_trng_state,
                       const cuda_PRNGeneratorTy &r, size_t num_seqs,
                       size_t first_seq, size_t n_blocks, size_t block_size) {
  kernel_ic_trng_setup<<<n_blocks, block_size>>>(d_trng_state, r, num_seqs,
                                                 first_seq);
  cuda_check(__FILE__, __LINE__);
}

}  // namespace ripples
