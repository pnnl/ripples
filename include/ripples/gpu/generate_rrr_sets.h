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

#ifndef RIPPLES_GPU_GENERATE_RRR_SETS_H
#define RIPPLES_GPU_GENERATE_RRR_SETS_H

#include <cstddef>
#include <utility>
#include <vector>

#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/diffusion_simulation.h"
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"
#include "ripples/graph.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace ripples {

//
// host-host API
//

// forward declarations to enable separate compilation
template <typename GraphTy>
using gpu_res_t = std::vector<std::vector<typename GraphTy::vertex_type>>;
using gpu_PRNGeneratorTy = trng::lcg64;
using gpu_PRNGeneratorsTy = std::vector<gpu_PRNGeneratorTy>;

constexpr size_t GPU_WALK_SIZE = 8;

//
// host-device API
//
using mask_word_t = int;  // TODO: vertex type hard-coded in nvgraph

void gpu_lt_rng_setup(gpu_PRNGeneratorTy *d_trng_state,
                      const gpu_PRNGeneratorTy &r, size_t num_seqs,
                      size_t first_seq, size_t n_blocks, size_t block_size);

void gpu_ic_rng_setup(gpu_PRNGeneratorTy *d_trng_state,
                      const gpu_PRNGeneratorTy &r, size_t num_seqs,
                      size_t first_seq, size_t n_blocks, size_t block_size);

template <GPURuntime R, typename GraphTy, typename gpu_PRNGeneratorTy>
extern void gpu_lt_kernel(size_t n_blocks, size_t block_size, size_t batch_size,
                          size_t num_nodes, gpu_PRNGeneratorTy *d_trng_states,
                          mask_word_t *d_res_masks, size_t num_mask_words,
                          gpu_ctx<R, GraphTy> *ctx,
                          typename GPU<R>::stream_type stream);

#if GPU_PROFILE
template <typename logst_t, typename sample_t>
void print_profile_counter(logst_t &logst, sample_t &sample,
                           const std::string &label) {
  if (!sample.empty()) {
    auto n = sample.size();
    std::sort(sample.begin(), sample.end());
    auto tot = std::accumulate(sample.begin(), sample.end(), size_t{0});
    logst->info("cnt={}\tmin={}\tmed={}\tmax={}\tavg={}", sample.size(),
                sample[0], sample[sample.size() / 2], sample.back(),
                (float)tot / sample.size());
    auto max_qi = 100, qi_step = 1;
    for (size_t qi = qi_step; qi < max_qi; qi += qi_step) {
      auto qp = (float)qi / max_qi * 100;
      auto si = qi * sample.size() / max_qi;
      logst->info("size {}%-percentile:\t{}", qp, sample[si]);
    }
  } else
    logst->info("*** tag: {} N/A", label);
}
#endif
}  // namespace ripples

#endif
