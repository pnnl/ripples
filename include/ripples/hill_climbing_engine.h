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

#ifndef RIPPLES_HILL_CLIMBING_ENGINE_H
#define RIPPLES_HILL_CLIMBING_ENGINE_H

#include <atomic>
#include <cstdint>
#include <vector>

#include "omp.h"

#include "spdlog/spdlog.h"
#include "trng/uniform01_dist.hpp"

namespace ripples {

//! Engine scheduling dynamically sampling tasks for the Hill Climbing.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ItrTy The type of the workload iterator.
template <typename GraphTy, typename ItrTy>
class HCSamplingWorker {
 public:
  //! Construct the Sampling worker.
  //! \param G The input Graph.
  HCSamplingWorker(const GraphTy &G) : G_(G) {}
  //! Destructor.
  virtual ~HCSamplingWorker() = default;

  virtual void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E) = 0;

 protected:
  const GraphTy &G_;
};

template <typename GraphTy, typename ItrTy, typename PRNG,
          typename diff_model_tag>
class HCCPUSamplingWorker : public HCSamplingWorker<GraphTy, ItrTy> {
  using vertex_type = typename GraphTy::vertex_type;

  using HCSamplingWorker<GraphTy, ItrTy>::G_;

 public:
  HCCPUSamplingWorker(const GraphTy &G, const PRNG &rng)
      : HCSamplingWorker<GraphTy, ItrTy>(G), rng_(rng), UD_() {}

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) < std::distance(B, E)) {
      auto first = B;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);

      if (last > E) last = E;
      batch(first, last);
    }
  }

 private:
  void batch(ItrTy B, ItrTy E) {
    for (; B != E; ++B) {
      size_t edge_number = 0;
      if (std::is_same<diff_model_tag, independent_cascade_tag>::value) {
        for (vertex_type v = 0; v < G_.num_nodes(); ++v) {
          for (auto &e : G_.neighbors(v)) {
            if (UD_(rng_) >= e.weight) (*B)[edge_number] = true;

            ++edge_number;
          }
        }
      } else if (std::is_same<diff_model_tag, linear_threshold_tag>::value) {
        for (vertex_type v = 0; v < G_.num_nodes(); ++v) {
          double threshold = UD_(rng_);
          for (auto &e : G_.neighbors(v)) {
            threshold -= e.weight;
            if (threshold <= 0) {
              (*B)[edge_number] = true;
            }
            ++edge_number;
          }
        }
      }
    }
  }

  static constexpr size_t batch_size_ = 32;
  PRNG rng_;
  trng::uniform01_dist<float> UD_;
};

template <typename GraphTy, typename ItrTy, typename PRNGTy,
          typename diff_model_tag>
class SamplingEngine {
  using vertex_type = typename GraphTy::vertex_type;
  using cpu_worker_type =
      HCCPUSamplingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>;

 public:
  SamplingEngine(const GraphTy &G, PRNGTy &master_rng, size_t cpu_workers,
                 size_t gpu_workers)
      : G_(G), cpu_workers_(cpu_workers), gpu_workers_(gpu_workers) {
    size_t num_threads = cpu_workers_ + gpu_workers_;
    // Construct workers.
    workers_.reserve(num_threads);

    for (size_t i = 0; i < cpu_workers_; ++i) {
      auto rng = master_rng;
      rng.split(i, num_threads);
      workers_.push_back(new cpu_worker_type(G_, rng));
    }
  }

  ~SamplingEngine() {
    // Free workers.
    for (auto &v : workers_) delete v;
  }

  void exec(ItrTy B, ItrTy E) {
    mpmc_head_.store(0);

#pragma omp parallel
    {
      size_t rank = omp_get_thread_num();
      workers_[rank]->svc_loop(mpmc_head_, B, E);
    }
  }

 private:
  const GraphTy &G_;
  size_t gpu_workers_;
  size_t cpu_workers_;

  std::vector<HCSamplingWorker<GraphTy, ItrTy> *> workers_;
  std::atomic<size_t> mpmc_head_{0};
};

}  // namespace ripples
#endif
