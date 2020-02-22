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
#include <memory>
#include <vector>

#include "omp.h"

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/uniform01_dist.hpp"

#ifdef RIPPLES_ENABLE_CUDA
#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_hc_engine.h"
#include "ripples/cuda/cuda_utils.h"
#endif

namespace ripples {

//! Engine scheduling dynamically sampling tasks for the Hill Climbing.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ItrTy The type of the workload iterator.
template <typename GraphTy, typename ItrTy>
class HCWorker {
 public:
  //! Construct the Sampling worker.
  //! \param G The input Graph.
  HCWorker(const GraphTy &G) : G_(G) {}
  //! Destructor.
  virtual ~HCWorker() = default;

  virtual void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E) = 0;

 protected:
  const GraphTy &G_;
};

template <typename GraphTy, typename ItrTy, typename PRNG,
          typename diff_model_tag>
class HCCPUSamplingWorker : public HCWorker<GraphTy, ItrTy> {
  using vertex_type = typename GraphTy::vertex_type;

  using HCWorker<GraphTy, ItrTy>::G_;

 public:
  HCCPUSamplingWorker(const GraphTy &G, const PRNG &rng)
      : HCWorker<GraphTy, ItrTy>(G), rng_(rng), UD_() {}

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
class HCGPUSamplingWorker : public HCWorker<GraphTy, ItrTy> {
#ifdef RIPPLES_ENABLE_CUDA
  using HCWorker<GraphTy, ItrTy>::G_;

 public:
  struct config_t {
    static constexpr size_t block_size_ = 256;
    static constexpr size_t num_threads_ = 1 << 15;

    size_t max_blocks_{0};

    config_t() : max_blocks_(num_threads_ / block_size_) {}

    size_t num_gpu_threads() const { return num_threads_; }
  };

  HCGPUSamplingWorker(const GraphTy &G, PRNGTy &rng, cuda_ctx<GraphTy> *ctx)
      : HCWorker<GraphTy, ItrTy>(G), ctx_(ctx), conf_(), master_rng_(rng) {
    cuda_malloc((void **)&d_trng_state_,
                conf_.num_gpu_threads() * sizeof(PRNGTy));
    cuda_malloc((void **)&d_flags_, G_.num_edges() * batch_size_);
  }

  ~HCGPUSamplingWorker() {
    cuda_free(d_trng_state_);
    cuda_free(d_flags_);
  }

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

  void rng_setup() {
    cuda_set_device(ctx_->gpu_id);
    cuda_lt_rng_setup(d_trng_state_, master_rng_, conf_.num_gpu_threads(), 0,
                      conf_.max_blocks_, conf_.block_size_);
  }

 private:
  void batch(ItrTy B, ItrTy E) {
    std::vector<char> flags(G_.num_edges() * batch_size_);
    cuda_set_device(ctx_->gpu_id);
    if (std::is_same<diff_model_tag, independent_cascade_tag>::value) {
      cuda_generate_samples_ic(conf_.max_blocks_, conf_.block_size_,
                               batch_size_, G_.num_edges(), d_trng_state_, ctx_,
                               d_flags_, cuda_stream_);
    } else if (std::is_same<diff_model_tag, linear_threshold_tag>::value) {
      assert(false && "Not Yet Implemented");
    }

    cuda_d2h(flags.data(), d_flags_, flags.size(), cuda_stream_);
    cuda_sync(cuda_stream_);

    auto Bf = flags.begin();
    auto Ef = Bf;
    std::advance(Ef, G_.num_nodes());
    for (; Ef < flags.end() && B < E; ++B) {
      std::transform(Bf, Ef, B->begin(),
                     [](char v) -> bool { v == 0 ? false : true; });
      std::advance(Bf, G_.num_nodes());
      std::advance(Ef, G_.num_nodes());
    }
  }

  static constexpr size_t batch_size_ = 32;
  cuda_ctx<GraphTy> *ctx_;
  config_t conf_;
  PRNGTy master_rng_;
  cudaStream_t cuda_stream_;
  trng::uniform01_dist<float> UD_;
  PRNGTy *d_trng_state_;
  char *d_flags_;
#endif
};

template <typename GraphTy, typename ItrTy, typename PRNGTy,
          typename diff_model_tag, typename CpuWorkerTy, typename GpuWorkerTy>
class PhaseEngine {
  using vertex_type = typename GraphTy::vertex_type;
  using worker_type = HCWorker<GraphTy, ItrTy>;
  using cpu_worker_type = CpuWorkerTy;
  using gpu_worker_type = GpuWorkerTy;

 public:
  PhaseEngine(const GraphTy &G, PRNGTy &master_rng, size_t cpu_workers,
              size_t gpu_workers, std::string loggerName)
      : G_(G), logger_(spdlog::stdout_color_st(loggerName)) {
    size_t num_threads = cpu_workers + gpu_workers;
    // Construct workers.
    logger_->debug("Number of Threads = {}", num_threads);
    workers_.reserve(num_threads);

    for (size_t i = 0; i < cpu_workers; ++i) {
      logger_->debug("> mapping: omp {}\t->CPU", i);
      auto rng = master_rng;
      rng.split(num_threads, i);
      auto w = new cpu_worker_type(G_, rng);
      workers_.push_back(w);
      cpu_workers_.push_back(w);
    }
#if RIPPLES_ENABLE_CUDA
    size_t num_devices = cuda_num_devices();
    for (size_t i = 0; i < gpu_workers; ++i) {
      size_t device_id = i % num_devices;
      logger_->debug("> mapping: omp {}\t->GPU {}/{}", i + cpu_workers,
                     device_id, num_devices);
      logger_->trace("Building Cuda Context");
      cuda_contexts_.push_back(cuda_make_ctx(G, device_id));
      logger_->trace("Cuda Context Built!");
      auto rng = master_rng;
      rng.split(num_threads, cpu_workers + i);
      auto w = new gpu_worker_type(G_, rng, cuda_contexts_.back());
      w->rng_setup();
      workers_.push_back(w);
      gpu_workers_.push_back(w);
    }
#endif
  }

  ~PhaseEngine() {
    // Free workers.
    for (auto &v : workers_) delete v;
#if RIPPLES_ENABLE_CUDA
    for (auto ctx : cuda_contexts_) {
      cuda_destroy_ctx(ctx);
      delete ctx;
    }
#endif
  }

 protected:
  const GraphTy &G_;

  std::shared_ptr<spdlog::logger> logger_;

  std::vector<cpu_worker_type *> cpu_workers_;
#if RIPPLES_ENABLE_CUDA
  std::vector<gpu_worker_type *> gpu_workers_;
  std::vector<cuda_ctx<GraphTy> *> cuda_contexts_;
#endif

  std::vector<worker_type *> workers_;
  std::atomic<size_t> mpmc_head_{0};
};

template <typename GraphTy, typename ItrTy, typename PRNGTy,
          typename diff_model_tag>
class SamplingEngine
    : public PhaseEngine<
          GraphTy, ItrTy, PRNGTy, diff_model_tag,
          HCCPUSamplingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>,
          HCGPUSamplingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>> {
  using phase_engine =
      PhaseEngine<GraphTy, ItrTy, PRNGTy, diff_model_tag,
                  HCCPUSamplingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>,
                  HCGPUSamplingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>>;

 public:
  SamplingEngine(const GraphTy &G, PRNGTy &master_rng, size_t cpu_workers,
                 size_t gpu_workers)
      : phase_engine(G, master_rng, cpu_workers, gpu_workers,
                     "SamplingEngine") {}

  void exec(ItrTy B, ItrTy E) {
    mpmc_head_.store(0);

    logger_->trace("Start Sampling");
#pragma omp parallel
    {
      assert(workers_.size() == omp_get_num_threads());
      size_t rank = omp_get_thread_num();
      workers_[rank]->svc_loop(mpmc_head_, B, E);
    }
    logger_->trace("End Sampling");
  }

 private:
  using phase_engine::logger_;
  using phase_engine::mpmc_head_;
  using phase_engine::workers_;
};
#if 0
template <typename GraphTy, typename ItrTy, typename PRNGTy,
          typename diff_model_tag>
class HCCPUCountingWorker {};

template <typename GraphTy, typename ItrTy, typename PRNGTy,
          typename diff_model_tag>
class HCGPUCountingWorker {};

template <typename GraphTy, typename ItrTy, typename PRNGTy,
          typename diff_model_tag>
class CountingEngine {
  using vertex_type = typename GraphTy::vertex_type;
  using worker_type = HCWorker<GraphTy, ItrTy>;
  using cpu_worker_type =
      HCCPUCountingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>;
  // using gpu_worker_type =
  //     HCGPUSamplingWorker<GraphTy, ItrTy, PRNGTy, diff_model_tag>;

 public:
  CountingEngine(const GraphTy &G, PRNGTy &master_rng, size_t cpu_workers,
                 size_t gpu_workers)
      : G_(G), logger_(spdlog::stdout_color_st("Counting Engine")) {
    size_t num_threads = cpu_workers + gpu_workers;
    // Construct workers.
    logger_->debug("Number of Threads = {}", num_threads);
    workers_.reserve(num_threads);

    for (size_t i = 0; i < cpu_workers; ++i) {
      logger_->debug("> mapping: omp {}\t->CPU", i);
      auto rng = master_rng;
      rng.split(num_threads, i);
      auto w = new cpu_worker_type(G_, rng);
      workers_.push_back(w);
      cpu_workers_.push_back(w);
    }
#if RIPPLES_ENABLE_CUDA
    size_t num_devices = cuda_num_devices();
    for (size_t i = 0; i < gpu_workers; ++i) {
      size_t device_id = i % num_devices;
      logger_->debug("> mapping: omp {}\t->GPU {}/{}", i + cpu_workers,
                     device_id, num_devices);
      logger_->trace("Building Cuda Context");
      cuda_contexts_.push_back(cuda_make_ctx(G, device_id));
      logger_->trace("Cuda Context Built!");
      auto rng = master_rng;
      rng.split(num_threads, cpu_workers + i);
      auto w = new gpu_worker_type(G_, rng, cuda_contexts_.back());
      w->rng_setup();
      workers_.push_back(w);
      gpu_workers_.push_back(w);
    }
#endif
  }

  ~CountingEngine() {
    // Free workers.
    for (auto &v : workers_) delete v;
#if RIPPLES_ENABLE_CUDA
    for (auto ctx : cuda_contexts_) {
      cuda_destroy_ctx(ctx);
      delete ctx;
    }
#endif
  }

  void exec(ItrTy B, ItrTy E) {
    mpmc_head_.store(0);

    logger_->trace("Start Sampling");
#pragma omp parallel
    {
      assert(workers_.size() == omp_get_num_threads());
      size_t rank = omp_get_thread_num();
      workers_[rank]->svc_loop(mpmc_head_, B, E);
    }
    logger_->trace("End Sampling");
  }

 private:
  const GraphTy &G_;
  // size_t gpu_workers_;
  // size_t cpu_workers_;

  std::shared_ptr<spdlog::logger> logger_;

  std::vector<cpu_worker_type *> cpu_workers_;
#if RIPPLES_ENABLE_CUDA
  std::vector<gpu_worker_type *> gpu_workers_;
  std::vector<cuda_ctx<GraphTy> *> cuda_contexts_;
#endif

  std::vector<worker_type *> workers_;
  std::atomic<size_t> mpmc_head_{0};
};
#endif
}  // namespace ripples
#endif
