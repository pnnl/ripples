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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "omp.h"

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "trng/uniform01_dist.hpp"

#include "ripples/bitmask.h"
#ifdef RIPPLES_ENABLE_CUDA
#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/cuda_graph.cuh"
#include "ripples/cuda/cuda_hc_engine.h"
#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/from_nvgraph/hc/bfs.hxx"
#endif

namespace ripples {

//! Engine scheduling dynamically sampling tasks for the Hill Climbing.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ItrTy The type of the workload iterator.
template <typename GraphTy, typename ItrTy>
class HCWorker {
 public:
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  //! Construct the Sampling worker.
  //! \param G The input Graph.
  HCWorker(const GraphTy &G) : G_(G) {}
  //! Destructor.
  virtual ~HCWorker() = default;

  virtual void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E,
                        std::vector<ex_time_ms> &record) = 0;

 protected:
  const GraphTy &G_;
};

template <typename GraphTy, typename ItrTy, typename PRNG,
          typename diff_model_tag>
class HCCPUSamplingWorker : public HCWorker<GraphTy, ItrTy> {
  using vertex_type = typename GraphTy::vertex_type;

  using HCWorker<GraphTy, ItrTy>::G_;

 public:
  using ex_time_ms = std::chrono::duration<double, std::milli>;

  HCCPUSamplingWorker(const GraphTy &G, const PRNG &rng)
      : HCWorker<GraphTy, ItrTy>(G), rng_(rng), UD_() {}

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E,
                std::vector<ex_time_ms> &record) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) < std::distance(B, E)) {
      auto first = B;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);

      if (last > E) last = E;
      auto start = std::chrono::high_resolution_clock::now();
      batch(first, last);
      auto end = std::chrono::high_resolution_clock::now();
      record.push_back(end - start);
    }
  }

 private:
  void batch(ItrTy B, ItrTy E) {
    for (; B != E; ++B) {
      size_t edge_number = 0;
      if (std::is_same<diff_model_tag, independent_cascade_tag>::value) {
        for (vertex_type v = 0; v < G_.num_nodes(); ++v) {
          for (auto &e : G_.neighbors(v)) {
            // (*B)[edge_number] = UD_(rng_) <= e.weight ? 1 : 0;
            if (UD_(rng_) <= e.weight)
              B->set(edge_number);
            ++edge_number;
          }
        }
      } else if (std::is_same<diff_model_tag, linear_threshold_tag>::value) {
        for (vertex_type v = 0; v < G_.num_nodes(); ++v) {
          double threshold = UD_(rng_);
          for (auto &e : G_.neighbors(v)) {
            threshold -= e.weight;
            if (threshold <= 0)
              B->set(edge_number);
            // (*B)[edge_number] = threshold <= 0 ? 1 : 0;
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
  using ex_time_ms = std::chrono::duration<double, std::milli>;

  struct config_t {
    static constexpr size_t block_size_ = 256;
    static constexpr size_t num_threads_ = 1 << 15;

    size_t max_blocks_{0};

    config_t() : max_blocks_(num_threads_ / block_size_) {}

    size_t num_gpu_threads() const { return num_threads_; }
  };

  HCGPUSamplingWorker(const GraphTy &G, PRNGTy &rng, cuda_ctx<GraphTy> *ctx)
      : HCWorker<GraphTy, ItrTy>(G), ctx_(ctx), conf_(), master_rng_(rng) {
    cuda_set_device(ctx_->gpu_id);
    cuda_stream_create(&cuda_stream_);

    cuda_malloc((void **)&d_trng_state_,
                conf_.num_gpu_threads() * sizeof(PRNGTy));
    cuda_malloc((void **)&d_flags_,
                ((G.num_edges() / (8 * sizeof(int)) + 1) * sizeof(int) * batch_size_));
  }

  ~HCGPUSamplingWorker() {
    cuda_set_device(ctx_->gpu_id);
    cuda_stream_destroy(cuda_stream_);
    cuda_free(d_trng_state_);
    cuda_free(d_flags_);
  }

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E,
                std::vector<ex_time_ms> &record) {
    size_t offset = 0;
    cuda_set_device(ctx_->gpu_id);   //Bug fix to prevent cuda device not ready
    while ((offset = mpmc_head.fetch_add(batch_size_)) < std::distance(B, E)) {
      auto first = B;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);

      if (last > E) last = E;
      auto start = std::chrono::high_resolution_clock::now();
      batch(first, last);
      auto end = std::chrono::high_resolution_clock::now();
      record.push_back(end - start);
    }
  }

  void rng_setup() {
    cuda_set_device(ctx_->gpu_id);
    cuda_lt_rng_setup(d_trng_state_, master_rng_, conf_.num_gpu_threads(), 0,
                      conf_.max_blocks_, conf_.block_size_);
  }

 private:
  void batch(ItrTy B, ItrTy E) {
    cuda_set_device(ctx_->gpu_id);
    if (std::is_same<diff_model_tag, independent_cascade_tag>::value) {
      cuda_generate_samples_ic(conf_.max_blocks_, conf_.block_size_,
                               batch_size_, G_.num_edges(), d_trng_state_, ctx_,
                               d_flags_, cuda_stream_);
    } else if (std::is_same<diff_model_tag, linear_threshold_tag>::value) {
      assert(false && "Not Yet Implemented");
    }

    for (size_t i = 0; B < E; ++B, ++i) {
      cuda_d2h(B->data(), d_flags_ + i * (B->bytes() / sizeof(int)),
               B->bytes(), cuda_stream_);
    }
    cuda_sync(cuda_stream_);
  }

  static constexpr size_t batch_size_ = 32;
  cuda_ctx<GraphTy> *ctx_;
  config_t conf_;
  PRNGTy master_rng_;
  cudaStream_t cuda_stream_;
  trng::uniform01_dist<float> UD_;
  PRNGTy *d_trng_state_;
  int *d_flags_;
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
  using ex_time_ms = std::chrono::duration<double, std::milli>;

  PhaseEngine(const GraphTy &G, PRNGTy &master_rng, size_t cpu_workers,
              size_t gpu_workers, std::string loggerName)
      : G_(G), logger_(spdlog::stdout_color_mt(loggerName)) {
    size_t num_threads = cpu_workers + gpu_workers;
    // Construct workers.
    logger_->debug("Number of Threads = {}", num_threads);
    workers_.resize(num_threads);
    cpu_workers_.resize(cpu_workers);
#if RIPPLES_ENABLE_CUDA
    gpu_workers_.resize(gpu_workers);
    cuda_contexts_.resize(gpu_workers);
#endif

#pragma omp parallel
    {
      int rank = omp_get_thread_num();
      if (rank < cpu_workers) {
        auto rng = master_rng;
        rng.split(num_threads, rank);
        auto w = new cpu_worker_type(G_, rng);
        workers_[rank] = w;
        cpu_workers_[rank] = w;
        logger_->debug("> mapping: omp {}\t->CPU", rank);
      } else {
#if RIPPLES_ENABLE_CUDA
        size_t num_devices = cuda_num_devices();
        size_t device_id = rank % num_devices;
        logger_->debug("> mapping: omp {}\t->GPU {}/{}", rank, device_id,
                       num_devices);
        logger_->trace("Building Cuda Context");
        cuda_contexts_[rank - cpu_workers] = cuda_make_ctx(G, device_id);
        auto rng = master_rng;
        rng.split(num_threads, rank);
        auto w =
            new gpu_worker_type(G_, rng, cuda_contexts_[rank - cpu_workers]);
        w->rng_setup();
        workers_[rank] = w;
        gpu_workers_[rank - cpu_workers] = w;
        logger_->trace("Cuda Context Built!");
#endif
      }
    }
  }

  ~PhaseEngine() {
    // Free workers.
    for (auto &v : workers_) delete v;
#if RIPPLES_ENABLE_CUDA
    for (auto ctx : cuda_contexts_) {
      cuda_set_device(ctx->gpu_id);
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

  using ex_time_ms = std::chrono::duration<double, std::milli>;

 public:
  SamplingEngine(const GraphTy &G, PRNGTy &master_rng, size_t cpu_workers,
                 size_t gpu_workers)
    : phase_engine(G, master_rng, cpu_workers, gpu_workers,
                     "SamplingEngine") {}

  void exec(ItrTy B, ItrTy E, std::vector<std::vector<ex_time_ms>> &record) {
    record.resize(workers_.size());
    mpmc_head_.store(0);

    logger_->trace("Start Sampling");
#pragma omp parallel
    {
      assert(workers_.size() == omp_get_num_threads());
      size_t rank = omp_get_thread_num();
      workers_[rank]->svc_loop(mpmc_head_, B, E, record[rank]);
    }
    logger_->trace("End Sampling");
  }

 private:
  using phase_engine::logger_;
  using phase_engine::mpmc_head_;
  using phase_engine::workers_;
};

namespace {
template <typename GraphTy, typename GraphMaskTy, typename Itr>
size_t BFS(GraphTy &G, GraphMaskTy &M, Itr b, Itr e, Bitmask<int> &visited) {
  using vertex_type = typename GraphTy::vertex_type;

  std::queue<vertex_type> queue;
  for (; b != e; ++b) {
    queue.push(*b);
  }

  while (!queue.empty()) {
    vertex_type u = queue.front();
    queue.pop();
    visited.set(u);

    size_t edge_number =
        std::distance(G.neighbors(0).begin(), G.neighbors(u).begin());

    for (auto v : G.neighbors(u)) {
      if (M.get(edge_number) && !visited.get(v.vertex)) {
        queue.push(v.vertex);
      }

      ++edge_number;
    }
  }

  return visited.popcount();
}

template <typename GraphTy, typename GraphMaskTy>
size_t BFS(GraphTy &G, GraphMaskTy &M, typename GraphTy::vertex_type v,
           Bitmask<int> visited) {
  using vertex_type = typename GraphTy::vertex_type;

  std::queue<vertex_type> queue;

  queue.push(v);
  visited.set(v);
  while (!queue.empty()) {
    vertex_type u = queue.front();
    queue.pop();

    size_t edge_number =
        std::distance(G.neighbors(0).begin(), G.neighbors(u).begin());
    for (auto v : G.neighbors(u)) {
      if (M.get(edge_number) && !visited.get(v.vertex)) {
        queue.push(v.vertex);
        visited.set(v.vertex);
      }
      ++edge_number;
    }
  }
  return visited.popcount();
}
}  // namespace

template <typename GraphTy, typename ItrTy>
class HCCPUCountingWorker : public HCWorker<GraphTy, ItrTy> {
  using vertex_type = typename GraphTy::vertex_type;
  using HCWorker<GraphTy, ItrTy>::G_;

 public:
  using ex_time_ms = std::chrono::duration<double, std::milli>;

  HCCPUCountingWorker(const GraphTy &G, std::vector<size_t> &count,
                      const std::set<vertex_type> &S)
      : HCWorker<GraphTy, ItrTy>(G), count_(count), S_(S) {}

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E,
                std::vector<ex_time_ms> &record) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) < std::distance(B, E)) {
      auto first = B;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);

      if (last > E) last = E;
      auto start = std::chrono::high_resolution_clock::now();
      batch(first, last);
      auto end = std::chrono::high_resolution_clock::now();
      record.push_back(end - start);
    }
  }

 private:
  void batch(ItrTy B, ItrTy E) {
    for (auto itr = B; itr < E; ++itr) {
      Bitmask<int> visited(G_.num_nodes());
      size_t base_count = BFS(G_, *itr, S_.begin(), S_.end(), visited);

      for (vertex_type v = 0; v < G_.num_nodes(); ++v) {
        if (S_.find(v) != S_.end()) continue;
        size_t update_count = base_count + 1;
        if (!visited.get(v)) {
          update_count = BFS(G_, *itr, v, visited);
        }
#pragma omp atomic
        count_[v] += update_count;
      }
    }
  }

  static constexpr size_t batch_size_ = 2;
  std::vector<size_t> &count_;
  const std::set<vertex_type> &S_;
};

template <typename GraphTy, typename ItrTy>
class HCGPUCountingWorker : public HCWorker<GraphTy, ItrTy> {
#ifdef RIPPLES_ENABLE_CUDA
  using vertex_type = typename GraphTy::vertex_type;
  using d_vertex_type = typename cuda_device_graph<GraphTy>::vertex_t;
  using bfs_solver_t = nvgraph::Bfs<int>;
  using HCWorker<GraphTy, ItrTy>::G_;

 public:
  using ex_time_ms = std::chrono::duration<double, std::milli>;

  struct config_t {
    config_t(size_t num_workers)
        : block_size_(bfs_solver_t::traverse_block_size()),
          max_blocks_(num_workers ? cuda_max_blocks() / num_workers : 0) {
      auto console = spdlog::get("console");
      console->trace(
          "> [GPUWalkWorkerIC::config_t] "
          "max_blocks_={}\tblock_size_={}",
          max_blocks_, block_size_);
    }

    size_t num_gpu_threads() const { return max_blocks_ * block_size_; }

    const size_t max_blocks_;
    const size_t block_size_;
  };

  HCGPUCountingWorker(const config_t &conf, const GraphTy &G,
                      cuda_ctx<GraphTy> *ctx, std::vector<size_t> &count,
                      const std::set<vertex_type> &S)
      : HCWorker<GraphTy, ItrTy>(G),
        conf_(conf),
        ctx_(ctx),
        count_(count),
        S_(S),
        edge_filter_(new d_vertex_type[G_.num_edges()]) {
    cuda_set_device(ctx_->gpu_id);
    cuda_stream_create(&cuda_stream_);

    // allocate host/device memory
    Bitmask<int> _(G_.num_edges());
    cuda_malloc((void **)&d_edge_filter_, _.bytes());

    // create the solver
    solver_ = new bfs_solver_t(this->G_.num_nodes(), this->G_.num_edges(),
                               cuda_graph_index(ctx_), cuda_graph_edges(ctx_),
                               cuda_graph_weights(ctx_), true,
                               TRAVERSAL_DEFAULT_ALPHA, TRAVERSAL_DEFAULT_BETA,
                               conf_.max_blocks_, cuda_stream_);
    solver_->configure(nullptr, nullptr, d_edge_filter_);
    visited_ = std::unique_ptr<int[]>(new int[solver_->bmap_size()]);
    cuda_sync(cuda_stream_);
  }

  ~HCGPUCountingWorker() {
    cuda_set_device(ctx_->gpu_id);

    delete solver_;
    cuda_stream_destroy(cuda_stream_);

    // free host/device memory
    cuda_free(d_edge_filter_);
  }

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy B, ItrTy E,
                std::vector<ex_time_ms> &record) {
    size_t offset = 0;
    cuda_set_device(ctx_->gpu_id);  //Bug fix to prevent cuda device not ready
    while ((offset = mpmc_head.fetch_add(batch_size_)) < std::distance(B, E)) {
      auto first = B;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);

      if (last > E) last = E;
      auto start = std::chrono::high_resolution_clock::now();
      batch(first, last);
      auto end = std::chrono::high_resolution_clock::now();
      record.push_back(end - start);
    }
  }

 private:
  void batch(ItrTy B, ItrTy E) {
    std::vector<d_vertex_type> seeds(S_.begin(), S_.end());
    for (auto itr = B; itr < E; ++itr) {
      cuda_h2d(d_edge_filter_, itr->data(), itr->bytes(), cuda_stream_);

      d_vertex_type base_count;
      solver_->traverse(seeds.data(), seeds.size(), visited_.get(),
                        &base_count);

      // cuda_d2h(predecessors_, d_predecessors_,
      // G_.num_nodes() * sizeof(d_vertex_type), cuda_stream_);
      cuda_sync(cuda_stream_);
      for (vertex_type v = 0; v < G_.num_nodes(); ++v) {
        if (S_.find(v) != S_.end()) continue;
        size_t update_count = base_count + 1;
        int m = 1 << (v % (8 * sizeof(int)));
        if ((visited_[v / (8 * sizeof(int))] && m) == 0) {
          d_vertex_type count;
          solver_->traverse(v, base_count, visited_.get(), &count);
          cuda_sync(cuda_stream_);

          update_count = count;
        }
#pragma omp atomic
        count_[v] += update_count;
      }
    }
  }

  static constexpr size_t batch_size_ = 2;
  config_t conf_;
  cuda_ctx<GraphTy> *ctx_;
  cudaStream_t cuda_stream_;
  bfs_solver_t *solver_;
  std::unique_ptr<d_vertex_type[]> edge_filter_;
  std::unique_ptr<int[]> visited_;
  d_vertex_type *d_edge_filter_;

  std::vector<size_t> &count_;
  const std::set<vertex_type> &S_;
#endif
};

template <typename GraphTy, typename ItrTy>
class SeedSelectionEngine {
  using vertex_type = typename GraphTy::vertex_type;
  using worker_type = HCWorker<GraphTy, ItrTy>;
  using cpu_worker_type = HCCPUCountingWorker<GraphTy, ItrTy>;
  using gpu_worker_type = HCGPUCountingWorker<GraphTy, ItrTy>;

 public:
  using ex_time_ms = std::chrono::duration<double, std::milli>;

  SeedSelectionEngine(const GraphTy &G, size_t cpu_workers, size_t gpu_workers)
      : G_(G),
        count_(G_.num_nodes()),
        S_(),
        logger_(spdlog::stdout_color_mt("SeedSelectionEngine")) {
    size_t num_threads = cpu_workers + gpu_workers;
    // Construct workers.
    logger_->debug("Number of Threads = {}", num_threads);
    workers_.resize(num_threads);
    cpu_workers_.resize(cpu_workers);
#if RIPPLES_ENABLE_CUDA
    gpu_workers_.resize(gpu_workers);
    cuda_contexts_.resize(gpu_workers);
#endif

#pragma omp parallel
    {
      int rank = omp_get_thread_num();
      if (rank < cpu_workers) {
        auto w = new cpu_worker_type(G_, count_, S_);
        workers_[rank] = w;
        cpu_workers_[rank] = w;
        logger_->debug("> mapping: omp {}\t->CPU", rank);
      } else {
#if RIPPLES_ENABLE_CUDA
        size_t num_devices = cuda_num_devices();
        size_t device_id = rank % num_devices;
        logger_->debug("> mapping: omp {}\t->GPU {}/{}", rank, device_id,
                       num_devices);
        logger_->trace("Building Cuda Context");
        cuda_contexts_[rank - cpu_workers] = cuda_make_ctx(G, device_id);
        typename gpu_worker_type::config_t gpu_conf(gpu_workers);
        auto w = new gpu_worker_type(gpu_conf, G_, cuda_contexts_[rank - cpu_workers],  //changed from .back()
                                     count_, S_);
        workers_[rank] = w;
        gpu_workers_[rank - cpu_workers] = w;
        logger_->trace("Cuda Context Built!");
#endif
      }
    }
  }

  ~SeedSelectionEngine() {
    // Free workers.
    for (auto &v : workers_) delete v;
#if RIPPLES_ENABLE_CUDA
    for (auto ctx : cuda_contexts_) {
      cuda_set_device(ctx->gpu_id);
      cuda_destroy_ctx(ctx);
      delete ctx;
    }
#endif
  }

  std::vector<vertex_type> exec(ItrTy B, ItrTy E, size_t k,
                                std::vector<std::vector<ex_time_ms>> &record) {
    logger_->trace("Start Seed Selection");

    record.resize(workers_.size());
    std::vector<vertex_type> result;
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
#pragma omp parallel for
      for (size_t j = 0; j < count_.size(); ++j) count_[j] = 0;

      mpmc_head_.store(0);
#pragma omp parallel
      {
        assert(workers_.size() == omp_get_num_threads());
        size_t rank = omp_get_thread_num();
        workers_[rank]->svc_loop(mpmc_head_, B, E, record[rank]);
      }

      auto itr = std::max_element(count_.begin(), count_.end());
      vertex_type v = std::distance(count_.begin(), itr);
      S_.insert(v);
      result.push_back(v);
      logger_->trace("Seed {} : {}[{}] = {}", i, v, G_.convertID(v), *itr);
    }

    logger_->trace("End Seed Selection");
    return result;
  }

 private:
  const GraphTy &G_;
  std::vector<size_t> count_;
  std::set<vertex_type> S_;
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
}  // namespace ripples
#endif
