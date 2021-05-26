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

#ifndef RIPPLES_STREAMING_RRR_GENERATOR_H
#define RIPPLES_STREAMING_RRR_GENERATOR_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "omp.h"

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "trng/uniform_int_dist.hpp"

#include "ripples/imm_execution_record.h"

#ifdef RIPPLES_ENABLE_CUDA
#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/from_nvgraph/imm/bfs.hxx"
#endif

#if CUDA_PROFILE
#include <chrono>
#endif

namespace ripples {

int streaming_command_line(std::unordered_map<size_t, size_t> &worker_to_gpu,
                           size_t streaming_workers,
                           size_t streaming_gpu_workers,
                           std::string gpu_mapping_string) {
  auto console = spdlog::get("console");
  if (!(streaming_workers > 0 && streaming_gpu_workers <= streaming_workers)) {
    console->error("invalid number of streaming workers");
    return -1;
  }

#ifdef RIPPLES_ENABLE_CUDA
  auto num_gpus = cuda_num_devices();
  if (!gpu_mapping_string.empty()) {
    size_t gpu_id = 0;
    std::istringstream iss(gpu_mapping_string);
    std::string token;
    while (worker_to_gpu.size() < streaming_gpu_workers &&
           std::getline(iss, token, ',')) {
      std::stringstream omp_num_ss(token);
      size_t omp_num;
      omp_num_ss >> omp_num;
      if (!(omp_num < streaming_workers)) {
        console->error("invalid worker in worker-to-GPU mapping: {}", omp_num);
        return -1;
      }
      if (worker_to_gpu.find(omp_num) != worker_to_gpu.end()) {
        console->error("duplicated worker-to-GPU mapping: {}", omp_num);
        return -1;
      }
      worker_to_gpu[omp_num] = gpu_id++;
      if (gpu_id == num_gpus) gpu_id = 0;
    }
    if (worker_to_gpu.size() < streaming_gpu_workers) {
      console->error("GPU mapping string is too short");
      return -1;
    }
  } else {
    // by default, map GPU workers after CPU workers
    size_t gpu_id = 0;
    size_t omp_num = streaming_workers - streaming_gpu_workers;
    for (; omp_num < streaming_workers; ++omp_num) {
      worker_to_gpu[omp_num] = gpu_id++;
      if (gpu_id == num_gpus) gpu_id = 0;
    }
  }
#else   // RIPPLES_ENABLE_CUDA

  assert(streaming_gpu_workers == 0);
#endif  // RIPPLES_ENABLE_CUDA
  return 0;
}

template <typename GraphTy, typename ItrTy>
class WalkWorker {
  using vertex_t = typename GraphTy::vertex_type;

 public:
  WalkWorker(const GraphTy &G) : G_(G) {}
  virtual ~WalkWorker() {}
  virtual void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin,
                        ItrTy end) = 0;

 protected:
  const GraphTy &G_;

#if CUDA_PROFILE
 public:
  virtual void begin_prof_iter() = 0;
  virtual void prof_record(typename IMMExecutionRecord::walk_iteration_prof &,
                           size_t) = 0;
#endif
};

template <typename GraphTy, typename PRNGeneratorTy, typename ItrTy,
          typename diff_model_tag>
class CPUWalkWorker : public WalkWorker<GraphTy, ItrTy> {
  using vertex_t = typename GraphTy::vertex_type;

 public:
  CPUWalkWorker(const GraphTy &G, const PRNGeneratorTy &rng)
      : WalkWorker<GraphTy, ItrTy>(G), rng_(rng), u_(0, G.num_nodes()) {}

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);
      if (last > end) last = end;
      batch(first, last);
    }
  }

 private:
  static constexpr size_t batch_size_ = 32;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;

  void batch(ItrTy first, ItrTy last) {
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto size = std::distance(first, last);
    auto local_rng = rng_;
    auto local_u = u_;
    for (;first != last; ++first) {
      vertex_t root = local_u(local_rng);

      AddRRRSet(this->G_, root, local_rng, *first, diff_model_tag{});
    }

    rng_ = local_rng;
    u_ = local_u;
#if CUDA_PROFILE
    auto &p(prof_bd.back());
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }

#if CUDA_PROFILE
 public:
  struct iter_profile_t {
    size_t n_{0};
    std::chrono::nanoseconds d_{0};
  };
  using profile_t = std::vector<iter_profile_t>;
  profile_t prof_bd;

  void begin_prof_iter() { prof_bd.emplace_back(); }
  void print_prof_iter(size_t i) {
    auto console = spdlog::get("console");
    assert(i < prof_bd.size());
    auto &p(prof_bd[i]);
    if (p.n_)
      console->info(
          "n-sets={}\tns={}\tb={}", p.n_, p.d_.count(),
          (float)p.n_ * 1e03 /
              std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                  .count());
    else
      console->info("> idle worker");
  }
  void prof_record(typename IMMExecutionRecord::walk_iteration_prof &r,
                   size_t i) {
    assert(i < prof_bd.size());
    typename IMMExecutionRecord::cpu_walk_prof res;
    auto &p(prof_bd[i]);
    res.NumSets = p.n_;
    res.Total = std::chrono::duration_cast<decltype(res.Total)>(p.d_);
    r.CPUWalks.push_back(res);
  }
#endif
};

template <typename GraphTy, typename PRNGeneratorTy, typename ItrTy,
          typename diff_model_tag>
class GPUWalkWorker;

#ifdef RIPPLES_ENABLE_CUDA
template <typename GraphTy, typename PRNGeneratorTy, typename ItrTy>
class GPUWalkWorker<GraphTy, PRNGeneratorTy, ItrTy, linear_threshold_tag>
    : public WalkWorker<GraphTy, ItrTy> {
  using vertex_t = typename GraphTy::vertex_type;

 public:
  struct config_t {
    config_t(size_t) {
      auto console = spdlog::get("console");
      assert(num_threads_ % block_size_ == 0);
      max_blocks_ = num_threads_ / block_size_;
#if CUDA_PROFILE
      console->info(
          "> [GPUWalkWorkerLT::config_t] "
          "block_size_={}\tnum_threads_={}\tmax_blocks_={}",
          block_size_, num_threads_, max_blocks_);
#endif
    }

    size_t num_gpu_threads() const { return num_threads_; }

    // configuration parameters
    static constexpr size_t block_size_ = 256;
    static constexpr size_t num_threads_ = 1 << 15;
    const size_t mask_words_ = 8;  // maximum walk size

    // inferred configuration
    size_t max_blocks_{0};
  };

  GPUWalkWorker(const config_t &conf, const GraphTy &G,
                const PRNGeneratorTy &rng,
                std::shared_ptr<cuda_ctx<GraphTy>> ctx)
      : WalkWorker<GraphTy, ItrTy>(G),
        conf_(conf),
        rng_(rng),
        u_(0, G.num_nodes()),
        cuda_ctx_(ctx) {
    cuda_set_device(ctx->gpu_id);
    cuda_stream_create(&cuda_stream_);

    // allocate host/device memory
    auto mask_size = conf.mask_words_ * sizeof(mask_word_t);
    lt_res_mask_ = (mask_word_t *)malloc(conf_.num_gpu_threads() * mask_size);
    cuda_malloc((void **)&d_lt_res_mask_, conf_.num_gpu_threads() * mask_size);

    // allocate device-size RNGs
    cuda_malloc((void **)&d_trng_state_,
                conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));
  }

  ~GPUWalkWorker() {
    cuda_set_device(cuda_ctx_->gpu_id);
    cuda_stream_destroy(cuda_stream_);
    // free host/device memory
    free(lt_res_mask_);
    cuda_free(d_lt_res_mask_);
    cuda_free(d_trng_state_);
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {
    cuda_set_device(cuda_ctx_->gpu_id);
    cuda_lt_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                      conf_.max_blocks_, conf_.block_size_);
  }

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
    cuda_set_device(cuda_ctx_->gpu_id);
    size_t offset = 0;
    auto batch_size = conf_.num_gpu_threads();
    while ((offset = mpmc_head.fetch_add(batch_size)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size);
      if (last > end) last = end;
      batch(first, last);
    }
  }

 private:
  config_t conf_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;
  cudaStream_t cuda_stream_;
  std::shared_ptr<cuda_ctx<GraphTy>> cuda_ctx_;

  // memory buffers
  mask_word_t *lt_res_mask_, *d_lt_res_mask_;
  PRNGeneratorTy *d_trng_state_;

  void batch(ItrTy first, ItrTy last) {
#if CUDA_PROFILE
    auto &p(prof_bd.back());
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto size = std::distance(first, last);

    cuda_lt_kernel(conf_.max_blocks_, conf_.block_size_, size,
                   this->G_.num_nodes(), d_trng_state_, d_lt_res_mask_,
                   conf_.mask_words_, cuda_ctx_.get(), cuda_stream_);
#if CUDA_PROFILE
    cuda_sync(cuda_stream_);
    auto t1 = std::chrono::high_resolution_clock::now();
    p.dwalk_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - start);
    auto t0 = t1;
#endif

    cuda_d2h(lt_res_mask_, d_lt_res_mask_,
             size * conf_.mask_words_ * sizeof(mask_word_t), cuda_stream_);
    cuda_sync(cuda_stream_);
#if CUDA_PROFILE
    t1 = std::chrono::high_resolution_clock::now();
    p.dd2h_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    t0 = t1;
#endif

    batch_lt_build(first, size);
#if CUDA_PROFILE
    t1 = std::chrono::high_resolution_clock::now();
    p.dbuild_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
#endif

#if CUDA_PROFILE
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - start);
    p.n_ += size;
#endif
  }

  void batch_lt_build(ItrTy first, size_t batch_size) {
#if CUDA_PROFILE
    auto &p(prof_bd.back());
#endif

    for (size_t i = 0; i < batch_size; ++i, ++first) {
      auto &rrr_set(*first);
      rrr_set.reserve(conf_.mask_words_);
      auto res_mask = lt_res_mask_ + (i * conf_.mask_words_);
      if (res_mask[0] != this->G_.num_nodes()) {
        // valid walk
        for (size_t j = 0;
             j < conf_.mask_words_ && res_mask[j] != this->G_.num_nodes();
             ++j) {
          rrr_set.push_back(res_mask[j]);
        }
      } else {
// invalid walk
#if CUDA_PROFILE
        p.num_exceedings_++;
#endif
        auto root = res_mask[1];
        AddRRRSet(this->G_, root, rng_, rrr_set,
                  ripples::linear_threshold_tag{});
      }

      std::stable_sort(rrr_set.begin(), rrr_set.end());
    }
  }

#if CUDA_PROFILE
  struct iter_profile_t {
    size_t n_{0}, num_exceedings_{0};
    std::chrono::nanoseconds d_{0}, dwalk_{0}, dd2h_{0}, dbuild_{0};
  };
  using profile_t = std::vector<iter_profile_t>;
  profile_t prof_bd;

 public:
  void begin_prof_iter() { prof_bd.emplace_back(); }
  void print_prof_iter(size_t i) {
    auto console = spdlog::get("console");
    assert(i < prof_bd.size());
    auto &p(prof_bd[i]);
    if (p.n_) {
      console->info(
          "n-sets={}\tn-exc={}\tns={}\tb={}", p.n_, p.num_exceedings_,
          p.d_.count(),
          (float)p.n_ * 1e03 /
              std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                  .count());
      console->info("walk={}\td2h={}\tbuild={}", p.dwalk_.count(),
                    p.dd2h_.count(), p.dbuild_.count());
      console->info("n. exceedings={} (/{}={})", p.num_exceedings_, p.n_,
                    (float)p.num_exceedings_ / p.n_);
    } else
      console->info("> idle worker");
  }
  void prof_record(typename IMMExecutionRecord::walk_iteration_prof &r,
                   size_t i) {
    assert(i < prof_bd.size());
    typename IMMExecutionRecord::gpu_walk_prof res;
    auto &p(prof_bd[i]);
    res.NumSets = p.n_;
    res.Total = std::chrono::duration_cast<decltype(res.Total)>(p.d_);
    res.Kernel = std::chrono::duration_cast<decltype(res.Kernel)>(p.dwalk_);
    res.D2H = std::chrono::duration_cast<decltype(res.D2H)>(p.dd2h_);
    res.Post = std::chrono::duration_cast<decltype(res.Post)>(p.dbuild_);
    r.GPUWalks.push_back(res);
  }
#endif
};

template <typename GraphTy, typename PRNGeneratorTy, typename ItrTy>
class GPUWalkWorker<GraphTy, PRNGeneratorTy, ItrTy, independent_cascade_tag>
    : public WalkWorker<GraphTy, ItrTy> {
  using vertex_t = typename GraphTy::vertex_type;

  using bfs_solver_t = nvgraph::Bfs<int, PRNGeneratorTy>;

 public:
  struct config_t {
    config_t(size_t num_workers)
        : block_size_(bfs_solver_t::traverse_block_size()),
          max_blocks_(num_workers ? cuda_max_blocks() / num_workers : 0) {
      auto console = spdlog::get("console");
      console->info(
          "> [GPUWalkWorkerIC::config_t] "
          "max_blocks_={}\tblock_size_={}",
          max_blocks_, block_size_);
    }

    size_t num_gpu_threads() const { return max_blocks_ * block_size_; }

    const size_t max_blocks_;
    const size_t block_size_;
  };

  GPUWalkWorker(const config_t &conf, const GraphTy &G,
                const PRNGeneratorTy &rng,
                std::shared_ptr<cuda_ctx<GraphTy>> ctx)
      : WalkWorker<GraphTy, ItrTy>(G),
        conf_(conf),
        rng_(rng),
        u_(0, G.num_nodes()),
        cuda_ctx_(ctx) {
    cuda_set_device(ctx->gpu_id);
    cuda_stream_create(&cuda_stream_);

    // allocate host/device memory
    ic_predecessors_ = (int *)malloc(
        G.num_nodes() * sizeof(typename cuda_device_graph<GraphTy>::vertex_t));
    cuda_malloc(
        (void **)&d_ic_predecessors_,
        G.num_nodes() * sizeof(typename cuda_device_graph<GraphTy>::vertex_t));

    // allocate device-size RNGs
    cuda_malloc((void **)&d_trng_state_,
                conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));

    // create the solver
    solver_ = new bfs_solver_t(
        this->G_.num_nodes(), this->G_.num_edges(),
        cuda_graph_index(cuda_ctx_.get()), cuda_graph_edges(cuda_ctx_.get()),
        cuda_graph_weights(cuda_ctx_.get()), true, TRAVERSAL_DEFAULT_ALPHA,
        TRAVERSAL_DEFAULT_BETA, conf_.max_blocks_, cuda_stream_);
    solver_->configure(nullptr, d_ic_predecessors_, nullptr);
  }

  ~GPUWalkWorker() {
    cuda_set_device(cuda_ctx_->gpu_id);

    delete solver_;
    cuda_stream_destroy(cuda_stream_);

    // free host/device memory
    free(ic_predecessors_);
    cuda_free(d_ic_predecessors_);
    cuda_free(d_trng_state_);
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {
    cuda_set_device(cuda_ctx_->gpu_id);
    cuda_ic_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                      conf_.max_blocks_, conf_.block_size_);
    solver_->rng(d_trng_state_);
  }

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
    // set device and stream
    cuda_set_device(cuda_ctx_->gpu_id);

    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);
      if (last > end) last = end;
      batch(first, last);
    }
  }

 private:
  static constexpr size_t batch_size_ = 32;
  config_t conf_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;

  // CUDA context
  cudaStream_t cuda_stream_;
  std::shared_ptr<cuda_ctx<GraphTy>> cuda_ctx_;

  // nvgraph machinery
  bfs_solver_t *solver_;

  // memory buffers
  typename cuda_device_graph<GraphTy>::vertex_t *ic_predecessors_,
      *d_ic_predecessors_;
  PRNGeneratorTy *d_trng_state_;

  void batch(ItrTy first, ItrTy last) {
#if CUDA_PROFILE
    auto &p(prof_bd.back());
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto size = std::distance(first, last);
    for (size_t wi = 0; wi < size; ++wi) {
#if CUDA_PROFILE
      auto t0 = std::chrono::high_resolution_clock::now();
#endif
      auto root = u_(rng_);
      solver_->traverse(reinterpret_cast<int>(root));
#if CUDA_PROFILE
      cuda_sync(cuda_stream_);
      auto t1 = std::chrono::high_resolution_clock::now();
      p.dwalk_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
      t0 = t1;
#endif

      cuda_d2h(ic_predecessors_, d_ic_predecessors_,
               this->G_.num_nodes() *
                   sizeof(typename cuda_device_graph<GraphTy>::vertex_t),
               cuda_stream_);
      cuda_sync(cuda_stream_);
#if CUDA_PROFILE
      t1 = std::chrono::high_resolution_clock::now();
      p.dd2h_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
      t0 = t1;
#endif

      ic_predecessors_[root] = root;
      ic_build(first++);
#if CUDA_PROFILE
      t1 = std::chrono::high_resolution_clock::now();
      p.dbuild_ +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
#endif
    }
#if CUDA_PROFILE
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }

  void ic_build(ItrTy dst) {
    auto &rrr_set(*dst);
    for (vertex_t i = 0; i < this->G_.num_nodes(); ++i)
      if (ic_predecessors_[i] != -1) rrr_set.push_back(i);
  }

#if CUDA_PROFILE
  struct iter_profile_t {
    size_t n_{0};
    std::chrono::nanoseconds d_{0}, dwalk_{0}, dd2h_{0}, dbuild_{0};
  };
  using profile_t = std::vector<iter_profile_t>;
  profile_t prof_bd;

 public:
  void begin_prof_iter() { prof_bd.emplace_back(); }
  void print_prof_iter(size_t i) {
    auto console = spdlog::get("console");
    assert(i < prof_bd.size());
    auto &p(prof_bd[i]);
    if (p.n_) {
      console->info(
          "n-sets={}\tns={}\tb={}", p.n_, p.d_.count(),
          (float)p.n_ * 1e03 /
              std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                  .count());
      console->info("walk={}\td2h={}\tbuild={}", p.dwalk_.count(),
                    p.dd2h_.count(), p.dbuild_.count());
    } else
      console->info("> idle worker");
  }
  void prof_record(typename IMMExecutionRecord::walk_iteration_prof &r,
                   size_t i) {
    assert(i < prof_bd.size());
    typename IMMExecutionRecord::gpu_walk_prof res;
    auto &p(prof_bd[i]);
    res.NumSets = p.n_;
    res.Total = std::chrono::duration_cast<decltype(res.Total)>(p.d_);
    res.Kernel = std::chrono::duration_cast<decltype(res.Kernel)>(p.dwalk_);
    res.D2H = std::chrono::duration_cast<decltype(res.D2H)>(p.dd2h_);
    res.Post = std::chrono::duration_cast<decltype(res.Post)>(p.dbuild_);
    r.GPUWalks.push_back(res);
  }
#endif
};
#endif  // RIPPLES_ENABLE_CUDA

template <typename GraphTy, typename PRNGeneratorTy, typename ItrTy,
          typename diff_model_tag>
class StreamingRRRGenerator {
  using vertex_t = typename GraphTy::vertex_type;

  using worker_t = WalkWorker<GraphTy, ItrTy>;
  using gpu_worker_t =
      GPUWalkWorker<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag>;
  using cpu_worker_t =
      CPUWalkWorker<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag>;

 public:
  StreamingRRRGenerator(const GraphTy &G, const PRNGeneratorTy &master_rng,
                        IMMExecutionRecord &record, size_t num_cpu_workers,
                        size_t num_gpu_workers,
                        const std::unordered_map<size_t, size_t> &worker_to_gpu)
      : num_cpu_workers_(num_cpu_workers),
        num_gpu_workers_(num_gpu_workers),
        record_(record),
        console(spdlog::get("Streaming Generator")) {
    if (!console) {
      console = spdlog::stdout_color_st("Streaming Generator");
    }
#ifdef RIPPLES_ENABLE_CUDA
    // init GPU contexts
    for (auto &m : worker_to_gpu) {
      auto gpu_id = m.second;
      if (cuda_contexts_.find(gpu_id) == cuda_contexts_.end()) {
        cuda_contexts_[gpu_id] =
            std::shared_ptr<cuda_ctx<GraphTy>>(cuda_make_ctx(G, gpu_id));
      }
    }

    typename gpu_worker_t::config_t gpu_conf(num_gpu_workers_);
    assert(gpu_conf.max_blocks_ * num_gpu_workers_ <= cuda_max_blocks());
    auto num_gpu_threads_per_worker = gpu_conf.num_gpu_threads();
    auto num_rng_sequences =
        num_cpu_workers_ + num_gpu_workers_ * (num_gpu_threads_per_worker + 1);
    auto gpu_seq_offset = num_cpu_workers_ + num_gpu_workers_;
#else
    assert(num_gpu_workers_ == 0);
    size_t num_rng_sequences = num_cpu_workers_;
#endif

    console->info("CPU Workers {}", num_cpu_workers);
    console->info("GPU Workers {}", num_gpu_workers);

    // translate user-mapping string into vector
    size_t gpu_worker_id = 0;
    size_t cpu_worker_id = 0;
    for (size_t omp_num = 0; omp_num < num_cpu_workers + num_gpu_workers;
         ++omp_num) {
#ifdef RIPPLES_ENABLE_CUDA
      if (worker_to_gpu.find(omp_num) != worker_to_gpu.end()) {
        // create and add a GPU worker
        auto gpu_id = worker_to_gpu.at(omp_num);
        assert(cuda_contexts_.find(gpu_id) != cuda_contexts_.end());
        console->info("> mapping: omp={}\t->\tGPU-device={}", omp_num, gpu_id);
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers_ + gpu_worker_id);
        auto w = new gpu_worker_t(gpu_conf, G, rng, cuda_contexts_.at(gpu_id));
        w->rng_setup(
            master_rng, num_rng_sequences,
            gpu_seq_offset + gpu_worker_id * num_gpu_threads_per_worker);
        workers.push_back(w);
        ++gpu_worker_id;
      } else
#endif
      {
        // create and add a CPU worker
        console->info("> mapping: omp={}\t->\tCPU", omp_num);
        console->info("cpu_worker_id = {}", cpu_worker_id);
        auto rng = master_rng;
        rng.split(num_rng_sequences, cpu_worker_id);
        workers.push_back(new cpu_worker_t(G, rng));
        ++cpu_worker_id;
      }
    }

    console->info("Configured");
  }

  StreamingRRRGenerator(StreamingRRRGenerator &&O)
      : num_cpu_workers_(O.num_cpu_workers_),
        num_gpu_workers_(O.num_gpu_workers_),
        max_batch_size_(O.max_batch_size_),
        console(std::move(O.console)),
#if RIPPLES_ENABLE_CUDA
        cuda_contexts_(std::move(O.cuda_contexts_)),
#endif
        workers(std::move(O.workers)),
        mpmc_head(O.mpmc_head.load()),
#if CUDA_PROFILE
        prof_bd(std::move(O.prof_bd)),
#endif
        record_(O.record_) {
  }

  ~StreamingRRRGenerator() {
#if CUDA_PROFILE
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(prof_bd.d);
    console->info("*** BEGIN Streaming Engine profiling");
    for (size_t i = 0; i < prof_bd.prof_bd.size(); ++i) {
      console->info("+++ BEGIN iter {}", i);
      console->info("--- CPU workers");
      for (auto &wp : cpu_workers) wp->print_prof_iter(i);
#ifdef RIPPLES_ENABLE_CUDA
      console->info("--- GPU workers");
      for (auto &wp : gpu_workers) wp->print_prof_iter(i);
#endif
      console->info("--- overall");
      auto &p(prof_bd.prof_bd[i]);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(p.d_);
      console->info("n. sets               = {}", p.n_);
      console->info("elapsed (ns)          = {}", p.d_.count());
      console->info("throughput (sets/sec) = {}",
                    (float)p.n_ * 1e03 / ms.count());
      console->info("+++ END iter {}", i);
      // execution record
      for (auto &wp : workers) {
        wp->prof_record(record_.WalkIterations[i], i);
      }
    }
    console->info("--- overall");
    console->info("n. sets               = {}", prof_bd.n);
    console->info("n. iters              = {}", prof_bd.prof_bd.size());
    console->info("elapsed (ms)          = {}", ms.count());
    console->info("throughput (sets/sec) = {}",
                  (float)prof_bd.n * 1e03 / ms.count());
    console->info("*** END Streaming Engine profiling");
#endif

    for (auto &w : workers) delete w;

#ifdef RIPPLES_ENABLE_CUDA
      // for (auto &m : cuda_contexts_) cuda_destroy_ctx(m.second);
#endif
  }

  IMMExecutionRecord &execution_record() { return record_; }

  void generate(ItrTy begin, ItrTy end) {
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &w : workers) w->begin_prof_iter();
    record_.WalkIterations.emplace_back();
#endif

    mpmc_head.store(0);

#pragma omp parallel num_threads(num_cpu_workers_ + num_gpu_workers_)
    {
      size_t rank = omp_get_thread_num();
      workers[rank]->svc_loop(mpmc_head, begin, end);
    }

#if CUDA_PROFILE
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    prof_bd.prof_bd.emplace_back(std::distance(begin, end), d);
    prof_bd.n += std::distance(begin, end);
    prof_bd.d += std::chrono::duration_cast<std::chrono::microseconds>(d);
    auto &ri(record_.WalkIterations.back());
    ri.NumSets = std::distance(begin, end);
    ri.Total = std::chrono::duration_cast<decltype(ri.Total)>(d);
#endif
  }

  bool isGpuEnabled() const { return num_gpu_workers_ != 0; }

 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  size_t max_batch_size_;
  std::shared_ptr<spdlog::logger> console;
#ifdef RIPPLES_ENABLE_CUDA
  std::unordered_map<size_t, std::shared_ptr<cuda_ctx<GraphTy>>> cuda_contexts_;
#endif
  std::vector<worker_t *> workers;
  std::atomic<size_t> mpmc_head{0};

#if CUDA_PROFILE
  struct iter_profile_t {
    iter_profile_t(size_t n, std::chrono::nanoseconds d) : n_(n), d_(d) {}

    size_t n_{0};
    std::chrono::nanoseconds d_{0};
  };
  struct profile_t {
    size_t n{0};
    std::chrono::microseconds d{0};
    std::vector<iter_profile_t> prof_bd;
  };
  profile_t prof_bd;
#endif
  IMMExecutionRecord &record_;
};
}  // namespace ripples

#endif  // RIPPLES_STREAMING_RRR_GENERATOR_H
