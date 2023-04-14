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
#include "ripples/batched_add_rrrset.h"
#include "ripples/add_rrrset.h"

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
#include "ripples/gpu/bfs.h"
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"
#include "ripples/gpu/generate_rrr_sets.h"
#endif

#if defined(RIPPLES_ENABLE_CUDA)
#define RUNTIME CUDA
#elif defined(RIPPLES_ENABLE_HIP)
#define RUNTIME HIP
#endif

#if GPU_PROFILE
#include <chrono>
#endif

namespace ripples {

template <typename GraphTy, typename ItrTy>
class WalkWorker {
  using vertex_t = typename GraphTy::vertex_type;

 public:
  WalkWorker(const GraphTy &G) : G_(G) {}
  virtual ~WalkWorker() {}
  virtual void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin,
                        ItrTy end) = 0;
  
  #ifdef REORDERING
  virtual void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin,
                        ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin) = 0;
  #endif

 protected:
  const GraphTy &G_;

#if GPU_PROFILE
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
    : WalkWorker<GraphTy, ItrTy>(G), rng_(rng), u_(0, G.num_nodes()), roots_(batch_size_) {}

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

  #ifdef REORDERING
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);
      if (last > end) last = end;
      batch(first, last, root_nodes_begin + offset);
    }
  }
  #endif

 private:
  static constexpr size_t batch_size_ = 64;
  std::vector<vertex_t> roots_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;

  void batch(ItrTy first, ItrTy last) {
#if GPU_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto size = std::distance(first, last);
    auto local_rng = rng_;
    auto local_u = u_;
    #if 0
    for (; first != last; ++first) {
      vertex_t root = local_u(local_rng);

      AddRRRSet(this->G_, root, local_rng, *first, diff_model_tag{});
    }
    #else
    std::generate(roots_.begin(), roots_.begin() + size, [&]() { return local_u(local_rng); } );
    auto v_start = roots_.begin();
    auto v_end = std::min(v_start + 64, roots_.begin() + size);
    while (v_start < (roots_.begin() + size)) {
      // std::cout << "CPU Processing " << size << std::endl;
      BatchedBFS(this->G_, v_start, v_end, first, local_rng, diff_model_tag{});
      // std::cout << "CPU Processed " << size << std::endl;

      first += std::distance(v_start, v_end);
      v_start += 64;
      v_end = std::min(v_start + 64, roots_.begin() + size);
    }
    #endif

    rng_ = local_rng;
    u_ = local_u;
#if GPU_PROFILE
    auto &p(prof_bd.back());
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }

#ifdef REORDERING
  void batch(ItrTy first, ItrTy last, typename std::vector<vertex_t>::iterator root_nodes_first) {
#if GPU_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto size = std::distance(first, last);
    auto v_start = root_nodes_first;
    // auto v_end = root_nodes_first + size;
    auto local_rng = rng_;
    // auto local_u = u_;
    // std::generate(roots_.begin(), roots_.begin() + size, [&]() { return local_u(local_rng); } );
    // auto v_start = roots_.begin();
    auto v_end = std::min(v_start + 64, v_start + size);
    while (v_start < (root_nodes_first + size)) {
      // std::cout << "CPU Processing " << size << std::endl;
      BatchedBFS(this->G_, v_start, v_end, first, local_rng, diff_model_tag{});
      // std::cout << "CPU Processed " << size << std::endl;

      first += std::distance(v_start, v_end);
      v_start += 64;
      v_end = std::min(v_start + 64, root_nodes_first + size);
    }
    rng_ = local_rng;
    // u_ = local_u;
#if GPU_PROFILE
    auto &p(prof_bd.back());
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }
  #endif

#if GPU_PROFILE
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

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
template <typename GraphTy, typename PRNGeneratorTy, typename ItrTy>
class GPUWalkWorker<GraphTy, PRNGeneratorTy, ItrTy, linear_threshold_tag>
    : public WalkWorker<GraphTy, ItrTy> {
  using vertex_t = typename GraphTy::vertex_type;

 public:
  struct config_t {
    config_t(size_t num_gpu_workers) {
      auto console = spdlog::get("console");
      assert(num_threads_ % block_size_ == 0);
      max_blocks_ = num_threads_ / block_size_;
      assert(max_blocks_ * num_gpu_workers <= GPU<RUNTIME>::max_blocks());
#if GPU_PROFILE
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
                std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> ctx)
      : WalkWorker<GraphTy, ItrTy>(G),
        conf_(conf),
        rng_(rng),
        u_(0, G.num_nodes()),
        gpu_ctx_(ctx) {
    GPU<RUNTIME>::set_device(ctx->gpu_id);
    gpu_stream_ = GPU<RUNTIME>::create_stream();

    // allocate host/device memory
    auto mask_size = conf.mask_words_ * sizeof(mask_word_t);
    lt_res_mask_ = (mask_word_t *)malloc(conf_.num_gpu_threads() * mask_size);
    GPU<RUNTIME>::device_malloc((void **)&d_lt_res_mask_,
                                conf_.num_gpu_threads() * mask_size);

    // allocate device-size RNGs
    GPU<RUNTIME>::device_malloc(
        (void **)&d_trng_state_,
        conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));
  }

  ~GPUWalkWorker() {
    GPU<RUNTIME>::set_device(gpu_ctx_->gpu_id);
    GPU<RUNTIME>::destroy_stream(gpu_stream_);
    // free host/device memory
    free(lt_res_mask_);
    GPU<RUNTIME>::device_free(d_lt_res_mask_);
    GPU<RUNTIME>::device_free(d_trng_state_);
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {
    GPU<RUNTIME>::set_device(gpu_ctx_->gpu_id);
    gpu_lt_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                     conf_.max_blocks_, conf_.block_size_);
  }

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
    GPU<RUNTIME>::set_device(gpu_ctx_->gpu_id);
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
  #ifdef REORDERING
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin) {}
  #endif

 private:
  config_t conf_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;
  typename GPU<RUNTIME>::stream_type gpu_stream_;
  std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> gpu_ctx_;

  // memory buffers
  using mask_word_t = int;  // TODO: We should abstract it.
  mask_word_t *lt_res_mask_, *d_lt_res_mask_;
  PRNGeneratorTy *d_trng_state_;

  void batch(ItrTy first, ItrTy last) {
#if GPU_PROFILE
    auto &p(prof_bd.back());
    auto start = std::chrono::high_resolution_clock::now();
#endif
    auto size = std::distance(first, last);

    gpu_lt_kernel<RUNTIME>(conf_.max_blocks_, conf_.block_size_, size,
                  this->G_.num_nodes(), d_trng_state_, d_lt_res_mask_,
                  conf_.mask_words_, gpu_ctx_.get(), gpu_stream_);
#if GPU_PROFILE
    GPU<RUNTIME>::stream_sync(gpu_stream_);
    auto t1 = std::chrono::high_resolution_clock::now();
    p.dwalk_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - start);
    auto t0 = t1;
#endif

    GPU<RUNTIME>::d2h(lt_res_mask_, d_lt_res_mask_,
                      size * conf_.mask_words_ * sizeof(mask_word_t),
                      gpu_stream_);
    GPU<RUNTIME>::stream_sync(gpu_stream_);
#if GPU_PROFILE
    t1 = std::chrono::high_resolution_clock::now();
    p.dd2h_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    t0 = t1;
#endif

    batch_lt_build(first, size);
#if GPU_PROFILE
    t1 = std::chrono::high_resolution_clock::now();
    p.dbuild_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
#endif

#if GPU_PROFILE
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - start);
    p.n_ += size;
#endif
  }

  void batch_lt_build(ItrTy first, size_t batch_size) {
#if GPU_PROFILE
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
#if GPU_PROFILE
        p.num_exceedings_++;
#endif
        auto root = res_mask[1];
        AddRRRSet(this->G_, root, rng_, rrr_set,
                  ripples::linear_threshold_tag{});
      }

      std::sort(rrr_set.begin(), rrr_set.end());
    }
  }

#if GPU_PROFILE
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

 public:
  struct config_t {
    config_t(size_t) {}

    // This should be unused here.
    size_t num_gpu_threads() const { return 1 << 15; }
  };
  GPUWalkWorker(const config_t &conf, const GraphTy &G,
                const PRNGeneratorTy &rng,
                std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> ctx)
      : WalkWorker<GraphTy, ItrTy>(G),
        rng_(rng),
        u_(0, G.num_nodes()),
        gpu_ctx_(ctx) {
    #ifdef HIERARCHICAL
    GPUCalculateDegrees(this->G_, *gpu_ctx_, ripples::independent_cascade_tag{},
                        small_frontier_max, medium_frontier_max, large_frontier_max,
                        extreme_frontier_max);
    #endif
  }

  #ifdef REORDERING
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {}
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin) {
  #else
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
  #endif
    // set device and stream
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size_)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_);
      if (last > end) last = end;
      #ifdef REORDERING
      // std::advance(root_nodes_begin, offset);
      batch(first, last, root_nodes_begin + offset);
      #else
      batch(first, last);
      #endif
      // std::cout << "GPUWalkWorker::svc_loop: " << offset << std::endl;
    }
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {}

 private:
  // static constexpr size_t batch_size_ = 16;
  // static constexpr size_t batch_size_ = 32;
  // static constexpr size_t batch_size_ = 1;
  static constexpr size_t batch_size_ = 64;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;
  std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> gpu_ctx_;
  #ifdef HIERARCHICAL
  int small_frontier_max, medium_frontier_max, large_frontier_max, extreme_frontier_max;
  #endif
  // Frontier<GraphTy> frontier, new_frontier;

  #ifdef REORDERING
  void batch(ItrTy first, ItrTy last, typename std::vector<vertex_t>::iterator root_nodes_first) {
  #else
  void batch(ItrTy first, ItrTy last) {
  #endif
    auto size = std::distance(first, last);
    #ifdef REORDERING
    auto roots_begin = root_nodes_first;
    auto roots_end = root_nodes_first + size;
    // Print the root nodes
    // std::cout << "Root nodes: ";
    // for (auto it = roots_begin; it != roots_end; it++) {
    //   std::cout << *it << " ";
    // }
    // std::cout << std::endl;
    #else
    std::vector<vertex_t> roots(size);
    trng::uniform_int_dist u(0, this->G_.num_nodes());
    std::generate(roots.begin(), roots.end(), [&]() { return u_(rng_); });
    auto roots_begin = std::begin(roots);
    auto roots_end = std::end(roots);
    #endif

    // std::cout << "-----GPU Processing " << size << std::endl;
    #if defined(HIERARCHICAL)
    uint64_t NumColors = sizeof(uint64_t) * 8;
    // uint32_t NumColors = sizeof(uint32_t) * 8;
    GPUBatchedTieredQueueBFS(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{}, small_frontier_max, medium_frontier_max, large_frontier_max,
                        extreme_frontier_max, NumColors);
    #elif defined(EXPERIMENTAL_SCAN_BFS)
    GPUBatchedScanBFS(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{});
    #else
    GPUBatchedBFS(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{});
    #endif
    // std::cout << "-----GPU Processed " << size << std::endl;
  }
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
                        size_t num_cpu_workers,
                        size_t num_gpu_workers,
                        const std::unordered_map<size_t, size_t> &worker_to_gpu)
      : num_cpu_workers_(num_cpu_workers),
        num_gpu_workers_(num_gpu_workers),
        console(spdlog::get("Streaming Generator")),
        master_rng_(master_rng),
        u_(0, G.num_nodes()){
    if (!console) {
      console = spdlog::stdout_color_st("Streaming Generator");
    }
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
    // init GPU contexts
    for (auto &m : worker_to_gpu) {
      auto gpu_id = m.second;
      if (gpu_contexts_.find(gpu_id) == gpu_contexts_.end()) {
        gpu_contexts_[gpu_id] = std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>>(
            make_gpu_context<RUNTIME>(G, gpu_id));
      }
    }

    typename gpu_worker_t::config_t gpu_conf(num_gpu_workers_);
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
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
      if (worker_to_gpu.find(omp_num) != worker_to_gpu.end()) {
        // create and add a GPU worker
        auto gpu_id = worker_to_gpu.at(omp_num);
        assert(gpu_contexts_.find(gpu_id) != gpu_contexts_.end());
        console->info("> mapping: omp={}\t->\tGPU-device={}", omp_num, gpu_id);
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers_ + gpu_worker_id);
        auto w = new gpu_worker_t(gpu_conf, G, rng, gpu_contexts_.at(gpu_id));
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
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
        gpu_contexts_(std::move(O.gpu_contexts_)),
#endif
        workers(std::move(O.workers)),
        mpmc_head(O.mpmc_head.load())
#if GPU_PROFILE
      , prof_bd(std::move(O.prof_bd))
#endif
  {}

  ~StreamingRRRGenerator() {
#if GPU_PROFILE
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(prof_bd.d);
    console->info("*** BEGIN Streaming Engine profiling");
    for (size_t i = 0; i < prof_bd.prof_bd.size(); ++i) {
      console->info("+++ BEGIN iter {}", i);
      console->info("--- CPU workers");
      for (auto &wp : cpu_workers) wp->print_prof_iter(i);
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
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
    }
    console->info("--- overall");
    console->info("n. sets               = {}", prof_bd.n);
    console->info("n. iters              = {}", prof_bd.prof_bd.size());
    console->info("elapsed (ms)          = {}", ms.count());
    console->info("throughput (sets/sec) = {}",
                  (float)prof_bd.n * 1e03 / ms.count());
    console->info("*** END Streaming Engine profiling");
#endif
#ifdef FRONTIER_PROFILE
  std::ofstream profileoutput;
  #if defined(HIERARCHICAL)
  profileoutput.open("hier_bfs_prof.csv", std::ios::out);
  #elif defined(EXPERIMENTAL_SCAN_BFS)
  profileoutput.open("scan_bfs_prof.csv", std::ios::out);
  #else
  profileoutput.open("sort_bfs_prof.csv", std::ios::out);
  #endif
  for(auto entry : profile_vector){
    // Output contents of entry to csv
    profileoutput <<
      entry.frontier_size << "," <<
      entry.frontier_time << "," <<
      entry.frontier_colors << "," <<
      entry.old_frontier_size << "," <<
      entry.scatter_time << "," << 
      entry.max_outdegree << "," <<
      entry.iteration << "," <<
      entry.edge_colors << "," <<
      entry.unique_colors << "\n";
  }
  profileoutput.close();
#endif

    for (auto &w : workers) delete w;

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
      // for (auto &m : gpu_contexts_) cuda_destroy_ctx(m.second);
#endif
  }

  void generate(ItrTy begin, ItrTy end, IMMExecutionRecord &record) {
#if GPU_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &w : workers) w->begin_prof_iter();
    record.WalkIterations.emplace_back();
#endif

    mpmc_head.store(0);

#ifdef REORDERING
    // Pregenerate random numbers for reordering
    std::vector<vertex_t> root_nodes(std::distance(begin, end));
    std::generate(root_nodes.begin(), root_nodes.end(), [&]() { return u_(master_rng_); });
    #ifdef SORTING
    std::sort(root_nodes.begin(), root_nodes.end());
    #endif
#endif

#pragma omp parallel num_threads(num_cpu_workers_ + num_gpu_workers_)
    {
      size_t rank = omp_get_thread_num();
      // std::cout << "rank = " << rank << std::endl;
      #ifdef REORDERING
      workers[rank]->svc_loop(mpmc_head, begin, end, root_nodes.begin());
      #else
      workers[rank]->svc_loop(mpmc_head, begin, end);
      #endif
    }

#if GPU_PROFILE
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    prof_bd.prof_bd.emplace_back(std::distance(begin, end), d);
    prof_bd.n += std::distance(begin, end);
    prof_bd.d += std::chrono::duration_cast<std::chrono::microseconds>(d);
    auto &ri(record.WalkIterations.back());
    ri.NumSets = std::distance(begin, end);
    ri.Total = std::chrono::duration_cast<decltype(ri.Total)>(d);
#endif
  }

  bool isGpuEnabled() const { return num_gpu_workers_ != 0; }

 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  size_t max_batch_size_;
  std::shared_ptr<spdlog::logger> console;
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
  std::unordered_map<size_t, std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>>>
      gpu_contexts_;
#endif
  std::vector<worker_t *> workers;
  std::atomic<size_t> mpmc_head{0};
  PRNGeneratorTy master_rng_;
  trng::uniform_int_dist u_;

#if GPU_PROFILE
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
};
}  // namespace ripples

#undef RUNTIME

#endif  // RIPPLES_STREAMING_RRR_GENERATOR_H
