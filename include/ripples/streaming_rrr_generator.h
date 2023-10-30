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
#include <iostream>

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

#ifdef PROFILE_OVERHEAD
std::string output_file_name;
std::atomic<size_t> gpu_id{0};
#endif // PROFILE_OVERHEAD

#ifdef UTILIZATION_PROFILE
std::string output_file_name;
std::atomic<size_t> gpu_id{0};
#endif // UTILIZATION_PROFILE

#if GPU_PROFILE
#include <chrono>
#endif

#define BENCHMARK

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
                        ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin,
                        size_t batch_size) = 0;
  #endif

  virtual size_t batch_size() const = 0;
  
  virtual bool is_cpu() const = 0;

  virtual void benchmark(ItrTy begin, typename std::vector<vertex_t>::iterator root_nodes_begin, size_t worksize, size_t batch_size)  = 0;

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
  CPUWalkWorker(const GraphTy &G, const PRNGeneratorTy &rng, size_t cpu_threads_per_team = 1)
    : WalkWorker<GraphTy, ItrTy>(G), rng_(rng), u_(0, G.num_nodes()), roots_(max_batch_size_),
      cpu_threads_per_team_(cpu_threads_per_team), cpu_ctx_(G.num_nodes()) {}

  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(max_batch_size_)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, max_batch_size_);
      if (last > end) last = end;
      batch(first, last);
    }
  }

  #ifdef REORDERING
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin,
                size_t batch_size = max_batch_size_) {
    size_t offset = 0;
    while ((offset = mpmc_head.fetch_add(batch_size)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size);
      if (last > end) last = end;
      batch(first, last, root_nodes_begin + offset);
    }
  }
  #endif

  size_t batch_size() const { return max_batch_size_; }

  bool is_cpu() const { return true; }

  void benchmark(ItrTy begin, typename std::vector<vertex_t>::iterator root_nodes_begin, size_t worksize, size_t batch_size){
    auto local_rng = rng_;
    for(size_t i = 0; i < worksize; i += batch_size){
      auto first = root_nodes_begin;
      std::advance(first, i);
      auto last = first;
      std::advance(last, batch_size);
      if (last > root_nodes_begin + worksize) last = root_nodes_begin + worksize;
      auto out_begin = std::min(begin + i, begin + worksize);
      BatchedBFSNeighborColorOMP(this->G_, first, last, out_begin, local_rng,
                                 diff_model_tag{}, cpu_ctx_,
                                 cpu_threads_per_team_);
    }
  }

 private:
  static constexpr size_t max_batch_size_ = 64;
  size_t cpu_threads_per_team_;
  BFSCPUContext cpu_ctx_;
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
    std::generate(roots_.begin(), roots_.begin() + size, [&]() { return local_u(local_rng[0][0]); } );
    auto v_start = roots_.begin();
    auto v_end = std::min(v_start + max_batch_size_, v_start + size);
    while (v_start < (roots_.begin() + size)) {
      BatchedBFS(this->G_, v_start, v_end, first, local_rng, diff_model_tag{});
      first += std::distance(v_start, v_end);
      v_start += max_batch_size_;
      v_end = std::min(v_start + max_batch_size_, roots_.begin() + size);
    }

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
    auto local_rng = rng_;
    auto v_end = std::min(v_start + max_batch_size_, v_start + size);
    while (v_start < (root_nodes_first + size)) {
      BatchedBFSNeighborColorOMP(this->G_, v_start, v_end, first, local_rng,
                                 diff_model_tag{}, cpu_ctx_,
                                 cpu_threads_per_team_);
      first += std::distance(v_start, v_end);
      v_start += max_batch_size_;
      v_end = std::min(v_start + max_batch_size_, root_nodes_first + size);
    }
    rng_ = local_rng;
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
                std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> ctx,
                const size_t gpu_batch_size,
                const float pause_threshold = 1.0f)
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
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin,
                size_t batch_size = 1 << 15) {
    GPU<RUNTIME>::set_device(gpu_ctx_->gpu_id);
    size_t offset = 0;
    auto batch_size_lt = conf_.num_gpu_threads();
    while ((offset = mpmc_head.fetch_add(batch_size_lt)) <
           std::distance(begin, end)) {
      auto first = begin;
      std::advance(first, offset);
      auto last = first;
      std::advance(last, batch_size_lt);
      if (last > end) last = end;
      batch(first, last);
    }
  }
  #endif

  size_t batch_size() const { return conf_.num_gpu_threads(); }

  bool is_cpu() const { return false; }

  void benchmark(ItrTy begin, typename std::vector<vertex_t>::iterator root_nodes_begin, size_t worksize, size_t batch_size){}

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
                std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> ctx,
                const size_t gpu_batch_size, const float pause_threshold = 1.0f)
      : WalkWorker<GraphTy, ItrTy>(G),
        rng_(rng),
        u_(0, G.num_nodes()),
        gpu_ctx_(ctx),
        batch_size_(gpu_batch_size),
        pause_threshold_(pause_threshold) {
    #ifdef HIERARCHICAL
    GPUCalculateDegrees(this->G_, *gpu_ctx_, ripples::independent_cascade_tag{},
                        small_frontier_max, medium_frontier_max, large_frontier_max,
                        extreme_frontier_max);
      #ifdef PAUSE_AND_RESUME
      bfs_ctx_ = BFSMultiContext<GraphTy, uint32_t, MAX_COLOR_WIDTH, ItrTy>(this->G_.num_nodes(), small_frontier_max, medium_frontier_max, large_frontier_max, extreme_frontier_max, ctx->gpu_id);
      #else // PAUSE_AND_RESUME
        #ifdef FUSED_COLOR_SET
          bfs_ctx_ = BFSMultiContext<GraphTy, uint32_t, MAX_COLOR_WIDTH>(this->G_.num_nodes(), small_frontier_max, medium_frontier_max, large_frontier_max, extreme_frontier_max, ctx->gpu_id);
        #else // FUSED_COLOR_SET
          bfs_ctx_ = BFSContext<GraphTy, decltype(NumColors)>(this->G_.num_nodes(), small_frontier_max, medium_frontier_max, large_frontier_max, extreme_frontier_max, ctx->gpu_id);
        #endif // FUSED_COLOR_SET
      #endif // PAUSE_AND_RESUME
    #endif // HIERARCHICAL
  }

  ~GPUWalkWorker() {
    // Print BFS Context to file
    #ifdef PROFILE_OVERHEAD
    std::string filename = output_file_name + "_bfs_ctx_" + std::to_string(gpu_ctx_->gpu_id) + "_" + std::to_string(gpu_id++) + ".csv";
    bfs_ctx_.print_to_file(filename);
    #endif // PAUSE_AND_RESUME
    #ifdef UTILIZATION_PROFILE
    std::string filename = output_file_name + "_utilization_" + std::to_string(gpu_ctx_->gpu_id) + "_" + std::to_string(gpu_id++) + ".csv";
    bfs_ctx_.print_utilization_to_file(filename);
    #endif
  }

  #ifdef REORDERING
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {}
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end, typename std::vector<vertex_t>::iterator root_nodes_begin,
  size_t batch_size = 0) {
  #else
  void svc_loop(std::atomic<size_t> &mpmc_head, ItrTy begin, ItrTy end) {
  #endif
    // set device and stream
    size_t offset = 0;
    size_t num_paused = 0;
    while (((offset = mpmc_head.fetch_add(batch_size - num_paused)) <
           std::distance(begin, end)) || num_paused != 0) {
      auto first = begin;
      std::advance(first, offset);
      if(first > end) first = end;
      auto last = first;
      std::advance(last, batch_size - num_paused);
      if (last > end) last = end;
      #ifdef PAUSE_AND_RESUME
      const bool reset = num_paused == 0;
      // If last batch, set threshold to 1.0f to ensure completion.
      const float threshold = last == end ? 1.0f : pause_threshold_;
      num_paused = batch(first, last, root_nodes_begin + offset, threshold, reset, batch_size);
      #else // PAUSE_AND_RESUME
      #ifdef REORDERING
      // std::advance(root_nodes_begin, offset);
      batch(first, last, root_nodes_begin + offset);
      #else // REORDERING
      batch(first, last);
      #endif // REORDERING
      #endif // PAUSE_AND_RESUME
    }
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {}
  
  size_t batch_size() const { return batch_size_; }

  bool is_cpu() const { return false; }

  void benchmark(ItrTy begin, typename std::vector<vertex_t>::iterator root_nodes_begin, size_t worksize, size_t batch_size){
    #ifdef PAUSE_AND_RESUME
    size_t i = 0;
    size_t remaining = 0;
    float threshold = 1.0f;
    while(i < worksize){
      auto first = root_nodes_begin;
      std::advance(first, i);
      auto last = first;
      std::advance(last, batch_size - remaining);
      auto out_begin = std::min(begin + i, begin + worksize);
      bool reset = remaining == 0;
      remaining = GPUBatchedBFSMultiColorFusedReload(this->G_, *gpu_ctx_, first, last,
                  out_begin, ripples::independent_cascade_tag{}, bfs_ctx_, threshold,
                  batch_size, reset);
      i += batch_size - remaining;
    }
    #else // !PAUSE_AND_RESUME
    for(size_t i = 0; i < worksize; i += batch_size){
      auto first = root_nodes_begin;
      std::advance(first, i);
      auto last = first;
      std::advance(last, batch_size);
      if (last > root_nodes_begin + worksize) last = root_nodes_begin + worksize;
      auto out_begin = std::min(begin + i, begin + worksize);
      #ifdef FUSED_COLOR_SET
      GPUBatchedBFSMultiColorFused(this->G_, *gpu_ctx_, first, last,
                  out_begin, ripples::independent_cascade_tag{}, bfs_ctx_);
      #else
      GPUBatchedTieredQueueBFS(this->G_, *gpu_ctx_, first, last,
                  out_begin, ripples::independent_cascade_tag{}, bfs_ctx_, NumColors);
      #endif
    }
    #endif // !PAUSE_AND_RESUME
  }

 private:
  size_t batch_size_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;
  std::shared_ptr<gpu_ctx<RUNTIME, GraphTy>> gpu_ctx_;
  #ifdef HIERARCHICAL
  int small_frontier_max, medium_frontier_max, large_frontier_max, extreme_frontier_max;
  uint64_t NumColors = sizeof(uint64_t) * 8;
  float pause_threshold_ = 1.0f;
  #ifdef PAUSE_AND_RESUME
  BFSMultiContext<GraphTy, uint32_t, MAX_COLOR_WIDTH, ItrTy>  bfs_ctx_;
  #else // PAUSE_AND_RESUME
  #ifdef FUSED_COLOR_SET
  BFSMultiContext<GraphTy, uint32_t, MAX_COLOR_WIDTH>  bfs_ctx_;
  #else // FUSED_COLOR_SET
  BFSContext<GraphTy, uint64_t>  bfs_ctx_;
  #endif // FUSED_COLOR_SET
  #endif // PAUSE_AND_RESUME
  #endif // HIERARCHICAL

  #ifdef PAUSE_AND_RESUME
  size_t batch(ItrTy first, ItrTy last, typename std::vector<vertex_t>::iterator root_nodes_first, float threshold, bool reset = true, size_t batch_size = 64) {
    auto size = std::distance(first, last);
    auto roots_begin = root_nodes_first;
    auto roots_end = root_nodes_first + size;
    #ifdef PROFILE_OVERHEAD
    return GPUBatchedBFSMultiColorFusedReload(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{}, bfs_ctx_, threshold,
                  batch_size, reset);
    #else // !PROFILE_OVERHEAD
    if((threshold == 1.0) && (reset == true)){
      // std::cout << "Running non-reload version" << std::endl;
      GPUBatchedBFSMultiColorFused(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{}, bfs_ctx_);
      return 0;
    }
    else{
      // std::cout << "Running reload version" << std::endl;
      return GPUBatchedBFSMultiColorFusedReload(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{}, bfs_ctx_, threshold,
                  batch_size, reset);
    }
    #endif // !PROFILE_OVERHEAD
  }

  #else // PAUSE_AND_RESUME

  #ifdef REORDERING
  void batch(ItrTy first, ItrTy last, typename std::vector<vertex_t>::iterator root_nodes_first) {
  #else
  void batch(ItrTy first, ItrTy last) {
  #endif
    auto size = std::distance(first, last);
    #ifdef REORDERING
    auto roots_begin = root_nodes_first;
    auto roots_end = root_nodes_first + size;
    #else
    std::vector<vertex_t> roots(size);
    trng::uniform_int_dist u(0, this->G_.num_nodes());
    std::generate(roots.begin(), roots.end(), [&]() { return u_(rng_); });
    auto roots_begin = std::begin(roots);
    auto roots_end = std::end(roots);
    #endif

    #if defined(FUSED_COLOR_SET)
    GPUBatchedBFSMultiColorFused(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{}, bfs_ctx_);
    #elif defined(HIERARCHICAL)
    GPUBatchedTieredQueueBFS(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{}, bfs_ctx_, NumColors);
    #elif defined(EXPERIMENTAL_SCAN_BFS)
    GPUBatchedScanBFS(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{});
    #else
    GPUBatchedBFS(this->G_, *gpu_ctx_, roots_begin, roots_end,
                  first, ripples::independent_cascade_tag{});
    #endif // FUSED_COLOR_SET
  }
  #endif // PAUSE_AND_RESUME
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
      CPUWalkWorker<GraphTy, std::vector<std::vector<PRNGeneratorTy>>, ItrTy, diff_model_tag>;

 public:
  StreamingRRRGenerator(const GraphTy &G, const PRNGeneratorTy &master_rng,
                        size_t num_cpu_workers,
                        size_t num_cpu_teams,
                        size_t num_gpu_workers,
                        size_t gpu_batch_size,
                        size_t cpu_batch_size,
                        const std::unordered_map<size_t, size_t> &worker_to_gpu,
                        float pause_threshold = 1.0f)
      : num_cpu_workers_(num_cpu_workers),
        num_cpu_teams_(num_cpu_teams),
        num_gpu_workers_(num_gpu_workers),
        gpu_batch_size_(gpu_batch_size),
        pause_threshold_(pause_threshold),
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
        num_cpu_workers_ * cpu_batch_size + num_gpu_workers_ * (num_gpu_threads_per_worker + 1);
    auto gpu_seq_offset = num_cpu_workers_ * cpu_batch_size + num_gpu_workers_;
#else
    assert(num_gpu_workers_ == 0);
    size_t num_rng_sequences = num_cpu_workers_ * cpu_batch_size;
#endif

    console->info("CPU Workers {}", num_cpu_workers);
    console->info("GPU Workers {}", num_gpu_workers);

    // translate user-mapping string into vector
    size_t gpu_worker_id = 0;
    size_t cpu_worker_id = 0;
    if(num_cpu_teams_){
      cpu_threads_per_team_ = num_cpu_workers_ / num_cpu_teams_;
    }
    else{
      cpu_threads_per_team_ = 1;
      num_cpu_teams_ = num_cpu_workers_;
    }
    for (size_t omp_num = 0; omp_num < num_cpu_teams_ + num_gpu_workers;
         ++omp_num) {
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
      if (worker_to_gpu.find(omp_num) != worker_to_gpu.end()) {
        // create and add a GPU worker
        auto gpu_id = worker_to_gpu.at(omp_num);
        assert(gpu_contexts_.find(gpu_id) != gpu_contexts_.end());
        console->info("> mapping: omp={}\t->\tGPU-device={}", omp_num, gpu_id);
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers_ + gpu_worker_id);
        auto w = new gpu_worker_t(gpu_conf, G, rng, gpu_contexts_.at(gpu_id), gpu_batch_size_, pause_threshold);
        w->rng_setup(
            master_rng, num_rng_sequences,
            gpu_seq_offset + gpu_worker_id * num_gpu_threads_per_worker);
        gpu_batch_size_ = w->batch_size();
        workers.push_back(w);
        ++gpu_worker_id;
      } else
#endif
      {
        // create and add a CPU worker
        console->info("> mapping: omp={}\t->\tCPU", omp_num);
        console->info("cpu_worker_id = {}", cpu_worker_id);
        std::vector<std::vector<PRNGeneratorTy>> rng(cpu_threads_per_team_, std::vector<PRNGeneratorTy>(cpu_batch_size));
        #pragma omp parallel for
        for (size_t i = 0; i < cpu_threads_per_team_; ++i) {
          for (size_t j = 0; j < cpu_batch_size; ++j){
            rng[i][j] = master_rng;
            rng[i][j].split(num_rng_sequences, cpu_worker_id * cpu_threads_per_team_ * cpu_batch_size + i * cpu_batch_size + j);
          }
        }
        workers.push_back(new cpu_worker_t(G, rng, cpu_threads_per_team_));
        if(cpu_batch_size == 0){
          cpu_batch_size_ = workers.back()->batch_size();
        }
        else{
          cpu_batch_size_ = cpu_batch_size;
        }
        ++cpu_worker_id;
      }
    }

    console->info("Configured");
  }

  StreamingRRRGenerator(StreamingRRRGenerator &&O)
      : num_cpu_workers_(O.num_cpu_workers_),
        num_cpu_teams_(O.num_cpu_teams_),
        num_gpu_workers_(O.num_gpu_workers_),
        cpu_batch_size_(O.cpu_batch_size_),
        gpu_batch_size_(O.gpu_batch_size_),
        max_batch_size_(O.max_batch_size_),
        console(std::move(O.console)),
        pause_threshold_(O.pause_threshold_),
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

    // // Figure out if the total batch size is larger than the work needed to be executed
    // size_t total_batch_size = num_cpu_workers_ * cpu_batch_size_ + num_gpu_workers_ * gpu_batch_size_;
    // size_t work_size_ = std::distance(begin, end);
    // size_t new_gpu_batch_size_ = gpu_batch_size_;
    // size_t new_cpu_batch_size_ = cpu_batch_size_;
    // if (total_batch_size > work_size_) {
    //   // If so, we need to adjust the batch sizes, prioritizing the GPU workers
    //   if (num_gpu_workers_ > 0) {
    //     new_gpu_batch_size_ = std::min((std::distance(begin, end) + num_gpu_workers_ - 1) / num_gpu_workers_, gpu_batch_size_);
    //   }
    //   work_size_ -= num_gpu_workers_ * new_gpu_batch_size_;
    //   if (num_cpu_workers_ > 0) {
    //     new_cpu_batch_size_ = std::min((work_size_ + num_cpu_workers_ - 1) / num_cpu_workers_, cpu_batch_size_);
    //   }
    // }
    // Set omp max levels to 3 to allow for nested parallelism
    if(num_cpu_teams_){
      omp_set_max_active_levels(3);
      #pragma omp parallel num_threads(num_cpu_teams_) proc_bind(spread)
      {
        size_t rank_outer = omp_get_thread_num();
        if(num_gpu_workers_){
          // CPU + GPU
          #pragma omp parallel num_threads(1 + num_gpu_workers_/num_cpu_teams_) proc_bind(close)
          {
            size_t rank_inner = omp_get_thread_num();
            size_t rank = rank_outer*(1 + num_gpu_workers_/num_cpu_teams_) + rank_inner;
            // Convert above into printf
            // printf("OMP Rank: %d | HW Thread: %d\n", rank, sched_getcpu());
            #ifdef REORDERING
            if(workers[rank]->is_cpu()){
              workers[rank]->svc_loop(mpmc_head, begin, end, root_nodes.begin(), cpu_batch_size_);
            }
            else{
              workers[rank]->svc_loop(mpmc_head, begin, end, root_nodes.begin(), gpu_batch_size_);
            }
            #else
            workers[rank]->svc_loop(mpmc_head, begin, end);
            #endif
          }
        }
        // CPU Only
        else{
          #ifdef REORDERING
          workers[rank_outer]->svc_loop(mpmc_head, begin, end, root_nodes.begin(), cpu_batch_size_);
          #else
          workers[rank_outer]->svc_loop(mpmc_head, begin, end);
          #endif
        }
      }
    }
    else{
      // GPU Only
      #pragma omp parallel num_threads(num_gpu_workers_) proc_bind(spread)
      {
        #ifdef REORDERING
        workers[omp_get_thread_num()]->svc_loop(mpmc_head, begin, end, root_nodes.begin(), gpu_batch_size_);
        #else
        workers[omp_get_thread_num()]->svc_loop(mpmc_head, begin, end);
        #endif
      }
    }

    #if 0
    // Check begin through end to ensure none of the vectors are empty
    size_t empty_vectors = 0;
    for(auto it = begin; it != end; it++) {
      if (it->empty()) {
        empty_vectors++;
      }
    }
    std::cout << "Empty vectors found: " << empty_vectors << std::endl;
    #endif // 0 or 1

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
  #if 0
  // ensure vectors from begin to end are not empty
  for (auto it = begin; it != end; it++) {
    assert(!it->empty());
  }
  #endif
  }

  bool isGpuEnabled() const { return num_gpu_workers_ != 0; }

  #if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
  void benchmark(size_t num_batches, size_t iterations, IMMExecutionRecord &record){
    console->info("Microbenchmarking");
    // Measure time of for loop
    auto micro_start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < iterations; ++i){
      // Benchmark the CPU and GPU workers
      size_t work_size = (num_gpu_workers_ * gpu_batch_size_ + num_cpu_teams_ * cpu_batch_size_)*num_batches;
      std::vector<vertex_t> root_nodes(work_size);
      std::vector<RRRset<GraphTy>> RR_bench(work_size);
      
      std::generate(root_nodes.begin(), root_nodes.end(), [&]() { return u_(master_rng_); });
      #ifdef SORTING
      std::sort(root_nodes.begin(), root_nodes.end());
      #endif

      // Atomic GPU time
      std::atomic<size_t> gpu_time(0);

      // Atomic CPU time
      std::atomic<size_t> cpu_time(0);

      std::vector<size_t> offsets(num_cpu_teams_ + num_gpu_workers_);
      offsets[0] = 0;
      for(size_t i = 1; i < num_cpu_teams_ + num_gpu_workers_; i++){
        size_t batch_size = workers[i-1]->is_cpu() ? cpu_batch_size_ : gpu_batch_size_;
        offsets[i] = offsets[i-1] + batch_size*num_batches;
      }

      // // Print batch_sizes
      // std::cout << "CPU batch size: " << cpu_batch_size_ << std::endl;
      // std::cout << "GPU batch size: " << gpu_batch_size_ << std::endl;
      omp_set_max_active_levels(3);
      #pragma omp parallel num_threads(num_cpu_teams_) proc_bind(spread)
      {
        size_t rank_outer = omp_get_thread_num();
        #pragma omp parallel num_threads(1 + num_gpu_workers_/num_cpu_teams_) proc_bind(close)
        {
          size_t rank_inner = omp_get_thread_num();
          size_t rank = rank_outer*(1 + num_gpu_workers_/num_cpu_teams_) + rank_inner;
          // Convert above into printf
          // printf("OMP Rank: %d | HW Thread: %d\n", rank, sched_getcpu());
          bool is_cpu = workers[rank]->is_cpu();
          size_t batch_size = is_cpu ? cpu_batch_size_ : gpu_batch_size_;
          // Time below section
          auto start = std::chrono::high_resolution_clock::now();
          workers[rank]->benchmark(RR_bench.begin() + offsets[rank], root_nodes.begin() + offsets[rank], batch_size*num_batches, batch_size);
          auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start);
          if(is_cpu){
            cpu_time += d.count();
          } else {
            gpu_time += d.count();
          }
        }
      }

      auto old_cpu_batch_size = cpu_batch_size_;

      // Average cpu and gpu times per set
      size_t cpu_avg = (cpu_time / num_cpu_teams_);
      size_t gpu_avg = (gpu_time / num_gpu_workers_);
      // std::cout << "CPU avg: " << cpu_avg << " ns" << std::endl;
      // std::cout << "GPU avg: " << gpu_avg << " ns" << std::endl;

      // Adjust CPU batch size to be roughly the same as the GPU batch size
      cpu_batch_size_ = std::min((size_t)std::round(((double)cpu_batch_size_ * (double)gpu_avg / (double)cpu_avg)), (size_t)64);
      // Print adjusted
      // std::cout << "Adjusted CPU batch size: " << cpu_batch_size_ << std::endl;
      if(cpu_batch_size_ == old_cpu_batch_size){
        break;
      }
    }
    auto micro_end = std::chrono::high_resolution_clock::now();
    console->info("Adjusted CPU batch size to {}", cpu_batch_size_);

    record.Microbenchmarking = micro_end - micro_start;
    record.CPUBatchSize = cpu_batch_size_;
  }
#endif // RIPPLES_ENABLE_CUDA || RIPPLES_ENABLE_HIP

 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  size_t num_cpu_teams_, cpu_threads_per_team_;
  size_t cpu_batch_size_ = 64;
  size_t gpu_batch_size_ = 1024;
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
  float pause_threshold_;

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
