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

#ifndef RIPPLES_CUDA_STREAMING_RRR_GENERATOR_OMP_H
#define RIPPLES_CUDA_STREAMING_RRR_GENERATOR_OMP_H

#include <algorithm>
#include <cstdlib>
#include <vector>

#if CUDA_PROFILE
#include <chrono>
#endif

#include "spdlog/spdlog.h"
#include "trng/uniform_int_dist.hpp"

#include "ripples/generate_rrr_sets.h"

namespace ripples {

template <typename GraphTy, typename PRNGeneratorTy>
class StreamingRRRGenerator {
  using rrr_set_t = std::vector<typename GraphTy::vertex_type>;
  using rrr_sets_t = std::vector<rrr_set_t>;

  class Worker {
   public:
    virtual ~Worker() {}

    virtual void batch(rrr_set_t *first, size_t size,
                       ripples::linear_threshold_tag &&, size_t rank) = 0;
    virtual void batch(rrr_set_t *first, size_t size,
                       ripples::independent_cascade_tag &&, size_t rank) = 0;

#if CUDA_PROFILE
    struct iter_profile_t {
      size_t n_{0}, num_exceedings_{0};
      std::chrono::nanoseconds d_{0};
    };
    using profile_t = std::vector<iter_profile_t>;
    profile_t prof_bd;

    void begin_prof_iter() { prof_bd.emplace_back(); }
#endif
  };

  class CPUWorker : public Worker {
   public:
    CPUWorker(const GraphTy &G, const PRNGeneratorTy &rng)
        : G_(G), rng_(rng), u_(0, G_.num_nodes()) {}

   private:
    const GraphTy &G_;
    PRNGeneratorTy rng_;
    trng::uniform_int_dist u_;

    void batch(rrr_set_t *first, size_t size,
               ripples::independent_cascade_tag &&m, size_t rank) {
      batch_dispatcher(first, size,
                       std::forward<ripples::independent_cascade_tag>(m), rank);
    }

    void batch(rrr_set_t *first, size_t size, ripples::linear_threshold_tag &&m,
               size_t rank) {
      batch_dispatcher(first, size,
                       std::forward<ripples::linear_threshold_tag>(m), rank);
    }

    template <typename diff_model_tag>
    void batch_dispatcher(rrr_set_t *first, size_t size, diff_model_tag &&model,
                          size_t rank) {
#if CUDA_PROFILE
      auto start = std::chrono::high_resolution_clock::now();
#endif
      for (auto last = first + size; first != last; ++first) {
        typename GraphTy::vertex_type root = u_(rng_);
        AddRRRSet(G_, root, rng_, *first, std::forward<diff_model_tag>(model));
      }
#if CUDA_PROFILE
      auto &p(this->prof_bd.back());
      p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - start);
      p.n_ += size;
      spdlog::get("console")->info(
          "> [CPUWorker @{} batch_dispatcher] profile size={} n={} d={}",
          (void *)this, this->prof_bd.size(), p.n_, p.d_.count());
#endif
    }
  };

  class GPUWorker : public Worker {
   public:
    struct config_t {
      config_t() {
        auto CFG = configuration();

        // configuration parameters
        mask_words_ = 8;
        num_active_threads_ = CFG.cuda_num_threads;
        num_warps_per_block_ = CFG.cuda_block_density;
        num_active_threads_per_warp_ = CFG.cuda_warp_density;

        // number of (active) blocks
        auto num_active_threads_per_block =
            num_active_threads_per_warp_ * num_warps_per_block_;
        assert(num_active_threads_ % num_active_threads_per_block == 0);
        num_blocks_ = num_active_threads_ / num_active_threads_per_block;

        // number or (active+inactive) threads per block
        assert(num_active_threads_ % num_active_threads_per_warp_ == 0);
        auto num_warps = num_active_threads_ / num_active_threads_per_warp_;
        auto num_threads = cuda_warp_size() * num_warps;
        assert(num_threads % num_blocks_ == 0);
        block_size_ = num_threads / num_blocks_;

        // distance between two active threads in a warp
        assert(cuda_warp_size() % num_active_threads_per_warp_ == 0);
        warp_step_ = cuda_warp_size() / num_active_threads_per_warp_;
      }

      size_t num_gpu_threads() const { return num_active_threads_; }

      // configuration parameters
      size_t mask_words_;
      size_t num_active_threads_;
      size_t num_warps_per_block_;  // block density: 1 to num_active_threads_ /
                                    // num_active_threads_per_warp_
      size_t num_active_threads_per_warp_;  // warp density: 1 to CUDA warp size

      // inferred configuration
      size_t block_size_, warp_step_, num_blocks_;
    };

    GPUWorker(const config_t &conf, const GraphTy &G, const PRNGeneratorTy &rng)
        : conf_(conf), G_(G), rng_(rng), u_(0, G_.num_nodes()) {
      // allocate host/device memory
      auto mask_size = conf.mask_words_ * sizeof(mask_word_t);
      res_mask_ = (mask_word_t *)malloc(conf_.num_gpu_threads() * mask_size);
      cuda_malloc((void **)&d_res_mask_, conf_.num_gpu_threads() * mask_size);
      cuda_malloc((void **)&d_trng_state_,
                  conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));
    }

    ~GPUWorker() {
      // free host/device memory
      free(res_mask_);
      cuda_free(d_res_mask_);
      cuda_free(d_trng_state_);
    }

    void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                   size_t first_seq) {
      cuda_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                     conf_.num_blocks_, conf_.block_size_, conf_.warp_step_);
    }

    static void init(const GraphTy &G) { cuda_graph_init(G); }

    static void fini() { cuda_graph_fini(); }

   private:
    config_t conf_;
    const GraphTy &G_;
    PRNGeneratorTy rng_;
    trng::uniform_int_dist u_;

    // memory buffers
    mask_word_t *res_mask_, *d_res_mask_;
    PRNGeneratorTy *d_trng_state_;

    void batch(rrr_set_t *first, size_t size,
               ripples::independent_cascade_tag &&m, size_t rank) {
      batch_dispatcher(first, size,
                       std::forward<ripples::independent_cascade_tag>(m), rank);
    }

    void batch(rrr_set_t *first, size_t size, ripples::linear_threshold_tag &&m,
               size_t rank) {
      batch_dispatcher(first, size,
                       std::forward<ripples::linear_threshold_tag>(m), rank);
    }

    template <typename diff_model_tag>
    void batch_dispatcher(rrr_set_t *first, size_t size, diff_model_tag &&m,
                          size_t rank) {
#if CUDA_PROFILE
      auto start = std::chrono::high_resolution_clock::now();
#endif
      auto max_batch_size = conf_.num_active_threads_;
      auto num_batches = (size + max_batch_size - 1) / max_batch_size;
      auto last = first + size;
      for (size_t bi = 0; bi < num_batches; ++bi, first += max_batch_size) {
        auto batch_offset = bi * max_batch_size;
        auto batch_size = std::min(size - batch_offset, max_batch_size);
        batch_kernel(batch_size, std::forward<diff_model_tag>(m));
        batch_d2h(batch_size);
        batch_build(first, batch_size, std::forward<diff_model_tag>(m));
      }
#if CUDA_PROFILE
      auto &p(this->prof_bd.back());
      p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - start);
      p.n_ += size;
      spdlog::get("console")->info(
          "> [GPUWorker @{} batch_dispatcher] profile size={} n-sets={} ns={}",
          (void *)this, this->prof_bd.size(), p.n_, p.d_.count());
#endif
    }

    template <typename diff_model_tag>
    void batch_kernel(size_t batch_size, diff_model_tag &&) {
      if (std::is_same<diff_model_tag, ripples::linear_threshold_tag>::value) {
        cuda_lt_kernel(conf_.num_blocks_, conf_.block_size_, batch_size,
                       G_.num_nodes(), conf_.warp_step_, d_trng_state_,
                       d_res_mask_, conf_.mask_words_);
      } else if (std::is_same<diff_model_tag,
                              ripples::independent_cascade_tag>::value) {
        cuda_ic_kernel(conf_.num_blocks_, conf_.block_size_, batch_size,
                       G_.num_nodes(), conf_.warp_step_, d_trng_state_,
                       d_res_mask_, conf_.mask_words_);
      } else {
        spdlog::error("invalid diffusion model");
        exit(1);
      }
    }

    void batch_d2h(size_t batch_size) {
      cuda_d2h(res_mask_, d_res_mask_,
               batch_size * conf_.mask_words_ * sizeof(mask_word_t));
    }

    template <typename diff_model_tag>
    void batch_build(rrr_set_t *first, size_t batch_size,
                     diff_model_tag &&model_tag) {
#if CUDA_PROFILE
      auto &p(this->prof_bd.back());
#endif

      for (size_t i = 0; i < batch_size; ++i, ++first) {
        auto &rrr_set(*first);
        rrr_set.reserve(conf_.mask_words_);
        auto res_mask = res_mask_ + (i * conf_.mask_words_);
        if (res_mask[0] != G_.num_nodes()) {
          // valid walk
          for (size_t j = 0;
               j < conf_.mask_words_ && res_mask[j] != G_.num_nodes(); ++j) {
            rrr_set.push_back(res_mask[j]);
          }
        } else {
// invalid walk
#if CUDA_PROFILE
          p.num_exceedings_++;
#endif
          auto root = res_mask[1];
          AddRRRSet(G_, root, rng_, rrr_set,
                    std::forward<diff_model_tag>(model_tag));
        }

        std::stable_sort(rrr_set.begin(), rrr_set.end());

#if CUDA_CHECK
        check_lt(rrr_set, G_, first);
#endif
      }
    }
  };

 public:
  StreamingRRRGenerator(const GraphTy &G, const PRNGeneratorTy &master_rng,
                        size_t num_cpu_workers, size_t num_gpu_workers)
      : num_cpu_workers_(num_cpu_workers), num_gpu_workers_(num_gpu_workers) {
    // init GPU
    GPUWorker::init(G);

    // configuration
    typename GPUWorker::config_t gpu_conf;
    max_batch_size_ = 32 * gpu_conf.num_gpu_threads();

    // RNG sequences
    auto num_rng_sequences =
        num_cpu_workers + num_gpu_workers * (gpu_conf.num_gpu_threads() + 1);
    auto gpu_seq_offset = num_cpu_workers + num_gpu_workers;

    for (size_t i = 0; i < num_cpu_workers_; ++i) {
      auto rng = master_rng;
      rng.split(num_rng_sequences, i);
      workers.push_back(new CPUWorker(G, rng));
    }

    for (size_t i = 0; i < num_gpu_workers_; ++i) {
      auto rng = master_rng;
      rng.split(num_rng_sequences, num_cpu_workers + i);
      auto w = new GPUWorker(gpu_conf, G, rng);
      w->rng_setup(master_rng, num_rng_sequences,
                   gpu_seq_offset + i * gpu_conf.num_gpu_threads());
      workers.push_back(w);
    }
  }

#if CUDA_PROFILE
  template <typename iterator>
  void print_prof_iter(size_t i, iterator first, iterator last) {
    auto console = spdlog::get("console");
    size_t n_idle = 0;
    for (; first != last; ++first) {
      auto &profs((*first)->prof_bd);
      assert(i < profs.size());
      auto &p(profs.at(i));
      if (p.n_)
        console->info(
            "n-sets={}\tn-exc={}\tns={}\tb={}", p.n_, p.num_exceedings_, p.d_.count(),
            (float)p.n_ * 1e03 /
                std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                    .count());
      else
        ++n_idle;
    }
    if (n_idle) console->info("> {} idle workers", n_idle);
  }
#endif

  ~StreamingRRRGenerator() {
#if CUDA_PROFILE
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(prof_bd.d);
    auto console = spdlog::get("console");
    console->info("*** BEGIN Streaming Engine profiling");
    auto first_gpu_worker = workers.begin();
    std::advance(first_gpu_worker, num_cpu_workers_);
    for (size_t i = 0; i < prof_bd.prof_bd.size(); ++i) {
      console->info("+++ BEGIN iter {}", i);
      console->info("--- CPU workers");
      print_prof_iter(i, workers.begin(), first_gpu_worker);
      console->info("--- GPU workers");
      print_prof_iter(i, first_gpu_worker, workers.end());
      auto &p(prof_bd.prof_bd[i]);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(p.d_);
      console->info("--- overall");
      console->info("n. sets               = {}", p.n_);
      console->info("elapsed (ns)          = {}", p.d_.count());
      console->info("throughput (sets/sec) = {}",
                    (float)p.n_ * 1e03 / ms.count());
      console->info("+++ END iter {}", i);
    }
    console->info("n. sets               = {}", prof_bd.n);
    auto n_excs = std::accumulate(
        workers.begin(), workers.end(), 0, [](size_t acc, const Worker *w) {
          return std::accumulate(
              w->prof_bd.begin(), w->prof_bd.end(), acc,
              [](size_t acc, const typename Worker::iter_profile_t &p) {
                return acc + p.num_exceedings_;
              });
        });
    console->info("n. exceedings         = {} (/{}={})", n_excs, prof_bd.n,
                  (float)n_excs / prof_bd.n);
    console->info("n. iters              = {}", prof_bd.prof_bd.size());
    console->info("elapsed (ms)          = {}", ms.count());
    console->info("throughput (sets/sec) = {}",
                  (float)prof_bd.n * 1e06 / ms.count());
    console->info("*** END Streaming Engine profiling");
#endif

    for (auto &w : workers) delete w;
    GPUWorker::fini();
  }

  template <typename diff_model_tag>
  rrr_sets_t generate(size_t theta, diff_model_tag &&m) {
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &w : workers) w->begin_prof_iter();
#endif

    rrr_sets_t res(theta);
    auto sets_ptr_ = res.data();
    auto last = sets_ptr_ + theta;
    auto num_batches = (theta + max_batch_size_ - 1) / max_batch_size_;

#pragma omp parallel num_threads(num_cpu_workers_ + num_gpu_workers_)
    {
      size_t rank = omp_get_thread_num();

#pragma omp for schedule(dynamic)
      for (size_t bi = 0; bi < num_batches; ++bi) {
        auto batch_offset = bi * max_batch_size_;
        auto batch_size = std::min(theta - batch_offset, max_batch_size_);
        workers[rank]->batch(sets_ptr_ + batch_offset, batch_size,
                             std::forward<diff_model_tag>(m), rank);
      }
    }

#if CUDA_PROFILE
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    prof_bd.prof_bd.emplace_back(theta, d);
    prof_bd.n += theta;
    prof_bd.d += std::chrono::duration_cast<std::chrono::microseconds>(d);
#endif

    return res;
  }

 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  size_t max_batch_size_;  // TODO differentiate small-large batches
  std::vector<Worker *> workers;

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
};
}  // namespace ripples

#endif  // RIPPLES_CUDA_STREAMING_RRR_GENERATOR_H
