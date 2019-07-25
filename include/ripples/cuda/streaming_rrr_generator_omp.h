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

#include "ripples/imm.h"
#include "spdlog/spdlog.h"
#include "trng/uniform_int_dist.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/generate_rrr_sets.h"

#include "ripples/cuda/from_nvgraph/bfs.hxx"

namespace ripples {

template <typename GraphTy>
class Worker {
  using vertex_t = typename GraphTy::vertex_type;
  using rrr_set_t = std::vector<vertex_t>;

 public:
  Worker(const GraphTy &G) : G_(G) {}
  virtual ~Worker() {}
  virtual void batch(rrr_set_t *first, size_t size) = 0;

 protected:
  const GraphTy &G_;

#if CUDA_PROFILE
 public:
  virtual void begin_prof_iter() = 0;
#endif
};

template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
class CPUWorker : public Worker<GraphTy> {
  using vertex_t = typename GraphTy::vertex_type;
  using rrr_set_t = std::vector<vertex_t>;

 public:
  CPUWorker(const GraphTy &G, const PRNGeneratorTy &rng)
      : Worker<GraphTy>(G), rng_(rng), u_(0, G.num_nodes()) {}

 private:
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;

  void batch(rrr_set_t *first, size_t size) {
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    for (auto last = first + size; first != last; ++first) {
      vertex_t root = u_(rng_);
      AddRRRSet(this->G_, root, rng_, *first, diff_model_tag{});
    }
#if CUDA_PROFILE
    auto &p(this->prof_bd.back());
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }

#if CUDA_PROFILE
public:
  struct iter_profile_t {
    size_t n_{0}, num_exceedings_{0};
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
         "n-sets={}\tn-exc={}\tns={}\tb={}", p.n_, p.num_exceedings_,
         p.d_.count(),
         (float)p.n_ * 1e03 /
             std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                 .count());
   else
     console->info("> idle worker");
 }
#endif
};

template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
class GPUWorker;

template <typename GraphTy, typename PRNGeneratorTy>
class GPUWorker<GraphTy, PRNGeneratorTy, linear_threshold_tag>
    : public Worker<GraphTy> {
  using vertex_t = typename GraphTy::vertex_type;
  using rrr_set_t = std::vector<vertex_t>;

 public:
  struct config_t {
    config_t(size_t max_batch_size) : num_threads_(max_batch_size) {
      assert(num_threads_ % block_size_ == 0);
      max_blocks_ = num_threads_ / block_size_;

      printf(
          "*** DBG *** > [GPUWorkerLT::config_t] "
          "block_size_=%d\tnum_threads_=%d\tmax_blocks_=%d\n",
          block_size_, num_threads_, max_blocks_);
    }

    size_t num_gpu_threads() const { return num_threads_; }

    // configuration parameters
    static constexpr size_t block_size_ = 256;
    const size_t num_threads_;
    const size_t mask_words_ = 8;  // maximum walk size

    // inferred configuration
    size_t max_blocks_{0};
  };

  GPUWorker(const config_t &conf, const GraphTy &G, const PRNGeneratorTy &rng,
            cudaStream_t cuda_stream)
      : Worker<GraphTy>(G), cuda_stream_(cuda_stream),
        conf_(conf),
        u_(0, G.num_nodes()) {
    // allocate host/device memory
    auto mask_size = conf.mask_words_ * sizeof(mask_word_t);
    lt_res_mask_ = (mask_word_t *)malloc(conf_.num_gpu_threads() * mask_size);
    cuda_malloc((void **)&d_lt_res_mask_, conf_.num_gpu_threads() * mask_size);

    // allocate device-size RNGs
    cuda_malloc((void **)&d_trng_state_,
                conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));
  }

  ~GPUWorker() {
    // free host/device memory
    free(lt_res_mask_);
    cuda_free(d_lt_res_mask_);
    cuda_free(d_trng_state_);
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {
    cuda_lt_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                      conf_.max_blocks_, conf_.block_size_);
  }

 private:
  config_t conf_;
  cudaStream_t cuda_stream_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;

  // memory buffers
  mask_word_t *lt_res_mask_, *d_lt_res_mask_;
  PRNGeneratorTy *d_trng_state_;

  void batch(rrr_set_t *first, size_t size) {
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif

    cuda_lt_kernel(conf_.max_blocks_, conf_.block_size_, size,
                   this->G_.num_nodes(), d_trng_state_, d_lt_res_mask_,
                   conf_.mask_words_, this->cuda_stream_);
    cuda_d2h(lt_res_mask_, d_lt_res_mask_,
             size * conf_.mask_words_ * sizeof(mask_word_t),
             this->cuda_stream_);
    cuda_sync(this->cuda_stream_);
    batch_lt_build(first, size);

#if CUDA_PROFILE
    auto &p(this->prof_bd.back());
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }

  void batch_lt_build(rrr_set_t *first, size_t batch_size) {
#if CUDA_PROFILE
    auto &p(this->prof_bd.back());
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
public:
  struct iter_profile_t {
    size_t n_{0}, num_exceedings_{0};
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
          "n-sets={}\tn-exc={}\tns={}\tb={}", p.n_, p.num_exceedings_,
          p.d_.count(),
          (float)p.n_ * 1e03 /
              std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                  .count());
    else
      console->info("> idle worker");
  }
#endif
};

template <typename GraphTy, typename PRNGeneratorTy>
class GPUWorker<GraphTy, PRNGeneratorTy, independent_cascade_tag>
    : public Worker<GraphTy> {
  using vertex_t = typename GraphTy::vertex_type;
  using rrr_set_t = std::vector<vertex_t>;
  using bfs_solver_t = nvgraph::Bfs<int, PRNGeneratorTy>;

 public:
  struct config_t {
    config_t(size_t num_workers)
        : block_size_(bfs_solver_t::traverse_block_size()),
          max_blocks_(num_workers ? cuda_max_blocks() / num_workers : 0) {
      printf(
          "*** DBG *** > [GPUWorkerIC::config_t] "
          "max_blocks_=%d\tblock_size_=%d\n",
          max_blocks_, block_size_);
    }

    size_t num_gpu_threads() const { return max_blocks_ * block_size_; }

    const size_t max_blocks_;
    const size_t block_size_;
  };

  GPUWorker(const config_t &conf, const GraphTy &G, const PRNGeneratorTy &rng,
            cudaStream_t cuda_stream)
      : Worker<GraphTy>(G), cuda_stream_(cuda_stream),
        conf_(conf),
        u_(0, G.num_nodes()),
        // TODO stream
        solver(G.num_nodes(), G.num_edges(), cuda_graph_index(),
               cuda_graph_edges(), cuda_graph_weights(), true,
               TRAVERSAL_DEFAULT_ALPHA, TRAVERSAL_DEFAULT_BETA,
               conf_.max_blocks_, cuda_stream) {
    // allocate host/device memory
    ic_predecessors_ = (int *)malloc(
        G.num_nodes() * sizeof(typename cuda_device_graph::vertex_t));
    cudaMalloc((void **)&d_ic_predecessors_,
               G.num_nodes() * sizeof(typename cuda_device_graph::vertex_t));

    // allocate device-size RNGs
    cuda_malloc((void **)&d_trng_state_,
                conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));

    solver.configure(nullptr, d_ic_predecessors_, nullptr);
  }

  ~GPUWorker() {
    // free host/device memory
    free(ic_predecessors_);
    cudaFree(d_ic_predecessors_);
    cuda_free(d_trng_state_);
  }

  void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                 size_t first_seq) {
    cuda_ic_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                      conf_.max_blocks_, conf_.block_size_);
    solver.rng(d_trng_state_);
  }

 private:
  config_t conf_;
  cudaStream_t cuda_stream_;
  PRNGeneratorTy rng_;
  trng::uniform_int_dist u_;

  // nvgraph machinery
  bfs_solver_t solver;

  // memory buffers
  typename cuda_device_graph::vertex_t *ic_predecessors_, *d_ic_predecessors_;
  PRNGeneratorTy *d_trng_state_;

  void batch(rrr_set_t *first, size_t size) {
#if CUDA_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    for (size_t wi = 0; wi < size; ++wi) {
      solver.traverse((int)u_(rng_));
      cuda_d2h(
          ic_predecessors_, d_ic_predecessors_,
          this->G_.num_nodes() * sizeof(typename cuda_device_graph::vertex_t),
          this->cuda_stream_);
      cuda_sync(this->cuda_stream_);
      ic_build(first++);
    }
#if CUDA_PROFILE
    auto &p(this->prof_bd.back());
    p.d_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start);
    p.n_ += size;
#endif
  }

  void ic_build(rrr_set_t *dst) {
    auto &rrr_set(*dst);
    for (vertex_t i = 0; i < this->G_.num_nodes(); ++i)
      if (ic_predecessors_[i] != -1) rrr_set.push_back(i);
  }

#if CUDA_PROFILE
 public:
  struct iter_profile_t {
    size_t n_{0}, num_exceedings_{0};
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
          "n-sets={}\tn-exc={}\tns={}\tb={}", p.n_, p.num_exceedings_,
          p.d_.count(),
          (float)p.n_ * 1e03 /
              std::chrono::duration_cast<std::chrono::milliseconds>(p.d_)
                  .count());
    else
      console->info("> idle worker");
  }
#endif
};

template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
class StreamingRRRGenerator {
  using vertex_t = typename GraphTy::vertex_type;
  using rrr_set_t = std::vector<vertex_t>;
  using rrr_sets_t = std::vector<rrr_set_t>;

  using gpu_worker_t = GPUWorker<GraphTy, PRNGeneratorTy, diff_model_tag>;
  using cpu_worker_t = CPUWorker<GraphTy, PRNGeneratorTy, diff_model_tag>;

 public:
  StreamingRRRGenerator(const GraphTy &G, const PRNGeneratorTy &master_rng,
                        size_t num_cpu_workers, size_t num_gpu_workers)
      : num_cpu_workers_(num_cpu_workers), num_gpu_workers_(num_gpu_workers) {
    // init GPU
    cuda_graph_init(G);

    // TODO factorize
    std::vector<cudaStream_t> cuda_streams;
    if (std::is_same<diff_model_tag, independent_cascade_tag>::value) {
      max_batch_size_ = 32;
      typename gpu_worker_t::config_t gpu_conf(num_gpu_workers_);
      assert(gpu_conf.max_blocks_ * num_gpu_workers <= cuda_max_blocks());
      auto num_gpu_threads_per_worker = gpu_conf.num_gpu_threads();

      auto num_rng_sequences =
          num_cpu_workers_ +
          num_gpu_workers_ * (num_gpu_threads_per_worker + 1);
      auto gpu_seq_offset = num_cpu_workers_ + num_gpu_workers_;

      // CPU workers
      for (size_t i = 0; i < num_cpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, i);
        workers.push_back(new cpu_worker_t(G, rng));
      }

      // GPU workers
      for (size_t i = 0; i < num_gpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers_ + i);
        cuda_streams.emplace_back();
        auto &stream(cuda_streams.back());
        cudaStreamCreate(&stream);
        auto w = new gpu_worker_t(gpu_conf, G, rng, stream);
        w->rng_setup(master_rng, num_rng_sequences,
                     gpu_seq_offset + i * num_gpu_threads_per_worker);
        workers.push_back(w);
      }
    } else if (std::is_same<diff_model_tag,
                            ripples::linear_threshold_tag>::value) {
      max_batch_size_ = 1 << 15;
      typename gpu_worker_t::config_t gpu_conf(max_batch_size_);
      assert(gpu_conf.max_blocks_ * num_gpu_workers <= cuda_max_blocks());
      auto num_gpu_threads_per_worker = gpu_conf.num_gpu_threads();

      auto num_rng_sequences =
          num_cpu_workers_ +
          num_gpu_workers_ * (num_gpu_threads_per_worker + 1);
      auto gpu_seq_offset = num_cpu_workers_ + num_gpu_workers_;

      // CPU workers
      for (size_t i = 0; i < num_cpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, i);
        workers.push_back(new cpu_worker_t(G, rng));
      }

      for (size_t i = 0; i < num_gpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers + i);
        cuda_streams.emplace_back();
        auto &stream(cuda_streams.back());
        cudaStreamCreate(&stream);
        auto w = new gpu_worker_t(gpu_conf, G, rng, stream);
        w->rng_setup(master_rng, num_rng_sequences,
                     gpu_seq_offset + i * num_gpu_threads_per_worker);
        workers.push_back(w);
      }
    } else
      assert(false);
  }

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
      for (auto it = workers.begin(); it != first_gpu_worker; ++it)
        (dynamic_cast<cpu_worker_t *>(*it))->print_prof_iter(i);
      console->info("--- GPU workers");
      for (auto it = first_gpu_worker; it != workers.end(); ++it)
        (dynamic_cast<gpu_worker_t *>(*it))->print_prof_iter(i);
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
    auto n_excs = std::accumulate(
        first_gpu_worker, workers.end(), 0,
        [](size_t acc, Worker<GraphTy> *w) {
          return std::accumulate(
              (dynamic_cast<gpu_worker_t *>(w))->prof_bd.begin(),
              (dynamic_cast<gpu_worker_t *>(w))->prof_bd.end(), acc,
              [](size_t acc,
                 const typename gpu_worker_t::iter_profile_t &p) {
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
    cuda_graph_fini();
  }

  rrr_sets_t generate(size_t theta) {
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
        workers[rank]->batch(sets_ptr_ + batch_offset, batch_size);
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
  size_t max_batch_size_;
  std::vector<Worker<GraphTy> *> workers;

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
