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
#include "ripples/imm.h"

#include "ripples/cuda/from_nvgraph/bfs.hxx"

namespace ripples {

template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
class StreamingRRRGenerator {
  using vertex_t = typename GraphTy::vertex_type;
  using rrr_set_t = std::vector<vertex_t>;
  using rrr_sets_t = std::vector<rrr_set_t>;

  class Worker {
   public:
    Worker(const GraphTy &G) : G_(G) {}
    virtual ~Worker() {}
    virtual void batch(rrr_set_t *first, size_t size) = 0;

#if CUDA_PROFILE
    struct iter_profile_t {
      size_t n_{0}, num_exceedings_{0};
      std::chrono::nanoseconds d_{0};
    };
    using profile_t = std::vector<iter_profile_t>;
    profile_t prof_bd;

    void begin_prof_iter() { prof_bd.emplace_back(); }
#endif

   protected:
    const GraphTy &G_;
  };

  class CPUWorker : public Worker {
   public:
    CPUWorker(const GraphTy &G, const PRNGeneratorTy &rng)
        : Worker(G), rng_(rng), u_(0, G.num_nodes()) {}

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
  };

  // TODO factorize RNGs
  class GPUWorker : public Worker {
   public:
    GPUWorker(const GraphTy &G) : Worker(G) {}
    virtual ~GPUWorker() {}

    static void init(const GraphTy &G) { cuda_graph_init(G); }
    static void fini() { cuda_graph_fini(); }

    virtual void batch(rrr_set_t *first, size_t size) = 0;
  };

  class GPUWorkerLT : public GPUWorker {
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

    GPUWorkerLT(const config_t &conf, const GraphTy &G,
                const PRNGeneratorTy &rng)
        : GPUWorker(G), conf_(conf), u_(0, G.num_nodes()) {
      // allocate host/device memory
      auto mask_size = conf.mask_words_ * sizeof(mask_word_t);
      lt_res_mask_ = (mask_word_t *)malloc(conf_.num_gpu_threads() * mask_size);
      cuda_malloc((void **)&d_lt_res_mask_,
                  conf_.num_gpu_threads() * mask_size);

      // allocate device-size RNGs
      cuda_malloc((void **)&d_trng_state_,
                  conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));
    }

    ~GPUWorkerLT() {
      // free host/device memory
      free(lt_res_mask_);
      cuda_free(d_lt_res_mask_);
      cuda_free(d_trng_state_);
    }

    void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                   size_t first_seq) {
      cuda_lt_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                        conf_.num_blocks_, conf_.block_size_, conf_.warp_step_);
    }

   private:
    config_t conf_;
    PRNGeneratorTy rng_;
    trng::uniform_int_dist u_;

    // memory buffers
    mask_word_t *lt_res_mask_, *d_lt_res_mask_;
    PRNGeneratorTy *d_trng_state_;

    void batch(rrr_set_t *first, size_t size) {
#if CUDA_PROFILE
      auto start = std::chrono::high_resolution_clock::now();
#endif
      auto max_batch_size = conf_.num_active_threads_;
      auto num_batches = (size + max_batch_size - 1) / max_batch_size;
      auto last = first + size;
      for (size_t bi = 0; bi < num_batches; ++bi, first += max_batch_size) {
        auto batch_offset = bi * max_batch_size;
        auto batch_size = std::min(size - batch_offset, max_batch_size);
        cuda_lt_kernel(conf_.num_blocks_, conf_.block_size_, batch_size,
                       this->G_.num_nodes(), conf_.warp_step_, d_trng_state_,
                       d_lt_res_mask_, conf_.mask_words_);
        cuda_d2h(lt_res_mask_, d_lt_res_mask_,
                 batch_size * conf_.mask_words_ * sizeof(mask_word_t));
        batch_lt_build(first, batch_size);
      }
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
  };  // GPUWorkerLT

  class GPUWorkerIC : public GPUWorker {
   public:
   // TODO drop
    struct config_t {
      config_t(size_t n_edges) {
        block_size_ = nvgraph::Bfs<int>::traverse_block_size();
        num_blocks_ = nvgraph::Bfs<int>::traverse_max_num_blocks(n_edges);
      }

      size_t num_gpu_threads() const {
        return num_blocks_ * block_size_;
      }

      size_t num_blocks_, block_size_;
    };

    GPUWorkerIC(const config_t &conf, const GraphTy &G,
                const PRNGeneratorTy &rng)
        : GPUWorker(G),
          conf_(conf),
          u_(0, G.num_nodes()),
          // TODO stream
          solver(G.num_nodes(), G.num_edges(), cuda_graph_index(),
                 cuda_graph_edges(), true, TRAVERSAL_DEFAULT_ALPHA,
                 TRAVERSAL_DEFAULT_BETA) {
      // allocate host/device memory
      ic_predecessors_ = (int *)malloc(
          G.num_nodes() * sizeof(typename cuda_device_graph::vertex_t));
      cudaMalloc((void **)&d_ic_predecessors_,
                 G.num_nodes() * sizeof(typename cuda_device_graph::vertex_t));

      // allocate device-size RNGs
      cuda_malloc(
          (void **)&d_trng_state_,
          conf_.num_gpu_threads() * sizeof(PRNGeneratorTy));

      solver.configure(nullptr, d_ic_predecessors_, nullptr);
    }

    ~GPUWorkerIC() {
      // free host/device memory
      free(ic_predecessors_);
      cudaFree(d_ic_predecessors_);
      cuda_free(d_trng_state_);
    }

    void rng_setup(const PRNGeneratorTy &master_rng, size_t num_seqs,
                   size_t first_seq) {
      cuda_ic_rng_setup(d_trng_state_, master_rng, num_seqs, first_seq,
                        conf_.num_blocks_, conf_.block_size_,
                        this->G_.num_nodes());
    }

   private:
    config_t conf_;
    PRNGeneratorTy rng_;
    trng::uniform_int_dist u_;

    // nvgraph machinery
    nvgraph::Bfs<int> solver;

    // memory buffers
    typename cuda_device_graph::vertex_t *ic_predecessors_, *d_ic_predecessors_;
    PRNGeneratorTy *d_trng_state_;

    void batch(rrr_set_t *first, size_t size) {
#if CUDA_PROFILE
      auto start = std::chrono::high_resolution_clock::now();
#endif
      for (size_t wi = 0; wi < size; ++wi) {
        solver.traverse((int)u_(rng_));
        cuda_d2h(ic_predecessors_, d_ic_predecessors_,
                 this->G_.num_nodes() *
                     sizeof(typename cuda_device_graph::vertex_t));
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
  };  // GPUWorkerIC

 public:
  StreamingRRRGenerator(const GraphTy &G, const PRNGeneratorTy &master_rng,
                        size_t num_cpu_workers, size_t num_gpu_workers)
      : num_cpu_workers_(num_cpu_workers), num_gpu_workers_(num_gpu_workers) {
    // init GPU
    GPUWorker::init(G);

    // TODO factorize
    if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
      typename GPUWorkerIC::config_t gpu_conf(G.num_edges());
      max_batch_size_ = 32;
      auto num_gpu_threads = gpu_conf.num_gpu_threads();

      auto num_rng_sequences =
          num_cpu_workers + num_gpu_workers * (num_gpu_threads + 1);
      auto gpu_seq_offset = num_cpu_workers + num_gpu_workers;

      // CPU workers
      for (size_t i = 0; i < num_cpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, i);
        workers.push_back(new CPUWorker(G, rng));
      }

      for (size_t i = 0; i < num_gpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers + i);
        auto w = new GPUWorkerIC(gpu_conf, G, rng);
        w->rng_setup(master_rng, num_rng_sequences,
                     gpu_seq_offset + i * num_gpu_threads);
        workers.push_back(w);
      }
    } else if (std::is_same<diff_model_tag,
                            ripples::linear_threshold_tag>::value) {
      typename GPUWorkerLT::config_t gpu_conf;
      auto num_gpu_threads = gpu_conf.num_gpu_threads();
      max_batch_size_ = 32 * num_gpu_threads;

      auto num_rng_sequences =
          num_cpu_workers + num_gpu_workers * (num_gpu_threads + 1);
      auto gpu_seq_offset = num_cpu_workers + num_gpu_workers;

      // CPU workers
      for (size_t i = 0; i < num_cpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, i);
        workers.push_back(new CPUWorker(G, rng));
      }

      for (size_t i = 0; i < num_gpu_workers_; ++i) {
        auto rng = master_rng;
        rng.split(num_rng_sequences, num_cpu_workers + i);
        auto w = new GPUWorkerLT(gpu_conf, G, rng);
        w->rng_setup(master_rng, num_rng_sequences,
                     gpu_seq_offset + i * num_gpu_threads);
        workers.push_back(w);
      }
    } else
      assert(false);
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
            "n-sets={}\tn-exc={}\tns={}\tb={}", p.n_, p.num_exceedings_,
            p.d_.count(),
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
