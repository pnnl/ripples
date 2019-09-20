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

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "omp.h"

#include "ripples/generate_rrr_sets.h"

#if RIPPLES_ENABLE_CUDA
#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/find_most_influential.h"
#endif


namespace ripples {

template <typename GraphTy>
class FindMostInfluentialWorker {
 public:
  using vertex_type = typename GraphTy::vertex_type;

  virtual ~FindMostInfluentialWorker() {}

  virtual void InitialCount() = 0;

  virtual void UpdateCounters(vertex_type last_seed) = 0;

  virtual void ReduceCounters() = 0;
};

#if RIPPLES_ENABLE_CUDA
template <typename GraphTy>
class GPUFindMostInfluentialWorker : public FindMostInfluentialWorker<GraphTy>
{
 public:
  using vertex_type = typename GraphTy::vertex_type;

  GPUFindMostInfluentialWorker(size_t device_number, uint32_t * d_counters,
                               uint32_t * d_rr_vertices, uint32_t * d_rr_edges,
                               size_t num_nodes,
                               std::vector<uint32_t *> & device_counters,
                               std::vector<uint32_t> & global_counters,
                               std::vector<std::pair<size_t, size_t>> & reduction_tree)
      : device_number_(device_number)
      , d_counters_(d_counters)
      , d_rr_vertices_(d_rr_vertices)
      , d_rr_edges_(d_rr_edges)
      , num_nodes_(num_nodes)
      , global_counters_(global_counters)
  {
    cuda_set_device(device_number);
    cuda_stream_create(&stream_);
    reduction_target_ = reduction_tree[device_number].first;
    cuda_enable_p2p(reduction_target_);
    reduction_step_ = reduction_tree[device_number].second;
  }

  ~GPUFindMostInfluentialWorker() {
    cuda_disable_p2p(reduction_target_);
    cuda_stream_destroy(stream_);
  }

  void InitialCount() {
    CudaCountOccurrencies(d_counters_, d_rr_edges_,
                          d_rr_set_size_, num_nodes_, stream_);
  }
  void UpdateCounters(vertex_type last_seed) {}
  void ReduceCounters(size_t step) {
    if (step != reduction_step_) return;

    // Accumulate in target array.
    CudaReduceCounters(stream_, d_counters_, d_counters_dest, num_nodes_);
  }

 private:
  cudaStream_t stream_;
  size_t device_number_;
  size_t reduction_step_;
  size_t reduction_target_;
  uint32_t * d_counters_;
  uint32_t * d_counters_dest;
  uint32_t * d_rr_vertices_;
  uint32_t * d_rr_edges_;
  size_t d_rr_set_size_;

  char * d_mask;
  size_t num_nodes_;
  std::vector<uint32_t> & global_counters_;
};

#endif

template <typename GraphTy>
class CPUFindMostInfluentialWorker : public FindMostInfluentialWorker<GraphTy> {
  using vertex_type = typename GraphTy::vertex_type;
  using rrr_set_iterator = typename RRRsets<GraphTy>::iterator;
 public:
  CPUFindMostInfluentialWorker(
      std::vector<vertex_type> & global_count,
      std::vector<std::pair<vertex_type, size_t>> & queue_storage,
      rrr_set_iterator begin, rrr_set_iterator end, size_t num_threads)
      : global_count_(global_count)
      , queue_storage_(queue_storage)
      , begin_(begin)
      , end_(end)
      , num_threads_(num_threads)
  {}

  void InitialCount() {
    CountOccurrencies(begin_, end_, global_count_.begin(),
                      global_count_.end(), num_threads_);

    InitHeapStorage(global_count_.begin(), global_count_.end(),
                    queue_storage_.begin(), queue_storage_.end(),
                    num_threads_);
  }

  void UpdateCounters(vertex_type last_seed) {
    auto cmp = [=](const RRRset<GraphTy> &a) -> auto {
      return !std::binary_search(a.begin(), a.end(), last_seed);
    };

    auto itr = partition(begin_, end_, cmp, num_threads_);

    if (std::distance(itr, end_) < std::distance(begin_, itr)) {
      ripples::UpdateCounters(itr, end_, global_count_, num_threads_);
    } else {
#pragma omp parallel for simd num_threads(num_threads_)
      for (size_t i = 0; i < global_count_.size(); ++i)
        global_count_[i] = 0;
      CountOccurrencies(begin_, itr, global_count_.begin(),
                        global_count_.end(), num_threads_);
    }
    end_ = itr;
  }

  void ReduceCounters() {}

 private:
  std::vector<vertex_type> & global_count_;
  std::vector<std::pair<vertex_type, size_t>> & queue_storage_;
  rrr_set_iterator begin_;
  rrr_set_iterator end_;
  size_t num_threads_;
};


template <typename GraphTy>
struct CompareHeap {
  using vertex_type = typename GraphTy::vertex_type;

  bool operator()(std::pair<vertex_type, size_t> &a,
                  std::pair<vertex_type, size_t> &b) {
    return a.second < b.second;
  }
};


template <typename GraphTy>
class StreamingFindMostInfluential {
  using vertex_type = typename GraphTy::vertex_type;
  using worker_type = FindMostInfluentialWorker<GraphTy>;
  using cpu_worker_type = CPUFindMostInfluentialWorker<GraphTy>;
#if RIPPLES_ENABLE_CUDA
  using gpu_worker_type = GPUFindMostInfluentialWorker<GraphTy>;
#endif

  CompareHeap<GraphTy> cmpHeap;
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, size_t>,
                          std::vector<std::pair<vertex_type, size_t>>,
                          decltype(cmpHeap)>;

 public:
  StreamingFindMostInfluential(const GraphTy &G, RRRsets<GraphTy> &RRRsets)
      : num_cpu_workers_(0)
      , num_gpu_workers_(0)
      , workers_()
      , vertex_coverage_(G.num_nodes())
      , queue_storage_(G.num_nodes())
      , RRRsets_(RRRsets)
  {
    #pragma omp single
    { num_cpu_workers_ = omp_get_max_threads(); }
#if RIPPLES_ENABLE_CUDA
    // Get Number of device and allocate 1 thread each.
    num_gpu_workers_ = cuda_num_devices();
    num_cpu_workers_ -= num_gpu_workers_;

    // Allocate Counters
    d_counters_.reserve(num_gpu_workers_);
    for (size_t i = 0; i < num_gpu_workers_; ++i) {
      uint32_t * ptr;
      cuda_malloc(reinterpret_cast<void**>(&ptr), sizeof(uint32_t) * G.num_nodes());
      d_counters_.push_back(ptr);
    }

    // Data loading into device

    // Merge Procedure

    // Define Reduction tree on GPU workers.
    auto tree = cuda_get_reduction_tree();

    // Construct GPU workers
#endif
    workers_.push_back(
        new CPUFindMostInfluentialWorker<GraphTy>(vertex_coverage_,
                                                  queue_storage_,
                                                  RRRsets_.begin(),
                                                  RRRsets_.end(),
                                                  num_cpu_workers_));
  }

  ~StreamingFindMostInfluential() {
#if RIPPLES_ENABLE_CUDA
    for (auto b : d_counters_) {
      cuda_free(b);
    }
#endif
    for (auto w : workers_) {
      delete w;
    }
  }

  void InitialCount() {
    #pragma omp parallel num_threads(num_gpu_workers_ + 1)
    {
      size_t rank = omp_get_thread_num();
      if (rank < 1)
        workers_[rank]->InitialCount();
    }
  }

  void UpdateCounters(vertex_type last_seed) {
    #pragma omp parallel num_threads(num_gpu_workers_ + 1)
    {
      size_t rank = omp_get_thread_num();
      if (rank < 1)
        workers_[rank]->UpdateCounters(last_seed);
    }
  }

  priorityQueue getHeap() {
    priorityQueue queue(cmpHeap, std::move(queue_storage_));
    return queue;
  }

  std::pair<vertex_type, size_t> getNextSeed(priorityQueue & queue_) {
    while (!queue_.empty()) {
      auto element = queue_.top();
      queue_.pop();

      if (element.second > vertex_coverage_[element.first]) {
        element.second = vertex_coverage_[element.first];
        queue_.push(element);
        continue;
      }
      return element;
    }
    throw std::logic_error("Reached a mighty Unreachable State");
  }

  auto find_most_influential_set(
      size_t k) {
    omp_set_nested(true);
    InitialCount();

    auto queue = getHeap();

    std::vector<vertex_type> result;
    result.reserve(k);
    size_t uncovered = RRRsets_.size();

    while (uncovered != 0) {
      auto element = getNextSeed(queue);

      uncovered -= element.second;
      result.push_back(element.first);

      UpdateCounters(element.first);

      if (result.size() == k) break;
    }

    double f = double(RRRsets_.size() - uncovered) / RRRsets_.size();
    return std::make_pair(f, result);
  }
  
 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  RRRsets<GraphTy> &RRRsets_;
  std::vector<worker_type *> workers_;
  std::vector<uint32_t *> d_counters_;
  std::vector<uint32_t> vertex_coverage_;
  std::vector<std::pair<vertex_type, size_t>> queue_storage_;
};

}  // namespace ripples
