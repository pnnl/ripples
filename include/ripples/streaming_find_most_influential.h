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

#ifndef RIPPLES_STREAMING_FIND_MOST_INFLUENTIAL_H
#define RIPPLES_STREAMING_FIND_MOST_INFLUENTIAL_H

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "omp.h"

#include "ripples/generate_rrr_sets.h"
#include "ripples/partition.h"

#ifdef RIPPLES_ENABLE_CUDA
#include "ripples/cuda/cuda_utils.h"
#include "ripples/cuda/find_most_influential.h"
#endif

namespace ripples {

template <typename GraphTy>
class FindMostInfluentialWorker {
 public:
  using rrr_set_iterator = typename RRRsets<GraphTy>::iterator;
  using vertex_type = typename GraphTy::vertex_type;

  virtual ~FindMostInfluentialWorker() {}

  virtual PartitionIndices<rrr_set_iterator> LoadData(rrr_set_iterator B,
                                                      rrr_set_iterator E) = 0;

  virtual void InitialCount() = 0;

  virtual void UpdateCounters(vertex_type last_seed) = 0;

  virtual void ReduceCounters(size_t step) = 0;

  virtual void set_first_rrr_set(rrr_set_iterator I) = 0;

  virtual bool has_work() = 0;
};

#ifdef RIPPLES_ENABLE_CUDA
template <typename GraphTy>
class GPUFindMostInfluentialWorker : public FindMostInfluentialWorker<GraphTy> {
 public:
  using rrr_set_iterator =
      typename FindMostInfluentialWorker<GraphTy>::rrr_set_iterator;
  using vertex_type = typename GraphTy::vertex_type;

  GPUFindMostInfluentialWorker(size_t device_number, size_t num_nodes,
                               std::vector<uint32_t *> &device_counters,
                               size_t reduction_target, size_t reduction_step,
                               uint32_t *d_counters_dest)
      : device_number_(device_number),
        d_counters_(device_counters[device_number]),
        d_rr_vertices_(nullptr),
        d_rr_edges_(nullptr),
        d_mask_(nullptr),
        d_rr_set_size_(0),
        num_nodes_(num_nodes),
        reduction_target_(reduction_target),
        reduction_step_(reduction_step),
        d_counters_dest_(d_counters_dest) {
    cuda_set_device(device_number);
    cuda_stream_create(&stream_);
    if (reduction_target_ != device_number) {
      cuda_enable_p2p(reduction_target_);
    }
  }

  virtual ~GPUFindMostInfluentialWorker() {
    cuda_set_device(device_number_);

    if (reduction_target_ != device_number_) {
      cuda_disable_p2p(reduction_target_);
    }
    cuda_stream_destroy(stream_);

    cuda_free(d_pool_);
    // cuda_free(d_rr_vertices_);
    // cuda_free(d_rr_edges_);
    // cuda_free(d_mask_);
  }

  void set_first_rrr_set(rrr_set_iterator I) {}

  bool has_work() { return d_rr_set_size_ != 0; }

  PartitionIndices<rrr_set_iterator> LoadData(rrr_set_iterator B,
                                              rrr_set_iterator E) {
    cuda_set_device(device_number_);
    // Ask runtime available memory.  The best thing we can do is guessing.
    // Memory fragmentation might get in the way, so we ask the runtime
    // for what is free and then ask for half of that.
    size_t avail_space = cuda_available_memory() >> 1;
    bool allocSuccess = cuda_malloc(reinterpret_cast<void **>(&d_pool_), avail_space);
    assert(allocSuccess &&
           "Not enough memory on the GPUs. Our heuristic for acquiring memory"
           "to perferm seed-selection failed.  Please, re-run the application"
           "using --seed-select-max-gpu-workers 0.");
    cuda_memset(reinterpret_cast<void *>(d_pool_), 0, avail_space);

    size_t space = 0;

    auto pivot = B;
    size_t num_elements = 0;
    for (; pivot < E && space < avail_space; ++pivot) {
      // Two uint32_t per the RRR sets + 1 byte for the mask.
      num_elements += pivot->size();
      space += pivot->size() * sizeof(uint32_t) + sizeof(uint32_t);
    }

    // cuda_malloc(reinterpret_cast<void **>(&d_mask_), std::distance(B, pivot));
    d_mask_ = d_pool_;
    // cuda_memset(reinterpret_cast<void *>(d_mask_), 0, std::distance(B, pivot));
    // cuda_check(__FILE__, __LINE__);
    space -= sizeof(uint32_t) * std::distance(B, pivot);

    size_t BufferSize = 1 << 24;

    // cuda_malloc(reinterpret_cast<void **>(&d_rr_edges_), space >> 1);
    d_rr_edges_ = d_mask_ + std::distance(B, pivot);
    d_rr_vertices_ = d_rr_edges_ + num_elements;
    // cuda_malloc(reinterpret_cast<void **>(&d_rr_vertices_), space >> 1);

    std::vector<uint32_t> rr_edges_buffer_to_load;
    std::vector<uint32_t> rr_edges_buffer_to_send;
    rr_edges_buffer_to_load.reserve(BufferSize);
    rr_edges_buffer_to_send.reserve(BufferSize);
    std::vector<uint32_t> rr_vertices_buffer_to_load;
    std::vector<uint32_t> rr_vertices_buffer_to_send;
    rr_vertices_buffer_to_load.reserve(BufferSize);
    rr_vertices_buffer_to_send.reserve(BufferSize);

    uint32_t id = 0;
    auto to_copy = B;
    size_t elements_to_copy = num_elements;

    uint32_t *d_rrr_index = d_rr_vertices_;
    uint32_t *d_rrr_sets = d_rr_edges_;
    for (; to_copy < pivot; ++to_copy, ++id) {
      if (rr_edges_buffer_to_send.size() > BufferSize) break;

      rr_edges_buffer_to_send.insert(rr_edges_buffer_to_send.end(),
                                     to_copy->begin(), to_copy->end());
      rr_vertices_buffer_to_send.insert(rr_vertices_buffer_to_send.end(),
                                        to_copy->size(), id);
      elements_to_copy -= to_copy->size();
      d_rr_set_size_ += to_copy->size();
    }

    while (elements_to_copy > 0) {
      cuda_h2d(reinterpret_cast<void *>(d_rrr_sets),
               reinterpret_cast<void *>(rr_edges_buffer_to_send.data()),
               sizeof(uint32_t) * rr_edges_buffer_to_send.size(), stream_);

      cuda_h2d(reinterpret_cast<void *>(d_rrr_index),
               reinterpret_cast<void *>(rr_vertices_buffer_to_send.data()),
               sizeof(uint32_t) * rr_vertices_buffer_to_send.size(), stream_);
      for (; to_copy < pivot; ++to_copy, ++id) {
        if (rr_edges_buffer_to_load.size() > BufferSize) break;

        rr_edges_buffer_to_load.insert(rr_edges_buffer_to_load.end(),
                                       to_copy->begin(), to_copy->end());
        rr_vertices_buffer_to_load.insert(rr_vertices_buffer_to_load.end(),
                                          to_copy->size(), id);
        elements_to_copy -= to_copy->size();
        d_rr_set_size_ += to_copy->size();
      }

      cuda_sync(stream_);

      d_rrr_index += rr_vertices_buffer_to_send.size();
      d_rrr_sets += rr_edges_buffer_to_send.size();

      rr_vertices_buffer_to_send.swap(rr_vertices_buffer_to_load);
      rr_edges_buffer_to_send.swap(rr_edges_buffer_to_load);

      rr_vertices_buffer_to_load.clear();
      rr_edges_buffer_to_load.clear();
    }

    if (rr_vertices_buffer_to_send.size() > 0) {
      cuda_h2d(reinterpret_cast<void *>(d_rrr_index),
               reinterpret_cast<void *>(rr_vertices_buffer_to_send.data()),
               sizeof(uint32_t) * rr_vertices_buffer_to_send.size(), stream_);

      cuda_h2d(reinterpret_cast<void *>(d_rrr_sets),
               reinterpret_cast<void *>(rr_edges_buffer_to_send.data()),
               sizeof(uint32_t) * rr_edges_buffer_to_send.size(), stream_);
      cuda_sync(stream_);
    }
    return PartitionIndices<rrr_set_iterator>(B, E, pivot);
  }

  void InitialCount() {
    cuda_set_device(device_number_);

    cuda_memset(d_counters_, 0, num_nodes_ * sizeof(uint32_t), stream_);

    CudaCountOccurrencies(d_counters_, d_rr_edges_, d_rr_set_size_, num_nodes_,
                          stream_);

    cuda_sync(stream_);
  }

  void UpdateCounters(vertex_type last_seed) {
    cuda_set_device(device_number_);

    CudaUpdateCounters(stream_, d_rr_set_size_, d_rr_vertices_, d_rr_edges_,
                       d_mask_, d_counters_, num_nodes_, last_seed);
    cuda_sync(stream_);
  }

  void ReduceCounters(size_t step) {
    if (step != reduction_step_) return;

    cuda_set_device(device_number_);

    // Accumulate in target array.
    CudaReduceCounters(stream_, d_counters_, d_counters_dest_, num_nodes_);
  }

 private:
  cudaStream_t stream_;
  size_t device_number_;
  size_t reduction_step_;
  size_t reduction_target_;
  uint32_t *d_counters_;
  uint32_t *d_counters_dest_;
  uint32_t *d_rr_vertices_;
  uint32_t *d_rr_edges_;
  uint32_t *d_pool_;
  size_t d_rr_set_size_;

  uint32_t *d_mask_;
  size_t num_nodes_;
};

#endif

template <typename GraphTy>
class CPUFindMostInfluentialWorker : public FindMostInfluentialWorker<GraphTy> {
  using vertex_type = typename GraphTy::vertex_type;
  using rrr_set_iterator =
      typename FindMostInfluentialWorker<GraphTy>::rrr_set_iterator;

 public:
  CPUFindMostInfluentialWorker(
      std::vector<vertex_type> &global_count,
      std::vector<std::pair<vertex_type, size_t>> &queue_storage,
      rrr_set_iterator begin, rrr_set_iterator end, size_t num_threads,
      uint32_t *d_cpu_counters)
      : global_count_(global_count),
        queue_storage_(queue_storage),
        begin_(begin),
        end_(end),
        num_threads_(num_threads),
        d_cpu_counters_(d_cpu_counters) {}

  virtual ~CPUFindMostInfluentialWorker() {}

  PartitionIndices<rrr_set_iterator> LoadData(rrr_set_iterator B,
                                              rrr_set_iterator E) {
    return PartitionIndices<rrr_set_iterator>(end_, end_, end_);
  }

  bool has_work() { return begin_ != end_; }

  void set_first_rrr_set(rrr_set_iterator I) { begin_ = I; }

  void InitialCount() {
    CountOccurrencies(begin_, end_, global_count_.begin(), global_count_.end(),
                      num_threads_);

    // We have GPU workers so we won't use the heap.
    if (d_cpu_counters_ != nullptr) return;


    InitHeapStorage(global_count_.begin(), global_count_.end(),
                    queue_storage_.begin(), queue_storage_.end(), num_threads_);
  }

  void UpdateCounters(vertex_type last_seed) {
    if (!has_work()) return;

    auto cmp = [=](const RRRset<GraphTy> &a) -> auto {
      return !std::binary_search(a.begin(), a.end(), last_seed);
    };

    auto itr = partition(begin_, end_, cmp, num_threads_);

    if (std::distance(itr, end_) < std::distance(begin_, itr)) {
      ripples::UpdateCounters(itr, end_, global_count_, num_threads_);
    } else {
#pragma omp parallel for simd num_threads(num_threads_)
      for (size_t i = 0; i < global_count_.size(); ++i) global_count_[i] = 0;
      CountOccurrencies(begin_, itr, global_count_.begin(), global_count_.end(),
                        num_threads_);
    }
    end_ = itr;
  }

  void ReduceCounters(size_t step) {
#ifdef RIPPLES_ENABLE_CUDA
    if (step == 1 && has_work()) {
      cuda_set_device(size_t(0));

      cuda_h2d(reinterpret_cast<void *>(d_cpu_counters_),
               reinterpret_cast<void *>(global_count_.data()),
               sizeof(uint32_t) * global_count_.size());
    }
#endif
  }

 private:
  std::vector<vertex_type> &global_count_;
  std::vector<std::pair<vertex_type, size_t>> &queue_storage_;
  rrr_set_iterator begin_;
  rrr_set_iterator end_;
  size_t num_threads_;
  uint32_t *d_cpu_counters_;
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
#ifdef RIPPLES_ENABLE_CUDA
  using gpu_worker_type = GPUFindMostInfluentialWorker<GraphTy>;
#endif
  using rrr_set_iterator =
      typename FindMostInfluentialWorker<GraphTy>::rrr_set_iterator;

  CompareHeap<GraphTy> cmpHeap;
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, size_t>,
                          std::vector<std::pair<vertex_type, size_t>>,
                          decltype(cmpHeap)>;

 public:
  StreamingFindMostInfluential(const GraphTy &G, RRRsets<GraphTy> &RRRsets,
                               size_t num_max_cpus, size_t num_gpus)
      : num_cpu_workers_(num_max_cpus),
        num_gpu_workers_(num_gpus),
        workers_(),
        vertex_coverage_(G.num_nodes()),
        queue_storage_(G.num_nodes()),
        d_counters_(num_gpus, 0),
        RRRsets_(RRRsets),
        reduction_steps_(1),
        d_cpu_counters_(nullptr) {
#ifdef RIPPLES_ENABLE_CUDA
    // Get Number of device and allocate 1 thread each.
    // num_gpu_workers_ = cuda_num_devices();
    num_cpu_workers_ -= num_gpu_workers_;

    std::fill(vertex_coverage_.begin(), vertex_coverage_.end(), 0);

    // Allocate Counters
    if (num_gpu_workers_ > 0) {
#pragma omp parallel num_threads(num_gpu_workers_)
      {
        size_t rank = omp_get_thread_num();
        cuda_set_device(rank);
        cuda_malloc(reinterpret_cast<void **>(&d_counters_[rank]),
                    sizeof(uint32_t) * G.num_nodes());

        if (rank == 0) {
          cuda_malloc(reinterpret_cast<void **>(&d_cpu_counters_),
                      sizeof(uint32_t) * G.num_nodes());
        }
      }
    }
#endif

    workers_.push_back(new CPUFindMostInfluentialWorker<GraphTy>(
        vertex_coverage_, queue_storage_, RRRsets_.begin(), RRRsets_.end(),
        num_cpu_workers_, d_cpu_counters_));
#ifdef RIPPLES_ENABLE_CUDA
    if (num_gpu_workers_ == 0) return;

    // Define Reduction tree on GPU workers.
    auto tree = cuda_get_reduction_tree();

    // Construct GPU workers
    for (size_t i = 0; i < num_gpu_workers_; ++i) {
      reduction_steps_ = std::max(reduction_steps_, tree[i].second);


      uint32_t *dest = i == 0 ? d_cpu_counters_ : d_counters_[tree[i].first];

      workers_.push_back(new GPUFindMostInfluentialWorker<GraphTy>(
          i, G.num_nodes(), d_counters_, tree[i].first, tree[i].second, dest));
    }
#endif
  }

  ~StreamingFindMostInfluential() {
#ifdef RIPPLES_ENABLE_CUDA
    for (auto b : d_counters_) {
      cuda_free(b);
    }
    if (num_gpu_workers_ > 0) cuda_free(d_cpu_counters_);
#endif
    for (auto w : workers_) {
      delete w;
    }
  }

  void InitialCount() {
#pragma omp parallel num_threads(num_gpu_workers_ + 1)
    {
      size_t rank = omp_get_thread_num();
      workers_[rank]->InitialCount();
    }
  }

  void ReduceCounters() {
    if (num_gpu_workers_ == 0) return;

    if (!workers_[0]->has_work() && num_gpu_workers_ == 1) return;

    for (ssize_t i = reduction_steps_; i >= 0; --i) {
#pragma omp parallel num_threads(num_gpu_workers_ + 1)
      {
        size_t rank = omp_get_thread_num();

        if (workers_[rank]->has_work()) {
          workers_[rank]->ReduceCounters(i);
        }
      }
    }
  }

  void UpdateCounters(vertex_type last_seed) {
#pragma omp parallel num_threads(num_gpu_workers_ + 1)
    {
      size_t rank = omp_get_thread_num();
      workers_[rank]->UpdateCounters(last_seed);
    }
  }

  priorityQueue getHeap() {
    priorityQueue queue(cmpHeap, std::move(queue_storage_));
    return queue;
  }

  std::pair<vertex_type, size_t> getNextSeed(priorityQueue &queue_) {
#ifdef RIPPLES_ENABLE_CUDA
    if (num_gpu_workers_ != 0) {
      ReduceCounters();

      uint32_t *global_counter = d_counters_[0];
      if (workers_[0]->has_work()) global_counter = d_cpu_counters_;

      cuda_set_device(0);
      auto result = CudaMaxElement(global_counter, vertex_coverage_.size());
      return result;
    }
#endif

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

  void LoadDataToDevice() {
    if (num_gpu_workers_ == 0) return;

    std::vector<PartitionIndices<rrr_set_iterator>> indices(num_gpu_workers_);
#pragma omp parallel num_threads(num_gpu_workers_ + 1)
    {
      size_t rank = omp_get_thread_num();
      if (rank != 0) {
        size_t threadnum = omp_get_thread_num() - 1,
               numthreads = omp_get_num_threads() - 1;
        size_t low = RRRsets_.size() * threadnum / numthreads,
               high = RRRsets_.size() * (threadnum + 1) / numthreads;

        indices[threadnum] = workers_[rank]->LoadData(
            RRRsets_.begin() + low,
            std::min(RRRsets_.end(), RRRsets_.begin() + high));
      }
    }

    size_t num_threads = num_gpu_workers_;
    for (size_t j = 1; j < num_threads; j <<= 1) {
#pragma omp parallel num_threads(num_threads >> j)
      {
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < (num_threads - j); i += j * 2) {
          indices[i] = indices[i].mergeBlocks(indices[i + j],
                                              std::min(2 * j, num_threads));
        }
      }
    }
    workers_[0]->set_first_rrr_set(indices[0].pivot);
  }

  auto find_most_influential_set(size_t k) {
    omp_set_max_active_levels(2);

    LoadDataToDevice();

    InitialCount();

    auto queue = getHeap();

    std::vector<vertex_type> result;
    result.reserve(k);
    size_t uncovered = RRRsets_.size();

    std::chrono::duration<double, std::milli> seedSelection(0);
    while (uncovered != 0) {
      auto start = std::chrono::high_resolution_clock::now();
      auto element = getNextSeed(queue);
      auto end = std::chrono::high_resolution_clock::now();

      seedSelection += end - start;

      uncovered -= element.second;
      result.push_back(element.first);

      if (result.size() == k) break;

      UpdateCounters(element.first);
    }

    double f = double(RRRsets_.size() - uncovered) / RRRsets_.size();

    omp_set_max_active_levels(1);

    return std::make_pair(f, result);
  }

 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  ssize_t reduction_steps_;
  RRRsets<GraphTy> &RRRsets_;
  std::vector<worker_type *> workers_;
  std::vector<uint32_t *> d_counters_;
  uint32_t *d_cpu_counters_;
  std::vector<uint32_t> vertex_coverage_;
  std::vector<std::pair<vertex_type, size_t>> queue_storage_;
};

}  // namespace ripples

#endif
