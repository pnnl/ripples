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

#ifndef RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H
#define RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H

#include "ripples/find_most_influential.h"
#include "ripples/streaming_find_most_influential.h"
#include "ripples/utility.h"

#include "spdlog/spdlog.h"

namespace ripples {

template <typename GraphTy>
class MPIStreamingFindMostInfluential {
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
  MPIStreamingFindMostInfluential(const GraphTy &G, RRRsets<GraphTy> &RRRsets, size_t num_gpus)
      : num_cpu_workers_(0)
      , num_gpu_workers_(num_gpus)
      , workers_()
      , vertex_coverage_(G.num_nodes(), 0)
      , reduced_vertex_coverage_(G.num_nodes(), 0)
      , queue_storage_(G.num_nodes())
      , d_counters_(num_gpus, 0)
      , RRRsets_(RRRsets)
      , reduction_steps_(1)
      , d_cpu_counters_(nullptr)
  {
    #pragma omp single
    { num_cpu_workers_ = omp_get_max_threads(); }

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);


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
        cuda_malloc(reinterpret_cast<void**>(&d_counters_[rank]), sizeof(uint32_t) * G.num_nodes());

        if (rank == 0) {
          cuda_malloc(reinterpret_cast<void**>(&d_cpu_counters_), sizeof(uint32_t) * G.num_nodes());
	  cuda_malloc(reinterpret_cast<void**>(&d_cpu_reduced_counters_), sizeof(uint32_t) * G.num_nodes());
        }
      }
    }
#endif

    workers_.push_back(
        new CPUFindMostInfluentialWorker<GraphTy>(vertex_coverage_,
                                                  queue_storage_,
                                                  RRRsets_.begin(),
                                                  RRRsets_.end(),
                                                  num_cpu_workers_,
                                                  d_cpu_counters_));
#ifdef RIPPLES_ENABLE_CUDA
    if (num_gpu_workers_ == 0) return;

    // Define Reduction tree on GPU workers.
    auto tree = cuda_get_reduction_tree();

    // Construct GPU workers
    for (size_t i = 0; i < num_gpu_workers_; ++i) {
      reduction_steps_ = std::max(reduction_steps_, tree[i].second);

      // std::cout << "step " << tree[i].second << " : " << i << " -> " << tree[i].first << std::endl;

      uint32_t * dest = i == 0 ? d_cpu_counters_ : d_counters_[tree[i].first];

      workers_.push_back(
          new GPUFindMostInfluentialWorker<GraphTy>(
              i, G.num_nodes(), d_counters_, tree[i].first, tree[i].second, dest));
    }
#endif
  }

  ~MPIStreamingFindMostInfluential() {
#ifdef RIPPLES_ENABLE_CUDA
    for (auto b : d_counters_) {
      cuda_free(b);
    }
    if (num_gpu_workers_ > 0) {
      cuda_free(d_cpu_counters_);
      cuda_free(d_cpu_reduced_counters_);
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
      workers_[rank]->InitialCount();
    }
  }

  void ReduceCounters() {
    uint32_t * dest = reduced_vertex_coverage_.data();
    uint32_t * src = vertex_coverage_.data();

    if (num_gpu_workers_ != 0) {
      dest = d_cpu_reduced_counters_;
      src = d_cpu_counters_;

      // if (workers_[0]->has_work() || num_gpu_workers_ > 1) {

	for (ssize_t i = reduction_steps_; i >= 0; --i) {
          #pragma omp parallel num_threads(num_gpu_workers_ + 1)
	  {
	    size_t rank = omp_get_thread_num();

	    if (workers_[rank]->has_work()) {
	      workers_[rank]->ReduceCounters(i);
	    }
	  }
	// }
      }

    // std::cout << "Before Reduction " << src << std::endl;
    std::vector<uint32_t> tmp(vertex_coverage_.size(), 0);
    cuda_set_device(0);

    cuda_d2h(reinterpret_cast<void*>(tmp.data()),
             reinterpret_cast<void*>(src),
             sizeof(uint32_t) * vertex_coverage_.size());
      
    MPI_Reduce(tmp.data(), reduced_vertex_coverage_.data(), vertex_coverage_.size(),
               MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

    cuda_h2d(reinterpret_cast<void*>(dest),
             reinterpret_cast<void*>(reduced_vertex_coverage_.data()),
             sizeof(uint32_t) * vertex_coverage_.size());
    // std::cout << "After Reduction " << std::endl;
    } else {
      MPI_Reduce(src, dest, vertex_coverage_.size(),
                 MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
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

  std::pair<vertex_type, size_t> getNextSeed(priorityQueue & queue_) {
    ReduceCounters();
#ifdef RIPPLES_ENABLE_CUDA
    if (num_gpu_workers_ != 0) {
      uint32_t * global_counter = d_cpu_reduced_counters_;

      if (mpi_rank == 0) {
        cuda_set_device(0);
        auto result = CudaMaxElement(global_counter, vertex_coverage_.size());
        coveredAndSelected[0] += result.second;
        coveredAndSelected[1] = result.first;
      }

      MPI_Bcast(&coveredAndSelected, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
      // std::cout << "$$$$ " << mpi_rank << " "<< coveredAndSelected[0] << std::endl;
      return std::pair<vertex_type, size_t>(coveredAndSelected[1],
					    coveredAndSelected[0]);
    }
#endif

    while (!queue_.empty()) {
      if (mpi_rank == 0) {
        auto element = queue_.top();
        queue_.pop();

        if (element.second > reduced_vertex_coverage_[element.first]) {
          element.second = reduced_vertex_coverage_[element.first];
          queue_.push(element);
          continue;
        }
        coveredAndSelected[0] += element.second;
        coveredAndSelected[1] = element.first;
      }

      MPI_Bcast(&coveredAndSelected, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        
      return std::pair<vertex_type, size_t>(coveredAndSelected[1],
					    coveredAndSelected[0]);
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
	size_t threadnum = omp_get_thread_num() - 1, numthreads = omp_get_num_threads() - 1;
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
    omp_set_nested(true);

    LoadDataToDevice();
    
    InitialCount();
    // std::cout << "Initial Count Done" << std::endl;

    auto queue = getHeap();
    std::vector<vertex_type> result;
    result.reserve(k);

    std::chrono::duration<double, std::milli> seedSelection(0);
    while (true) {
      //      std::cout << "Get Seed" << std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      auto element = getNextSeed(queue);
      auto end = std::chrono::high_resolution_clock::now();

      seedSelection += end - start;
      // std::cout << "Selected : " << element.first << " " << element.second << std::endl;

      result.push_back(element.first);

      if (result.size() == k) break;

      // std::cout << "Update counters" << std::endl;
      UpdateCounters(element.first);
      // std::cout << "Done update counters" << std::endl;
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double f = double(coveredAndSelected[0]) / (world_size * RRRsets_.size());
    if (mpi_rank == 0) {
      std::cout << f << " = " << double(coveredAndSelected[0]) << "/ (" << world_size << " * " <<
          RRRsets_.size() << ")" << std::endl;
    }
    // double f = double(RRRsets_.size() - uncovered) / RRRsets_.size();

    // std::cout << "#### " << seedSelection.count() << std::endl;

    omp_set_nested(false);

    return std::make_pair(f, result);
  }
  
 private:
  size_t num_cpu_workers_, num_gpu_workers_;
  ssize_t reduction_steps_;
  RRRsets<GraphTy> &RRRsets_;
  std::vector<worker_type *> workers_;
  std::vector<uint32_t *> d_counters_;
  uint32_t * d_cpu_counters_;
  uint32_t * d_cpu_reduced_counters_;
  std::vector<uint32_t> vertex_coverage_;
  std::vector<uint32_t> reduced_vertex_coverage_;
  std::vector<std::pair<vertex_type, size_t>> queue_storage_;
  int mpi_rank;
  uint32_t coveredAndSelected[2] = {0, 0};
};


//! \brief Select k seeds starting from the a list of Random Reverse
//! Reachability Sets.
//!
//! \tparam GraphTy The graph type.
//! \tparam RRRset The type storing Random Reverse Reachability Sets.
//!
//! \param G The input graph.
//! \param k The size of the seed set.
//! \param RRRsets A vector of Random Reverse Reachability sets.
//! \param ex_tag The MPI+OpenMP execution tag.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.
#if 0
template <typename GraphTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            std::vector<RRRset> &RRRsets,
                            mpi_omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<uint32_t> vertexCoverage(G.num_nodes(), 0);
  std::vector<uint32_t> reduceCoverageInfo(G.num_nodes(), 0);

  auto cmp = [](std::pair<vertex_type, uint32_t> &a,
                std::pair<vertex_type, uint32_t> &b) {
    return a.second < b.second;
  };
  using priorityQueue =
      std::priority_queue<std::pair<vertex_type, uint32_t>,
                          std::vector<std::pair<vertex_type, uint32_t>>,
                          decltype(cmp)>;

  MPI_Win win;
  MPI_Win_create(reduceCoverageInfo.data(), G.num_nodes() * sizeof(uint32_t),
                 sizeof(uint32_t), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  CountOccurrencies(RRRsets.begin(), RRRsets.end(), vertexCoverage.begin(),
                    vertexCoverage.end(),
                    std::forward<omp_parallel_tag>(omp_parallel_tag{}));

  MPI_Win_fence(0, win);
  MPI_Accumulate(vertexCoverage.data(), G.num_nodes(), MPI_UINT32_T, 0, 0,
                 G.num_nodes(), MPI_UINT32_T, MPI_SUM, win);
  MPI_Win_fence(0, win);

  MPI_Win_free(&win);

  std::vector<std::pair<vertex_type, uint32_t>> queue_storage;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    queue_storage.resize(G.num_nodes());
    InitHeapStorage(reduceCoverageInfo.begin(), reduceCoverageInfo.end(),
                    queue_storage.begin(), queue_storage.end(),
                    std::forward<omp_parallel_tag>(omp_parallel_tag{}));
  }
  priorityQueue queue(cmp, std::move(queue_storage));

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  auto end = RRRsets.end();
  uint32_t coveredAndSelected[2] = {0, 0};

  while (result.size() < k) {
    if (rank == 0) {
      auto element = queue.top();
      queue.pop();

      if (element.second > reduceCoverageInfo[element.first]) {
        element.second = reduceCoverageInfo[element.first];
        queue.push(element);
        continue;
      }
      coveredAndSelected[0] += element.second;
      coveredAndSelected[1] = element.first;
    }

    MPI_Bcast(&coveredAndSelected, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    vertex_type v = coveredAndSelected[1];
    auto cmp = [=](const RRRset &a) -> auto {
      return !std::binary_search(a.begin(), a.end(), v);
    };

    auto itr = partition(RRRsets.begin(), end, cmp, omp_parallel_tag{});

    if (std::distance(itr, end) < std::distance(RRRsets.begin(), itr)) {
      UpdateCounters(itr, end, vertexCoverage, omp_parallel_tag{});
    } else {
#pragma omp parallel for simd
      for (size_t i = 0; i < vertexCoverage.size(); ++i) vertexCoverage[i] = 0;

      CountOccurrencies(RRRsets.begin(), itr, vertexCoverage.begin(),
                        vertexCoverage.end(), omp_parallel_tag{});
    }

    end = itr;

    MPI_Reduce(vertexCoverage.data(), reduceCoverageInfo.data(), G.num_nodes(),
               MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

    result.push_back(v);
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  double f = double(coveredAndSelected[0]) / (world_size * RRRsets.size());

  return std::make_pair(f, result);
}
#endif

template <typename GraphTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            std::vector<RRRset> &RRRsets,
			    bool enableGPU,
                            mpi_omp_parallel_tag &&ex_tag) {
  size_t num_gpu = 0;
#ifdef RIPPLES_ENABLE_CUDA
  if (enableGPU) {
    num_gpu = cuda_num_devices();
  }
#endif
  MPIStreamingFindMostInfluential<GraphTy> SE(G, RRRsets, num_gpu);
  return SE.find_most_influential_set(k);
}


}  // namespace ripples

#endif  // RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H
