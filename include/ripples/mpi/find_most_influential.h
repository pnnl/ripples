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

#include "ripples/generate_rrr_sets.h"
#include "ripples/partition.h"

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
#include "ripples/gpu/find_most_influential.h"
#include "ripples/gpu/gpu_graph.h"
#include "ripples/gpu/gpu_runtime_trait.h"
#endif
#include "spdlog/spdlog.h"

#define PRINTF_TIL_YOU_DROP

#ifdef _WIN32
#include <windows.h>

// Function to get memory information on Windows
size_t GetAvailableMemory() {
  MEMORYSTATUSEX memoryStatus;
  memoryStatus.dwSize = sizeof(MEMORYSTATUSEX);
  if (!GlobalMemoryStatusEx(&memoryStatus)) {
    std::cerr << "Error: Failed to get memory status." << std::endl;
    exit(1);
  }
  return memoryStatus.ullAvailPhys;
}

#elif __linux__
#include <sys/sysinfo.h>

// Function to get memory information on Linux
size_t GetAvailableMemory() {
  struct sysinfo info;
  if (sysinfo(&info) == -1) {
    std::cerr << "Error: Failed to get memory status." << std::endl;
    exit(1);
  }
  return info.freeram;
}

#else
#error "Unsupported platform"
#endif

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
  MPIStreamingFindMostInfluential(const GraphTy &G, RRRsets<GraphTy> &RRRsets,
                                  size_t num_max_cpu, size_t num_gpus)
      : num_cpu_workers_(num_max_cpu),
        num_gpu_workers_(num_gpus),
        workers_(),
        vertex_coverage_(G.num_nodes(), 0),
        reduced_vertex_coverage_(G.num_nodes(), 0),
        queue_storage_(G.num_nodes()),
        d_counters_(num_gpus, 0),
        RRRsets_(RRRsets),
        reduction_steps_(1),
        d_cpu_counters_(nullptr) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
    // Get Number of device and allocate 1 thread each.
    // num_gpu_workers_ = cuda_num_devices();
    num_cpu_workers_ -= num_gpu_workers_;

    // Allocate Counters
    if (num_gpu_workers_ > 0) {
#pragma omp parallel num_threads(num_gpu_workers_)
      {
        size_t rank = omp_get_thread_num();
        GPU<RUNTIME>::set_device(rank);
        GPU<RUNTIME>::device_malloc(
            reinterpret_cast<void **>(&d_counters_[rank]),
            sizeof(uint32_t) * G.num_nodes());

        if (rank == 0) {
          GPU<RUNTIME>::device_malloc(
              reinterpret_cast<void **>(&d_cpu_counters_),
              sizeof(uint32_t) * G.num_nodes());
          GPU<RUNTIME>::device_malloc(
              reinterpret_cast<void **>(&d_cpu_reduced_counters_),
              sizeof(uint32_t) * G.num_nodes());
        }
      }
    }
#endif

    workers_.push_back(new CPUFindMostInfluentialWorker<GraphTy>(
        vertex_coverage_, queue_storage_, RRRsets_.begin(), RRRsets_.end(),
        num_cpu_workers_, d_cpu_counters_));
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
    if (num_gpu_workers_ == 0) return;

    // Define Reduction tree on GPU workers.
    auto tree = GPU<RUNTIME>::build_reduction_tree();

    // Construct GPU workers
    for (size_t i = 0; i < num_gpu_workers_; ++i) {
      reduction_steps_ = std::max<size_t>(reduction_steps_, tree[i].second);

      // std::cout << "step " << tree[i].second << " : " << i << " -> " <<
      // tree[i].first << std::endl;

      uint32_t *dest = i == 0 ? d_cpu_counters_ : d_counters_[tree[i].first];

      workers_.push_back(new GPUFindMostInfluentialWorker<GraphTy>(
          i, G.num_nodes(), d_counters_, tree[i].first, tree[i].second, dest));
    }
#endif
  }

  ~MPIStreamingFindMostInfluential() {
#if defined(RIPPLES_ENABLE_CUDA) || defined (RIPPLES_ENABLE_HIP)
    for (auto b : d_counters_) {
      GPU<RUNTIME>::device_free(b);
    }
    if (num_gpu_workers_ > 0) {
      GPU<RUNTIME>::device_free(d_cpu_counters_);
      GPU<RUNTIME>::device_free(d_cpu_reduced_counters_);
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
      #ifdef PRINTF_TIL_YOU_DROP
      auto console = spdlog::get("console");
      console->info("Rank = {}, InitialCount {}", mpi_rank, rank);
      auto timeInitialCountStart = std::chrono::high_resolution_clock::now();
      #endif // PRINTF_TIL_YOU_DROP
      workers_[rank]->InitialCount();
    }
  }

  void To1D() {
    assert(workers_.size() == 1);
    rr_sizes_1d_.resize(workers_[0]->get_num_rr_sets());
    rr_sets_1d_.resize(workers_[0]->get_rr_set_size());

    auto begin = workers_[0]->get_begin();
    auto end = workers_[0]->get_end();
    size_t running_size = 0;
    size_t i = 0;
    for (auto itr = begin; itr != end; ++itr, ++i) {
      rr_sizes_1d_[i] = itr->size();
      running_size += itr->size();
      std::copy(itr->begin(), itr->end(), rr_sets_1d_.begin() + running_size);
    }
  }

  void From1D() {
    assert(workers_.size() == 1);
    #ifdef PRINTF_TIL_YOU_DROP
    auto console = spdlog::get("console");
    console->info("Rank = {}, From1D", mpi_rank);
    auto timeFrom1DStart = std::chrono::high_resolution_clock::now();
    #endif // PRINTF_TIL_YOU_DROP
    RRRsets_gathered_.resize(rr_sizes_1d_.size());
    // Prefix sum to get the offsets
    std::vector<size_t> rr_sizes_1d_prefix_sum(rr_sizes_1d_.size() + 1);
    rr_sizes_1d_prefix_sum[0] = 0;
    #ifdef PRINTF_TIL_YOU_DROP
    console->info("Rank = {}, Prefix Sum {} offsets", mpi_rank, rr_sizes_1d_.size());
    #endif // PRINTF_TIL_YOU_DROP
    std::partial_sum(rr_sizes_1d_.begin(), rr_sizes_1d_.end(),
                     rr_sizes_1d_prefix_sum.begin() + 1);
    #pragma omp parallel for
    for (size_t i = 0; i < rr_sizes_1d_.size(); ++i) {
      RRRsets_gathered_[i].resize(rr_sizes_1d_[i]);
      std::copy(rr_sets_1d_.begin() + rr_sizes_1d_prefix_sum[i],
                rr_sets_1d_.begin() + rr_sizes_1d_prefix_sum[i + 1],
                RRRsets_gathered_[i].begin());
    }
    #ifdef PRINTF_TIL_YOU_DROP
    // Sanity check on the gathered RRR sets
    size_t rr_set_size = 0;
    size_t num_rr_sets = RRRsets_gathered_.size();
    for (size_t i = 0; i < RRRsets_gathered_.size(); ++i) {
      rr_set_size += RRRsets_gathered_[i].size();
    }
    console->info("Rank = {}, Sanity Aggregated RR Set Size: {}, Num RR Sets: {}", mpi_rank, rr_set_size, num_rr_sets);
    #endif
  }

  void ConvergenceCheck() {
    assert(workers_.size() == 1);
    auto console = spdlog::get("console");
    #ifdef PRINTF_TIL_YOU_DROP
    console->info("Rank = {}, Checking Convergence", mpi_rank);
    auto timeConvergenceCheckStart = std::chrono::high_resolution_clock::now();
    #endif // PRINTF_TIL_YOU_DROP
    // Gather total size of all RR sets across ranks, send to rank 0
    constexpr int max_int32 = std::numeric_limits<int32_t>::max();
    size_t rr_set_size = workers_[0]->get_rr_set_size();
    size_t total_rr_set_size;
    size_t num_rr_sets = workers_[0]->get_num_rr_sets();
    MPI_Reduce(&rr_set_size, &total_rr_set_size, 1, MPI_UINT64_T,
               MPI_SUM, 0, MPI_COMM_WORLD);
    bool converged = false;
    if (mpi_rank == 0) {
      // Get available free memory on the host
      size_t free_memory = GetAvailableMemory();
      if (total_rr_set_size < max_int32) {
        if (total_rr_set_size * sizeof(vertex_type) < free_memory / 4) {
          converged = true;
          console->info("Converged: {} < {}",
                        total_rr_set_size * sizeof(vertex_type),
                        free_memory / 4);
        } else {
          console->info("Not Converged: {} >= {}",
                        total_rr_set_size * sizeof(vertex_type),
                        free_memory / 4);
        }
      } else {
        console->info("Not Converged (maxint32): {} >= {}", total_rr_set_size,
                      max_int32);
      }
    }
    MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(converged) {
      To1D();
      // Allocate enough memory on rank 0 to receive all RR offsets
      std::vector<size_t> rr_sizes_1d_gathered;
      std::vector<int> rr_sizes_1d_gathered_recv;
      std::vector<int> rr_sizes_1d_gathered_displ;
      std::vector<vertex_type> rr_sets_1d_gathered;
      std::vector<int> rr_sets_1d_gathered_recv;
      std::vector<int> rr_sets_1d_gathered_displ;
      rr_sizes_1d_gathered_recv.resize(world_size);
      rr_sets_1d_gathered_recv.resize(world_size);
      int int32_num_rr_sets = static_cast<int>(num_rr_sets);
      int int32_rr_set_size = static_cast<int>(rr_set_size);
      #ifdef PRINTF_TIL_YOU_DROP
      console->info("Rank = {}, Broadcasting RR Set Size: {}", mpi_rank, int32_rr_set_size);
      #endif // PRINTF_TIL_YOU_DROP
      MPI_Allgather(&int32_num_rr_sets, 1, MPI_INT32_T, rr_sizes_1d_gathered_recv.data(), 1,
              MPI_INT32_T, MPI_COMM_WORLD);
      MPI_Allgather(&int32_rr_set_size, 1, MPI_INT32_T, rr_sets_1d_gathered_recv.data(), 1,
              MPI_INT32_T, MPI_COMM_WORLD);
      #ifdef PRINTF_TIL_YOU_DROP
      console->info("Rank = {}, RR Set Size Broadcasted", mpi_rank);
      #endif // PRINTF_TIL_YOU_DROP
      size_t total_rr_sets = std::accumulate(rr_sizes_1d_gathered_recv.begin(),
                                              rr_sizes_1d_gathered_recv.end(), 0);
      rr_sizes_1d_gathered.resize(total_rr_sets);
      rr_sets_1d_gathered.resize(total_rr_set_size);
      rr_sizes_1d_gathered_displ.resize(world_size+1);
      rr_sets_1d_gathered_displ.resize(world_size+1);

      rr_sizes_1d_gathered_displ[0] = 0;
      std::partial_sum(rr_sizes_1d_gathered_recv.begin(),
                        rr_sizes_1d_gathered_recv.end(),
                        rr_sizes_1d_gathered_displ.begin()+1);

      rr_sets_1d_gathered_displ[0] = 0;
      std::partial_sum(rr_sets_1d_gathered_recv.begin(),
                        rr_sets_1d_gathered_recv.end(),
                        rr_sets_1d_gathered_displ.begin()+1);
      #ifdef PRINTF_TIL_YOU_DROP
      console->info("Rank = {}, RR Set Size Displacement Calculated", mpi_rank);
      for (int i = 0; i < world_size; ++i) {
        console->info("Rank = {}, Index = {}, RR Set Size Recv: {}", mpi_rank, i, rr_sets_1d_gathered_recv[i]);
        console->info("Rank = {}, Index = {}, RR Set Size Dipl: {}", mpi_rank, i, rr_sets_1d_gathered_displ[i]);
        console->info("Rank = {}, Index = {}, RR Sizes Recv: {}", mpi_rank, i, rr_sizes_1d_gathered_recv[i]);
        console->info("Rank = {}, Index = {}, RR Sizes Displ: {}", mpi_rank, i, rr_sizes_1d_gathered_displ[i]);
      }
      #endif // PRINTF_TIL_YOU_DROP
      // Gather all RR sets on rank 0
      MPI_Gatherv(rr_sizes_1d_.data(), static_cast<int>(rr_sizes_1d_.size()), MPI_UINT64_T,
                  rr_sizes_1d_gathered.data(), rr_sizes_1d_gathered_recv.data(),
                  rr_sizes_1d_gathered_displ.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD);
      
      MPI_Gatherv(rr_sets_1d_.data(), static_cast<int>(rr_sets_1d_.size()), MPI_UINT32_T,
                  rr_sets_1d_gathered.data(), rr_sets_1d_gathered_recv.data(),
                  rr_sets_1d_gathered_displ.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
      #ifdef PRINTF_TIL_YOU_DROP
      console->info("Rank = {}, RR Sets Gathered", mpi_rank);
      #endif // PRINTF_TIL_YOU_DROP
      if (mpi_rank == 0) {
        #ifdef PRINTF_TIL_YOU_DROP
        console->info("Rank = {}, From1D", mpi_rank);
        #endif // PRINTF_TIL_YOU_DROP
        rr_sets_1d_ = std::move(rr_sets_1d_gathered);
        rr_sizes_1d_ = std::move(rr_sizes_1d_gathered);
        From1D();
        #ifdef PRINTF_TIL_YOU_DROP
        console->info("Rank = {}, From1D Done", mpi_rank);
        #endif // PRINTF_TIL_YOU_DROP
        // Fill vertex_coverage_ to 0 in parallel
        std::fill(vertex_coverage_.begin(), vertex_coverage_.end(), 0);
        workers_[0] = new CPUFindMostInfluentialWorker<GraphTy>(
            vertex_coverage_, queue_storage_, RRRsets_gathered_.begin(),
            RRRsets_gathered_.end(), num_cpu_workers_, d_cpu_counters_);
        #ifdef PRINTF_TIL_YOU_DROP
        console->info("Rank = {}, Re-Initial Count", mpi_rank);
        #endif // PRINTF_TIL_YOU_DROP
        CountOccurrencies(RRRsets_gathered_.begin(), RRRsets_gathered_.end(),
                            vertex_coverage_.begin(), vertex_coverage_.end(),
                            num_cpu_workers_);
        #ifdef PRINTF_TIL_YOU_DROP
        console->info("Rank = {}, Re-Initial Count Done", mpi_rank);
        #endif // PRINTF_TIL_YOU_DROP
      }
    }
    #ifdef PRINTF_TIL_YOU_DROP
    auto timeConvergenceCheckEnd = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timeConvergenceCheckEnd - timeConvergenceCheckStart);
    console->info("Rank = {}, Finished Convergence Check: {} ms", mpi_rank, duration_ms.count());
    #endif // PRINTF_TIL_YOU_DROP
    mpi_converged_ = converged;
  }

  void ReduceCounters() {
    if(mpi_converged_) return;
    uint32_t *dest = reduced_vertex_coverage_.data();
    uint32_t *src = vertex_coverage_.data();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::vector<MPI_Request> requests(world_size);
    uint32_t chunk_size = vertex_coverage_.size() / world_size;
    uint32_t last_block_size = chunk_size + vertex_coverage_.size() % world_size;

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
    if (num_gpu_workers_ != 0) {
      dest = d_cpu_reduced_counters_;
      src = d_cpu_counters_;

      GPU<RUNTIME>::memset(reinterpret_cast<void *>(src), 0,
                           sizeof(uint32_t) * vertex_coverage_.size());

      for (ssize_t i = reduction_steps_; i >= 0; --i) {
#pragma omp parallel num_threads(num_gpu_workers_ + 1)
        {
          size_t rank = omp_get_thread_num();

          workers_[rank]->ReduceCounters(i);
        }
      }

      // std::cout << "Before Reduction " << src << std::endl;
      std::vector<uint32_t> tmp(vertex_coverage_.size(), 0);
      GPU<RUNTIME>::set_device(0);

      GPU<RUNTIME>::memset(reinterpret_cast<void *>(dest), 0,
                           sizeof(uint32_t) * vertex_coverage_.size());

      GPU<RUNTIME>::d2h(reinterpret_cast<void *>(tmp.data()),
                        reinterpret_cast<void *>(src),
                        sizeof(uint32_t) * vertex_coverage_.size());

      // MPI_Reduce(src, dest, vertex_coverage_.size(),
      // 	    MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
      for (size_t i = 0; i < world_size; ++i) {
        MPI_Ireduce(tmp.data() + i * chunk_size,
                    reduced_vertex_coverage_.data() + i * chunk_size,
                    i != (world_size - 1) ? chunk_size : last_block_size,
                    MPI_UINT32_T, MPI_SUM, i, MPI_COMM_WORLD, &requests[i]);
      }

    } else
#endif
    {
      for (size_t i = 0; i < world_size; ++i) {
        MPI_Ireduce(src + i * chunk_size, dest + i * chunk_size,
                    i != (world_size - 1) ? chunk_size : last_block_size,
                    MPI_UINT32_T, MPI_SUM, i, MPI_COMM_WORLD, &requests[i]);
      }
    }
    for (auto & request : requests)
      MPI_Wait(&request, MPI_STATUS_IGNORE);
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
    if (mpi_converged_) {
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
      if (num_gpu_workers_ != 0) {
        ReduceCounters();

        uint32_t *global_counter = d_counters_[0];
        if (workers_[0]->has_work()) global_counter = d_cpu_counters_;

        GPU<RUNTIME>::set_device(0);
        auto result = GPUMaxElement(global_counter, vertex_coverage_.size());
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
        coveredAndSelected[0] += element.second;
        coveredAndSelected[1] = element.first;
        return element;
      }
      throw std::logic_error("Reached a mighty Unreachable State");
    }
#ifdef PRINTF_TIL_YOU_DROP
    auto console = spdlog::get("console");
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    console->info("Rank = {}, Reducing Counters", world_rank);
    auto timeReduceCountersStart = std::chrono::high_resolution_clock::now();
    #endif // PRINTF_TIL_YOU_DROP
    ReduceCounters();
    #ifdef PRINTF_TIL_YOU_DROP
    auto timeReduceCountersEnd = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timeReduceCountersEnd - timeReduceCountersStart);
    console->info("Rank = {}, Reduced Counters: {} ms", world_rank, duration_ms.count());
    auto timeFindMaxStart = std::chrono::high_resolution_clock::now();
    #endif // PRINTF_TIL_YOU_DROP

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    uint32_t chunk_size = vertex_coverage_.size() / world_size;
    uint32_t last_block_size = chunk_size + vertex_coverage_.size() % world_size;

    uint32_t vertex = 0;
    uint32_t coverage = 0;
    // auto itr = std::max_element(reduced_vertex_coverage_.begin(), reduced_vertex_coverage_.end());
#pragma omp parallel
    {
      uint32_t vertex_local = 0;
      uint32_t coverage_local = 0;

      uint32_t end = std::min<uint32_t>((mpi_rank + 1) * chunk_size, reduced_vertex_coverage_.size());

#pragma omp for
      for (uint32_t i = mpi_rank * chunk_size; i < end; ++i) {
        if (coverage_local < reduced_vertex_coverage_[i]) {
          coverage_local = reduced_vertex_coverage_[i];
          vertex_local = i;
        }
      }

#pragma omp critical
      {
        if (coverage < coverage_local) {
          coverage = coverage_local;
          vertex = vertex_local;
        }
      }
    }

    // for (size_t i = 0; i < world_size; ++i) {
    //   if (mpi_rank == i) {
    //     for (auto v : reduced_vertex_coverage_) {
    //       std::cout << "[" << i << "] " << v << std::endl;
    //     }
    //   }
    //   MPI_Barrier(MPI_COMM_WORLD);
    // }

    coveredAndSelected[0] += coverage;
    coveredAndSelected[1] = vertex;

    #ifdef PRINTF_TIL_YOU_DROP
    auto timeFindMaxEnd = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timeFindMaxEnd - timeFindMaxStart);
    std::string duration_str;
    if (duration_ms.count() == 0){
      duration_str = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
          timeFindMaxEnd - timeFindMaxStart).count()) + " us";
    } else {
      duration_str = std::to_string(duration_ms.count()) + " ms";
    }
    console->info("Rank = {}, Found Local Max: {}", world_rank, duration_str);
    auto timeMPIAllReduceStart = std::chrono::high_resolution_clock::now();
    #endif // PRINTF_TIL_YOU_DROP
    uint32_t result[2];
    MPI_Allreduce(coveredAndSelected, result, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    // MPI_Bcast(&coveredAndSelected, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    #ifdef PRINTF_TIL_YOU_DROP
    auto timeMPIAllReduceEnd = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timeMPIAllReduceEnd - timeMPIAllReduceStart);
    console->info("Rank = {}, MPI Allreduce End: {} ms", world_rank, duration_ms.count());
    #endif // PRINTF_TIL_YOU_DROP

    coveredAndSelected[0] = result[0];
    coveredAndSelected[1] = result[1];

    return std::pair<vertex_type, size_t>(coveredAndSelected[1],
                                          coveredAndSelected[0]);
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

    #ifdef PRINTF_TIL_YOU_DROP
    auto console = spdlog::get("console");
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    console->info("Rank = {}, Entering InitialCount", world_rank);
    auto timeInitialCountStart = std::chrono::high_resolution_clock::now();
    #endif // PRINTF_TIL_YOU_DROP

    InitialCount();
    // std::cout << "Initial Count Done" << std::endl;

    #ifdef PRINTF_TIL_YOU_DROP
    auto timeInitialCountEnd = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timeInitialCountEnd - timeInitialCountStart);
    console->info("Rank = {}, Finished InitialCount: {} ms. Entering while loop!", world_rank,
                  duration_ms.count());
    #endif // PRINTF_TIL_YOU_DROP

    auto queue = getHeap();
    std::vector<vertex_type> result;
    result.reserve(k);

    std::chrono::duration<double, std::milli> seedSelection(0);
    while (true) {
      //      std::cout << "Get Seed" << std::endl;
      #ifdef PRINTF_TIL_YOU_DROP
      console->info("Rank = {}, Getting {}th Seed", world_rank, result.size());
      #endif // PRINTF_TIL_YOU_DROP
      auto start = std::chrono::high_resolution_clock::now();
      auto element = getNextSeed(queue);
      auto end = std::chrono::high_resolution_clock::now();

      #ifdef PRINTF_TIL_YOU_DROP
      auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          end - start);
      console->info("Rank = {}, Got {}th Seed: {} ms, SeedId = {}", world_rank,
                    result.size(), duration_ms.count(), element.first);
#endif // PRINTF_TIL_YOU_DROP

      seedSelection += end - start;

      result.push_back(element.first);

      if (result.size() == k) break;

      // std::cout << "Update counters" << std::endl;
      // std::cout << *std::max_element(vertex_coverage_.begin(), vertex_coverage_.end()) << std::endl;
      #ifdef PRINTF_TIL_YOU_DROP
      console->info("Rank = {}, Updating Counters", world_rank);
      auto timeUpdateCountersStart = std::chrono::high_resolution_clock::now();
      #endif // PRINTF_TIL_YOU_DROP
      UpdateCounters(element.first);
      #ifdef PRINTF_TIL_YOU_DROP
      auto timeUpdateCountersEnd = std::chrono::high_resolution_clock::now();
      duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          timeUpdateCountersEnd - timeUpdateCountersStart);
      std::string duration_time;
      if (duration_ms.count() == 0){
        duration_time = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
            timeUpdateCountersEnd - timeUpdateCountersStart).count()) + " us";
      } else {
        duration_time = std::to_string(duration_ms.count()) + " ms";
      }
      console->info(
          "Rank = {}, Current Seed = {}, Updated Counters: {} RR Sets Remaning: {} Local RR "
          "Set Size {}",
          world_rank, result.size(), duration_time, workers_[0]->get_num_rr_sets(),
          workers_[0]->get_rr_set_size());
#endif // PRINTF_TIL_YOU_DROP
      if(!mpi_converged_){
        ConvergenceCheck();
      }
      if (mpi_converged_ && mpi_rank != 0) break;
      // std::cout << "Done update counters" << std::endl;
      // std::cout << *std::max_element(vertex_coverage_.begin(), vertex_coverage_.end()) << std::endl;
    }

    if(mpi_converged_) {
      // Broadcast from rank 0 coveredAndSelected
      MPI_Bcast(coveredAndSelected, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
      // Broadcast result from rank 0
      result.resize(k);
      MPI_Bcast(result.data(), k, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double f = double(coveredAndSelected[0]) / (world_size * RRRsets_.size());
    // if (mpi_rank == 0) {
    //   std::cout << f << " = " << double(coveredAndSelected[0]) << "/ (" <<
    //   world_size << " * " <<
    //       RRRsets_.size() << ")" << std::endl;
    // }
    // double f = double(RRRsets_.size() - uncovered) / RRRsets_.size();

    // std::cout << "#### " << seedSelection.count() << std::endl;

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
  uint32_t *d_cpu_reduced_counters_;
  std::vector<uint32_t> vertex_coverage_;
  std::vector<uint32_t> reduced_vertex_coverage_;
  std::vector<std::pair<vertex_type, size_t>> queue_storage_;
  std::vector<size_t> rr_sizes_1d_;
  std::vector<vertex_type> rr_sets_1d_;
  RRRsets<GraphTy> RRRsets_gathered_;
  bool mpi_converged_ = false;
  int mpi_rank;
  uint32_t coveredAndSelected[2] = {0, 0};
};

template <typename GraphTy, typename ConfTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, const ConfTy &CFG,
                            std::vector<RRRset> &RRRsets, bool enableGPU,
                            mpi_omp_parallel_tag &&ex_tag) {
  size_t num_gpu = 0;
  size_t num_max_cpu = 0;
#pragma omp single
  {
    num_max_cpu =
        std::min<size_t>(omp_get_max_threads(), CFG.seed_select_max_workers);
  }
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
  if (enableGPU) {
    num_gpu = std::min<size_t>(GPURuntimeTrait<RUNTIME>::num_devices(),
                       CFG.seed_select_max_gpu_workers);
  }
#endif
  MPIStreamingFindMostInfluential<GraphTy> SE(G, RRRsets, num_max_cpu, num_gpu);
  return SE.find_most_influential_set(CFG.k);
}

}  // namespace ripples

#endif  // RIPPLES_MPI_FIND_MOST_INFLUENTIAL_H
