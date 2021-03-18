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

#include <cstdio>
#include <iostream>

#include "ripples/cuda/cuda_utils.h"

namespace ripples {

//
// debug utilities
//
void cuda_check(cudaError_t err, const char *fname, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "> CUDA error @%s:%d: name=%s msg='%s'\n", fname, line,
            cudaGetErrorName(err), cudaGetErrorString(err));
    fflush(stderr);
  }
}
  
void cuda_check(const char *fname, int line) {
  cuda_check(cudaGetLastError(), fname, line);
}

size_t cuda_max_blocks() {
  // TODO query CUDA runtime
  return 1 << 16;
}

size_t cuda_num_devices() {
  int res;
  auto e = cudaGetDeviceCount(&res);
  cuda_check(e, __FILE__, __LINE__);
  return res;
}

void cuda_set_device(size_t gpu_id) {
  auto e = cudaSetDevice(gpu_id);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_stream_create(cudaStream_t *sp) {
  auto e = cudaStreamCreate(sp);
  cuda_check(e, __FILE__, __LINE__);
}

void cuda_stream_destroy(cudaStream_t s) {
  auto e = cudaStreamDestroy(s);
  cuda_check(e, __FILE__, __LINE__);
}

bool cuda_malloc(void **dst, size_t size) {
  cudaError_t e = cudaMalloc(dst, size);
  cuda_check(e, __FILE__, __LINE__);
  return e == cudaSuccess;
}

void cuda_free(void *ptr) {
  cudaError_t e = cudaFree(ptr);
  cuda_check(e, __FILE__, __LINE__);
}

std::pair<std::vector<size_t>, std::vector<size_t>>
cuda_build_topology_graph() {
  size_t num_devices = cuda_num_devices();

  std::vector<size_t> index(num_devices + 1, 0);
  std::vector<size_t> edges;

  for (size_t i = 0; i < num_devices; ++i) {
    index[i + 1] = index[i];
    for (size_t j = 0; j < num_devices; ++j) {
      if (i == j) continue;

      int atomics = 0;
      cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, j, i);

      if (atomics) {
        index[i + 1] += 1;
        edges.push_back(j);
      }
    }
  }

  return std::make_pair(index, edges);
}

  // std::vector<std::pair<size_t, size_t>>
std::vector<std::pair<size_t, ssize_t>> cuda_get_reduction_tree() {
  auto topo = cuda_build_topology_graph();
  auto & index = topo.first;
  auto & edges = topo.second;
  
  size_t num_devices = cuda_num_devices();
  std::vector<bool> visited(num_devices);
  // Predecessor and Level.
  std::vector<std::pair<size_t, ssize_t>> result(num_devices, std::make_pair(size_t(0), ssize_t(-1)));

  std::vector<size_t> queue;
  queue.reserve(edges.size());

  queue.push_back(0);
  result[0] = std::make_pair(size_t(0), ssize_t(0));

  auto itr = queue.begin();
  auto level_end = queue.end();

  while (itr != queue.end()) {
    size_t v = *itr;
    visited[v] = true;

    for (size_t i = index[v]; i < index[v + 1]; ++i) {
      size_t n = edges[i];

      if (result[n].second == -1 || result[n].second > (result[v].second + 1)) {
	result[n] = std::make_pair(v, result[v].second + 1);
      }
      if (!visited[n]) {
        queue.push_back(n);
      }
    }

    if (++itr == level_end) {
      level_end = queue.end();
    }
  }

  return result;
}

void cuda_d2h(void *dst, void *src, size_t size,
              cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
  cuda_check(__FILE__, __LINE__);
}

void cuda_d2h(void *dst, void *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  cuda_check(__FILE__, __LINE__);
}

void cuda_h2d(void *dst, void *src, size_t size,
              cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
  cuda_check(__FILE__, __LINE__);
}

void cuda_h2d(void *dst, void *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  cuda_check(__FILE__, __LINE__);
}

void cuda_memset(void *dst, int val, size_t size, cudaStream_t s) {
  cudaMemsetAsync(dst, val, size, s);
  cuda_check(__FILE__, __LINE__);
}

void cuda_memset(void *dst, int val, size_t size) {
  cudaMemset(dst, val, size);
  cuda_check(__FILE__, __LINE__);
}

void cuda_sync(cudaStream_t stream) { cudaStreamSynchronize(stream); }

void cuda_enable_p2p(size_t dev_number) {
  cudaDeviceEnablePeerAccess(dev_number, 0);
}

void cuda_disable_p2p(size_t dev_number) {
  cudaDeviceDisablePeerAccess(dev_number);
}

size_t cuda_available_memory() {
  size_t total , free;
  cudaMemGetInfo(&free, &total);
  cuda_check(__FILE__, __LINE__);
  return free;
}

}  // namespace ripples
