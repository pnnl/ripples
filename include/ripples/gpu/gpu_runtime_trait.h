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

#ifndef RIPPLES_GPU_GPU_RUNTIME_TRAIT_H
#define RIPPLES_GPU_GPU_RUNTIME_TRAIT_H

#include <cstddef>
#include <limits>
#include <vector>

#if defined(RIPPLES_ENABLE_CUDA)
#include "ripples/cuda/cuda_utils.h"
#elif defined(RIPPLES_ENABLE_HIP)
#include "hip/hip_runtime.h"
#else
#error "Unsupported Runtime"
#endif

namespace ripples {
enum GPURuntime { CUDA, HIP };

#if defined(RIPPLES_ENABLE_CUDA)
#define RUNTIME ripples::GPURuntime::CUDA
#elif defined(RIPPLES_ENABLE_HIP)
#define RUNTIME ripples::GPURuntime::HIP
#endif

template <GPURuntime R>
class GPURuntimeTrait {
 public:
  using size_type = size_t;

  using device_id_type = size_t;
  using stream_type = size_t;

  static void set_device(const device_id_type &);
  static device_id_type num_devices();
  static size_type max_blocks();
  static size_type max_threads();

  static stream_type create_stream();
  static void destroy_stream(stream_type &);
  static void stream_sync(stream_type &);
  static void device_sync();

  static void device_malloc(void **, size_type);
  static void device_free(void *);
  static size_type available_memory();

  static void d2h(void *const, void *const, size_type);
  static void d2h(void *const, void *const, size_type, stream_type);

  static void h2d(void *const, void *const, size_type);
  static void h2d(void *const, void *const, size_type, stream_type);

  static void memset(void *const, int, size_type);
  static void memset(void *const, int, size_type, stream_type);

  static bool p2p_atomics(device_id_type, device_id_type);
  static void enable_p2p(device_id_type);
  static void disable_p2p(device_id_type);
};

#ifdef RIPPLES_ENABLE_CUDA

template <>
class GPURuntimeTrait<CUDA> {
 public:
  using size_type = size_t;
  using device_id_type = size_t;
  using stream_type = cudaStream_t;

  static void set_device(device_id_type ID) { cuda_set_device(ID); }
  static device_id_type num_devices() { return cuda_num_devices(); }
  static size_type max_blocks() { return cuda_max_blocks(); }

  static stream_type create_stream() {
    stream_type S;
    cuda_stream_create(&S);
    return S;
  }
  static void destroy_stream(stream_type &S) { cuda_stream_destroy(S); }
  static void stream_sync(stream_type &S) { cuda_sync(S); }
  static void device_sync() { cuda_device_sync(); }
  static void device_malloc(void **P, size_t size) { cuda_malloc(P, size); }
  static void device_free(void *P) { cuda_free(P); }
  static size_type available_memory() { return cuda_available_memory(); }
  static void d2h(void *const D, void *const S, size_type size) {
    cuda_d2h(D, S, size);
  }
  static void d2h(void *const D, void *const S, size_type size,
                  stream_type stream) {
    cuda_d2h(D, S, size, stream);
  }
  static void h2d(void *const D, void *const S, size_type size) {
    cuda_h2d(D, S, size);
  }
  static void h2d(void *const D, void *const S, size_type size,
                  stream_type stream) {
    cuda_h2d(D, S, size, stream);
  }

  static void memset(void *const ptr, int value, size_type size) {
    cuda_memset(ptr, value, size);
  }
  static void memset(void *const ptr, int value, size_type size,
                     stream_type S) {
    cuda_memset(ptr, value, size, S);
  }

  static bool p2p_atomics(device_id_type I, device_id_type J) {
    int atomics = 0;
    cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, I,
                              J);
    return atomics == 1;
  }

  static void enable_p2p(device_id_type D) { cudaDeviceEnablePeerAccess(D, 0); }
  static void disable_p2p(device_id_type D) { cudaDeviceDisablePeerAccess(D); }
};
#endif

#ifdef RIPPLES_ENABLE_HIP

template <>
class GPURuntimeTrait<HIP> {
 public:
  using size_type = size_t;
  using device_id_type = int;
  using stream_type = hipStream_t;

  static void set_device(device_id_type ID) {
    static_cast<void>(hipSetDevice(ID));
  }
  static device_id_type num_devices() {
    device_id_type res;
    static_cast<void>(hipGetDeviceCount(&res));
    return res;
  }
  static size_type max_blocks() {
    hipDeviceProp_t prop;
    static_cast<void>(hipGetDeviceProperties(&prop, 0));
    return prop.maxGridSize[0];
  }

  static size_type max_threads() {
    hipDeviceProp_t prop;
    static_cast<void>(hipGetDeviceProperties(&prop, 0));
    return prop.maxThreadsPerBlock;
  }

  static stream_type create_stream() {
    stream_type S;
    static_cast<void>(hipStreamCreate(&S));
    return S;
  }
  static void destroy_stream(stream_type &S) {
    static_cast<void>(hipStreamDestroy(S));
  }
  static void stream_sync(stream_type &S) {
    static_cast<void>(hipStreamSynchronize(S));
  }
  static void device_sync() { static_cast<void>(hipDeviceSynchronize()); }
  static void device_malloc(void **P, size_t size) {
    static_cast<void>(hipMalloc(P, size));
  }
  static void device_free(void *P) { static_cast<void>(hipFree(P)); }
  static size_t available_memory() {
    size_t total, free;
    static_cast<void>(hipMemGetInfo(&free, &total));
    return free;
  }
  static void d2h(void *const D, void *const S, size_type size) {
    static_cast<void>(hipMemcpy(D, S, size, hipMemcpyDeviceToHost));
  }
  static void d2h(void *const D, void *const S, size_type size,
                  stream_type stream) {
    static_cast<void>(
        hipMemcpyWithStream(D, S, size, hipMemcpyDeviceToHost, stream));
  }
  static void h2d(void *const D, void *const S, size_type size) {
    static_cast<void>(hipMemcpy(D, S, size, hipMemcpyHostToDevice));
  }
  static void h2d(void *const D, void *const S, size_type size,
                  stream_type stream) {
    static_cast<void>(
        hipMemcpyWithStream(D, S, size, hipMemcpyHostToDevice, stream));
  }
  static bool p2p_atomics(device_id_type I, device_id_type J) {
    int atomics = 0;
    static_cast<void>(hipDeviceGetP2PAttribute(
        &atomics, hipDevP2PAttrNativeAtomicSupported, I, J));
    return atomics == 1;
  }
  static void memset(void *const ptr, int value, size_type size) {
    static_cast<void>(hipMemset(ptr, value, size));
  }
  static void memset(void *const ptr, int value, size_type size,
                     stream_type S) {
    static_cast<void>(hipMemsetAsync(ptr, value, size, S));
  }
  static void enable_p2p(device_id_type D) {
    static_cast<void>(hipDeviceEnablePeerAccess(D, USE_PEER_NON_UNIFIED));
  }
  static void disable_p2p(device_id_type D) {
    static_cast<void>(hipDeviceDisablePeerAccess(D));
  }
};
#endif

template <GPURuntime R>
class GPU : public GPURuntimeTrait<R> {
 public:
  using device_id_type = typename GPURuntimeTrait<R>::device_id_type;
  using device_graph =
      std::pair<std::vector<device_id_type>, std::vector<device_id_type>>;
  using reduction_tree = std::vector<std::pair<device_id_type, device_id_type>>;

  static device_graph build_topology_graph() {
    size_t num_devices = GPU<R>::num_devices();

    typename device_graph::first_type index(num_devices + 1, 0);
    typename device_graph::second_type edges;

    for (size_t i = 0; i < num_devices; ++i) {
      index[i + 1] = index[i];
      for (size_t j = 0; j < num_devices; ++j) {
        if (i == j) continue;

        if (GPU<R>::p2p_atomics(j, i)) {
          index[i + 1] += 1;
          edges.push_back(j);
        }
      }
    }

    return std::make_pair(index, edges);
  }

  static reduction_tree build_reduction_tree() {
    auto topo = build_topology_graph();
    auto &index = topo.first;
    auto &edges = topo.second;

    size_t num_devices = GPU<R>::num_devices();
    std::vector<bool> visited(num_devices);
    // Predecessor and Level.
    reduction_tree result(
        num_devices,
        std::make_pair(0, std::numeric_limits<device_id_type>::max()));

    std::vector<device_id_type> queue;
    queue.reserve(edges.size());

    queue.push_back(0);
    result[0] = std::make_pair(device_id_type(0), device_id_type(0));

    auto itr = queue.begin();
    auto level_end = queue.end();

    while (itr != queue.end()) {
      auto v = *itr;
      visited[v] = true;

      for (auto i = index[v]; i < index[v + 1]; ++i) {
        auto n = edges[i];

        if (result[n].second == std::numeric_limits<device_id_type>::max() ||
            result[n].second > (result[v].second + 1)) {
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
};

}  // namespace ripples
#endif
