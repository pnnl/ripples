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

#ifndef RIPPLES_UTILITY_H
#define RIPPLES_UTILITY_H

#include <chrono>
#include <utility>
#include <memory>

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#include <machine/endian.h>

#define le64toh(x) OSSwapLittleToHostInt64(x)
#define htole64(x) OSSwapHostToLittleInt64(x)

#define le32toh(x) OSSwapLittleToHostInt32(x)
#define htole32(x) OSSwapHostToLittleInt32(x)
#else
#include <endian.h>
#endif

#ifndef __CUDACC__
#include "nlohmann/json.hpp"
#endif

namespace ripples {

//! Execution tag for the sequential algorithms.
struct sequential_tag {};
//! Execution tag for the OpenMP parallel algorithms.
struct omp_parallel_tag {};
//! Execution tag for the MPI+OpenMP parallel algorithms.
struct mpi_omp_parallel_tag {};
//! Execution tag for the CUDA parallel algorithms.
struct cuda_parallel_tag {};

//! Utility for measurements.
template <typename TimeTy = std::chrono::milliseconds>
struct measure {
  //! The type representing the time unit.
  using time_unit_type = TimeTy;

  //! Compute the execution time of a function.
  //!
  //! \tparam F The type of the callable object.
  //! \tparam Args The args of the arguments to be forwarded to F
  //!
  //! \param function The collable object to be executed.
  //! \param args The list of arguments to be forwarded to function.
  //! \return the execution time of function(args...)
  template <typename F, typename... Args>
  static auto exec_time(F &&function, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<decltype(function)>(function)(std::forward<Args>(args)...);
    return std::chrono::high_resolution_clock::now() - start;
  }
};

}  // namespace ripples

#ifndef __CUDACC__
namespace nlohmann {

template <typename T1, typename T2>
void to_json(nlohmann::json &j, const std::chrono::duration<T1, T2> &d) {
  j = d.count();
}

}  // namespace nlohmann
#endif

namespace ripples {

//! Dump a value of size N in binary format.
template <size_t N>
struct dump_v;

//! Dump a value of 4 bytes in binary format (little-endian).
template <>
struct dump_v<sizeof(uint32_t)> {
  //! Dump 4 bytes in little-endian format.
  //! \param v The input value
  //! \return v in little-endian format.
  static uint32_t value(uint32_t v) { return htole32(v); }
};

//! Dump a value of 8 bytes in binary format (little-endian).
template <>
struct dump_v<sizeof(uint64_t)> {
  //! Dump 8 bytes in little-endian format.
  //! \param v The input value
  //! \return v in little-endian format.
  static uint64_t value(uint64_t v) { return htole64(v); }
};

//! Load values in little-endian format into the host format.
template <size_t N>
struct load_v;

//! Load 4 bytes in little-endian format into the host format.
template <>
struct load_v<sizeof(uint32_t)> {
  //! Dump 4 bytes in little-endian format.
  //! \param v The input value
  //! \return v in little-endian format.
  static uint32_t value(uint32_t v) { return le32toh(v); }
};

//! Load 8 bytes in little-endian format into the host format.
template <>
struct load_v<sizeof(uint64_t)> {
  //! Dump 8 bytes in little-endian format.
  //! \param v The input value
  //! \return v in little-endian format.
  static uint64_t value(uint64_t v) { return le64toh(v); }
};

//! Dump and load sequences of types into/from little-endian format to the host
//! format.
template <typename T>
struct sequence_of : public dump_v<sizeof(T)> {
  //! The type of the value to be used.
  using value_type =
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

  //! Dump values into little-endian binary format.
  //!
  //! \tparam FStream The type of the output stream.
  //! \tparam Itr The type of the Iterators.
  //!
  //! \param FS The binary output stream to write to.
  //! \param B The begin of the sequence to dump.
  //! \param E The end of the sequence to dump.
  template <typename FStream, typename Itr>
  static void dump(FStream &FS, Itr B, Itr E) {
    std::vector<value_type> tmp(std::distance(B, E));
    std::transform(B, E, tmp.begin(), [](const T &V) -> value_type {
      const value_type v = *reinterpret_cast<const value_type *>(&V);
      return dump_v<sizeof(value_type)>::value(v);
    });

    FS.write(reinterpret_cast<char *>(tmp.data()),
             sizeof(value_type) * tmp.size());
  }

  //! Load a binary sequence in little-endian format into an output sequence in
  //! host format.
  //!
  //! \tparam Itr The type of the iterator in input.
  //! \tparam OItr The type of the iterator in output.
  //!
  //! \param B The begin of the sequence to load.
  //! \param E The end of the sequence to load.
  //! \param O the begin of the output sequence.
  template <typename Itr, typename OItr>
  static void load(Itr B, Itr E, OItr O) {
    std::transform(B, E, O, [](const T &V) -> T {
      value_type v = *reinterpret_cast<const value_type *>(&V);
      T tmp;
      *reinterpret_cast<value_type *>(&tmp) = load_v<sizeof(T)>::value(v);
      return tmp;
    });
  }
};

//! Obtain the same allocator type for allocating T from Alloc.
template <typename Alloc, typename T>
using rebind_alloc
  = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;

//! Obtain the same pointer-like type that points to T from
// the pointer-like member type in Alloc.
template <typename Alloc, typename T>
using rebind_alloc_pointer =
    typename std::pointer_traits<
        typename std::allocator_traits<Alloc>::pointer>::template rebind<T>;

}  // namespace ripples

#endif  // RIPPLES_UTILITY_H
