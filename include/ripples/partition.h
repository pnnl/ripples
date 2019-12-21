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

#ifndef RIPPLES_PARTITION_H
#define RIPPLES_PARTITION_H

#include <algorithm>
#include <iostream>

#include "ripples/utility.h"

namespace ripples {

//! Sequential swap ranges.
//!
//! \tparam ItrTy1 The iterator type of the first sequence.
//! \tparam ItrTy2 The iterator type of the second sequence.
//!
//! \param B The begin of the first sequence.
//! \param E The end of the second sequence.
//! \param O The begin of the second sequence.
//! \return The iterator to the one-past last element swapped.
template <typename ItrTy1, typename ItrTy2>
ItrTy2 swap_ranges(ItrTy1 B, ItrTy1 E, ItrTy2 O, sequential_tag) {
  return std::swap_ranges(B, E, O);
}

template <typename ItrTy1, typename ItrTy2>
ItrTy2 swap_ranges(ItrTy1 B, ItrTy1 E, ItrTy2 O, size_t num_threads) {
  size_t toBeSwaped = std::distance(B, E);
#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < toBeSwaped; ++i) {
    std::iter_swap(B + i, O + i);
  }
  return O + toBeSwaped;
}

//! Parallel swap ranges.
//!
//! \tparam ItrTy1 The iterator type of the first sequence.
//! \tparam ItrTy2 The iterator type of the second sequence.
//!
//! \param B The begin of the first sequence.
//! \param E The end of the second sequence.
//! \param O The begin of the second sequence.
//! \return The iterator to the one-past last element swapped.
template <typename ItrTy1, typename ItrTy2>
ItrTy2 swap_ranges(ItrTy1 B, ItrTy1 E, ItrTy2 O, omp_parallel_tag) {
  size_t num_threads(0);
#pragma omp single
  { num_threads = omp_get_max_threads(); }

  return swap_ranges(B, E, O, num_threads);
}

namespace {

template <typename ItrTy, typename ex_tag = omp_parallel_tag>
struct PartitionIndices {
  ItrTy begin;
  ItrTy end;
  ItrTy pivot;

  PartitionIndices() : begin(), end(), pivot() {}

  PartitionIndices(PartitionIndices &&O)
      : begin{std::move(O.begin)},
        end{std::move(O.end)},
        pivot{std::move(O.pivot)} {}

  PartitionIndices &operator=(PartitionIndices &&O) {
    this->begin = std::move(O.begin);
    this->end = std::move(O.end);
    this->pivot = std::move(O.pivot);
    return *this;
  }

  PartitionIndices(const PartitionIndices &O)
      : begin{O.begin}, end{O.end}, pivot{O.pivot} {}

  PartitionIndices &operator=(const PartitionIndices &O) {
    this->begin = O.begin;
    this->end = O.end;
    this->pivot = O.pivot;
    return *this;
  }

  PartitionIndices(ItrTy B, ItrTy E, ItrTy P) : begin{B}, end{E}, pivot{P} {}

  PartitionIndices(ItrTy B, ItrTy E) : PartitionIndices(B, E, E) {}

  bool operator==(const PartitionIndices &O) const {
    return this->begin == O.begin && this->end == O.end &&
           this->pivot == O.pivot;
  }

  PartitionIndices mergeBlocks(const PartitionIndices &O, size_t num_threads) {
    PartitionIndices result(*this);

    if (this->pivot == this->begin && O.pivot == O.begin) {
      result.end = O.end;
      return result;
    } else if (this->pivot == this->end) {
      result.end = O.end;
      result.pivot = O.pivot;
      return result;
    }

    if (std::distance(this->pivot, this->end) <
        std::distance(O.begin, O.pivot)) {
      size_t toBeMoved = std::distance(this->pivot, this->end);
      swap_ranges(this->pivot, this->end, std::prev(O.pivot, toBeMoved),
                  num_threads);
      result.pivot = std::prev(O.pivot, toBeMoved);
    } else {
      result.pivot = swap_ranges(O.begin, O.pivot, this->pivot, num_threads);
    }
    result.end = O.end;

    return result;
  }

  // PartitionIndices operator+(const PartitionIndices &O) {
  //   PartitionIndices result(*this);

  //   if (this->pivot == this->begin && O.pivot == O.begin) {
  //     result.end = O.end;
  //     return result;
  //   } else if (this->pivot == this->end) {
  //     result.end = O.end;
  //     result.pivot = O.pivot;
  //     return result;
  //   }

  //   if (std::distance(this->pivot, this->end) <
  //       std::distance(O.begin, O.pivot)) {
  //     size_t toBeMoved = std::distance(this->pivot, this->end);
  //     swap_ranges(this->pivot, this->end, std::prev(O.pivot, toBeMoved),
  //                 ex_tag{});
  //     result.pivot = std::prev(O.pivot, toBeMoved);
  //   } else {
  //     result.pivot = swap_ranges(O.begin, O.pivot, this->pivot, ex_tag{});
  //   }
  //   result.end = O.end;

  //   return result;
  // }
};

}  // namespace

template <typename ItrTy, typename UnaryPredicate>
ItrTy partition(ItrTy B, ItrTy E, UnaryPredicate P, size_t num_threads) {
  std::vector<PartitionIndices<ItrTy>> indices(num_threads,
                                               PartitionIndices<ItrTy>(B, E));

#pragma omp parallel num_threads(num_threads)
  {
    size_t num_elements = std::distance(B, E);
    size_t threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
    size_t low = num_elements * threadnum / numthreads,
           high = num_elements * (threadnum + 1) / numthreads;

    indices[threadnum].begin = B + low;
    indices[threadnum].end = std::min(E, B + high);
    indices[threadnum].pivot =
        std::partition(indices[threadnum].begin, indices[threadnum].end, P);
  }

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

  return indices[0].pivot;
}

//! Reorder a sequence in such a way that all the element for which a predicate
//! is true preceed the one for which the predicate is false.

//! \tparam ItrTy The type of the iterator of the input sequence.
//! \tparam UnaryPredicate The type of a unary predicate object.
//!
//! \param B The start of the sequence to be partitioned.
//! \param E The end of the sequence to be partitioned.
//! \param P A C++ collable object implementing the predicate.
//! \return An iterator to the first element for which the predicate is false.
template <typename ItrTy, typename UnaryPredicate>
ItrTy partition(ItrTy B, ItrTy E, UnaryPredicate P, omp_parallel_tag) {
  size_t num_threads(1);

#pragma omp single
  { num_threads = omp_get_max_threads(); }

  return partition(B, E, P, num_threads);
}

//! Reorder a sequence in such a way that all the element for which a predicate
//! is true preceed the one for which the predicate is false.
//!
//! \tparam ItrTy The type of the iterator of the input sequence.
//! \tparam UnaryPredicate The type of a unary predicate object.
//!
//! \param B The start of the sequence to be partitioned.
//! \param E The end of the sequence to be partitioned.
//! \param P A C++ collable object implementing the predicate.
//! \return An iterator to the first element for which the predicate is false.
template <typename ItrTy, typename UnaryPredicate>
ItrTy partition(ItrTy B, ItrTy E, UnaryPredicate P, sequential_tag) {
  return std::partition(B, E, P);
}

}  // namespace ripples

#endif /* RIPPLES_PARTITION_H */
