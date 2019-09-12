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

#ifndef RIPPLES_COUNTING_H
#define RIPPLES_COUNTING_H

#include <algorithm>
#include <iterator>

#include <omp.h>

#include "ripples/utility.h"

namespace ripples {

//! \brief Count the occurrencies of vertices in the RRR sets.
//!
//! \tparam InItr The input sequence iterator type.
//! \tparam OutItr The output sequence iterator type.
//!
//! \param in_begin The begin of the sequence of RRR sets.
//! \param in_end The end of the sequence of RRR sets.
//! \param out_begin The begin of the sequence storing the counters for each
//! vertex.
//! \param out_end The end of the sequence storing the counters for each vertex.
template <typename InItr, typename OutItr>
void CountOccurrencies(InItr in_begin, InItr in_end, OutItr out_begin,
                       OutItr out_end, sequential_tag &&) {
  using rrr_set_type = typename std::iterator_traits<InItr>::value_type;
  using vertex_type = typename rrr_set_type::value_type;
  for (; in_begin != in_end; ++in_begin) {
    std::for_each(in_begin->begin(), in_begin->end(),
                  [&](const vertex_type v) { *(out_begin + v) += 1; });
  }
}


template <typename InItr, typename OutItr>
void CountOccurrencies(InItr in_begin, InItr in_end, OutItr out_begin,
                       OutItr out_end, size_t num_threads) {
  using rrr_set_type = typename std::iterator_traits<InItr>::value_type;
  using vertex_type = typename rrr_set_type::value_type;

#pragma omp parallel num_threads(num_threads)
  {
    size_t num_elements = std::distance(out_begin, out_end);
    size_t threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
    vertex_type low = num_elements * threadnum / numthreads,
                high = num_elements * (threadnum + 1) / numthreads;

    for (auto itr = in_begin; itr != in_end; ++itr) {
      auto begin = std::lower_bound(itr->begin(), itr->end(), low);
      auto end = std::upper_bound(begin, itr->end(), high - 1);
      std::for_each(begin, end,
                    [&](const vertex_type v) { *(out_begin + v) += 1; });
    }
  }
}


//! \brief Count the occurrencies of vertices in the RRR sets.
//!
//! \tparam InItr The input sequence iterator type.
//! \tparam OutItr The output sequence iterator type.
//!
//! \param in_begin The begin of the sequence of RRR sets.
//! \param in_end The end of the sequence of RRR sets.
//! \param out_begin The begin of the sequence storing the counters for each
//! vertex.
//! \param out_end The end of the sequence storing the counters for each vertex.
template <typename InItr, typename OutItr>
void CountOccurrencies(InItr in_begin, InItr in_end, OutItr out_begin,
                       OutItr out_end, omp_parallel_tag &&) {
  size_t num_threads(1);

#pragma omp single
  { num_threads = omp_get_max_threads(); }

  CountOccurrencies(in_begin, in_end, out_begin, out_end, num_threads);
}


//! \brief Update the coverage counters.
//!
//! \tparam RRRsetsItrTy The iterator type of the sequence of RRR sets.
//! \tparam VertexCoverageVectorTy The type of the vector storing counters.
//!
//! \param B The start sequence of RRRsets covered by the just selected seed.
//! \param E The start sequence of RRRsets covered by the just selected seed.
//! \param vertexCoverage The vector storing the counters to be updated.
template <typename RRRsetsItrTy, typename VertexCoverageVectorTy>
void UpdateCounters(RRRsetsItrTy B, RRRsetsItrTy E,
                    VertexCoverageVectorTy &vertexCoverage, sequential_tag &&) {
  for (; B != E; ++B) {
    for (auto v : *B) {
      vertexCoverage[v] -= 1;
    }
  }
}


template <typename RRRsetsItrTy, typename VertexCoverageVectorTy>
void UpdateCounters(RRRsetsItrTy B, RRRsetsItrTy E,
                    VertexCoverageVectorTy &vertexCoverage,
                    size_t num_threads) {
  for (; B != E; ++B) {
#pragma omp parallel for num_threads(num_threads)
    for (size_t j = 0; j < (*B).size(); ++j) {
      vertexCoverage[(*B)[j]] -= 1;
    }
  }
}


//! \brief Update the coverage counters.
//!
//! \tparam RRRsetsItrTy The iterator type of the sequence of RRR sets.
//! \tparam VertexCoverageVectorTy The type of the vector storing counters.
//!
//! \param B The start sequence of RRRsets covered by the just selected seed.
//! \param E The start sequence of RRRsets covered by the just selected seed.
//! \param vertexCoverage The vector storing the counters to be updated.
template <typename RRRsetsItrTy, typename VertexCoverageVectorTy>
void UpdateCounters(RRRsetsItrTy B, RRRsetsItrTy E,
                    VertexCoverageVectorTy &vertexCoverage,
                    omp_parallel_tag &&) {
  size_t num_threads(1);

#pragma omp single
  { num_threads = omp_get_max_threads(); }

  UpdateCounters(B, E, vertexCoverage, num_threads);
}


//! \brief Initialize the Heap storage.
//!
//! \tparam InItr The input sequence iterator type.
//! \tparam OutItr The output sequence iterator type.
//!
//! \param in_begin The begin of the sequence of vertex counters.
//! \param in_end The end of the sequence of vertex counters.
//! \param out_begin The begin of the sequence used as storage in the Heap.
//! \param out_end The end of the sequence used as storage in the Heap.
template <typename InItr, typename OutItr>
void InitHeapStorage(InItr in_begin, InItr in_end, OutItr out_begin,
                     OutItr out_end, sequential_tag &&) {
  using value_type = typename std::iterator_traits<OutItr>::value_type;
  using vertex_type = typename value_type::first_type;

  for (vertex_type v = 0; in_begin != in_end; ++in_begin, ++v, ++out_begin) {
    *out_begin = {v, *in_begin};
  }
}


template <typename InItr, typename OutItr>
void InitHeapStorage(InItr in_begin, InItr in_end, OutItr out_begin,
                     OutItr out_end, size_t num_threads) {
  using value_type = typename std::iterator_traits<OutItr>::value_type;
  using vertex_type = typename value_type::first_type;

#pragma omp parallel for num_threads(num_threads)
  for (vertex_type v = 0; v < std::distance(in_begin, in_end); ++v) {
    *(out_begin + v) = {v, *(in_begin + v)};
  }
}


//! \brief Initialize the Heap storage.
//!
//! \tparam InItr The input sequence iterator type.
//! \tparam OutItr The output sequence iterator type.
//!
//! \param in_begin The begin of the sequence of vertex counters.
//! \param in_end The end of the sequence of vertex counters.
//! \param out_begin The begin of the sequence used as storage in the Heap.
//! \param out_end The end of the sequence used as storage in the Heap.
template <typename InItr, typename OutItr>
void InitHeapStorage(InItr in_begin, InItr in_end, OutItr out_begin,
                     OutItr out_end, omp_parallel_tag &&) {
  size_t num_threads(1);

#pragma omp single
  { num_threads = omp_get_max_threads(); }

  InitHeapStorage(in_begin, in_end, out_begin, out_end, num_threads);
}

}  // namespace ripplse

#endif /* RIPPLES_COUNTING_H */
