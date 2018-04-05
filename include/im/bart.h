//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2017 Pacific Northwest National Laboratory
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
//===----------------------------------------------------------------------===//

#ifndef IM_BART_H
#define IM_BART_H

#include <random>

#include "im/tim.h"
#include "im/bloomfilter.h"

namespace im {

struct bart_tag {};

//! \brief Generate Random RR sets.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \param theta The number of random RR set to be generated
//! \return A set of theta random RR set.
template <typename GraphTy>
std::vector<im::bloomfilter<uint64_t>>
generateRandomRRSet(
    GraphTy &G, size_t theta, double p, const bart_tag &) {
  std::vector<im::bloomfilter<uint64_t>> result;

#pragma omp parallel
  {
    size_t size = omp_get_num_threads();
    size_t rank = omp_get_thread_num();

    std::default_random_engine generator;
    generator.discard(rank * (theta * G.size() / size));

    std::vector<im::bloomfilter<uint64_t>> intermediate_result;
    for (size_t i = rank * theta/size; i < (rank + 1) * theta/size; ++i) {
      auto influenced_set = BFSOnRandomGraph(G, generator, p);
      intermediate_result.emplace_back(std::move(influenced_set));
    }
#pragma omp critical
    result.insert(result.end(), intermediate_result.begin(), intermediate_result.end());
  }
  return result;
}

//! \brief Find the most influential node
//!
//! \tparam GraphTy The type of the graph.
//! \tparam RRRSetlist The type used to store the Random RR set.
//! \param G The graph instance.
//! \param R A collection of random RR sets.
//! \return The vertex appearing the most in R.
template <typename GraphTy, typename RRRSetList>
typename GraphTy::vertex_type GetMostInfluential(GraphTy &G, RRRSetList &R, const bart_tag &) {
  struct MostInfluential {
    MostInfluential(const typename GraphTy::vertex_type &v, size_t c)
        : vertex(v), count(c) {}

    bool operator<(const MostInfluential &rhs) const {
      return count < rhs.count;
    }

    typename GraphTy::vertex_type vertex;
    size_t count;
  };

  MostInfluential MI(0, 0);

// #pragma omp declare reduction \
//   (MIMax:MostInfluential:omp_out=std::max(omp_out, omp_in))     \
//   initializer(omp_priv=MostInfluential(0, 0))

#pragma omp parallel
  {
    for (auto v : G) {
#pragma omp single nowait
      {
        MostInfluential lMI(v, 0);
        for (auto itr = R.begin(), end = R.end(); itr != end; ++itr) {
          if (itr->find(v)) ++lMI.count;
        }

#pragma omp critical
        MI = std::max(MI, lMI);
      }
    }
  }
  return MI.vertex;
}

//! \brief Remove all the RR set containing a given vertex.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam RRRSetlist The type used to store the Random RR set.
//! \param v The vertex.
//! \param R A collection of random RR sets.
//! \return The updated list of RR sets.
template <typename GraphTy, typename RRRSetList>
RRRSetList ReduceRandomRRSetList(typename GraphTy::vertex_type v,
                                 RRRSetList &R, const bart_tag &) {
  RRRSetList result;

#pragma omp parallel for schedule(static)
  for (auto itr = R.begin(); itr < R.end(); ++itr) {
    if (itr->find(v)) continue;

#pragma omp critical
    result.emplace_back(std::move(*itr));
  }
  return result;
}

//! \brief The TIM influence maximization algorithm.
//!
//! \tparm GraphTy The type of the graph.
//!
//! \param G The instance of the graph.
//! \param k The size of the seed set.
template <typename GraphTy>
std::unordered_set<typename GraphTy::vertex_type> influence_maximization(
    GraphTy &G, size_t k, double epsilon, double p, const bart_tag & tag) {
  // Estimate the number of Random Reverse Reacheable Sets needed
  // Algorithm 2 in Tang Y. et all
  size_t theta = thetaEstimation(G, k, epsilon);

  // - Random Reverse Reacheable Set initialize to the empty set
  using RRRSet = im::bloomfilter<uint64_t>;
  std::vector<RRRSet> R = generateRandomRRSet(G, theta, p, tag);

  // - Initialize the seed set to the empty set
  std::unordered_set<typename GraphTy::vertex_type> seedSet;
  while (seedSet.size() < k && !R.empty()) {
    // 1 - Find the most influential vertex v
    typename GraphTy::vertex_type v = GetMostInfluential(G, R, tag);

    // 2 - Add v to seedSet
    seedSet.insert(v);

    // 3 - Remove all the RRRSet that includes v
    R = std::move(ReduceRandomRRSetList<GraphTy>(v, R, tag));
  }
  return seedSet;
}

}

#endif /* IM_BART_H */
