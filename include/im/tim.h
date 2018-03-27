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

#ifndef IM_TIM_H
#define IM_TIM_H

#include <cmath>
#include <cstddef>
#include <set>
#include <random>

#include "im/bfs.h"

#include "boost/math/special_functions/binomial.hpp"

namespace im {

struct tim_tag {};

//! \brief TIM theta estimation function.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \param k The size of the seed set
template <typename GraphTy>
size_t thetaEstimation(GraphTy &G, size_t k, double epsilon) {
  // Compute KPT* according to Algorithm 2
  size_t KPTStar = 1;
  for (size_t i = 1; i < G.scale(); i <<= 1) {
    double c_i = 6 * log10(G.scale()) + 6 * log10(log2(G.scale())) * i;
    double sum = 0;

    {
      size_t end = c_i;
#pragma omp parallel for schedule(static) reduction(+:sum)
      for (size_t j = 0; j < end; ++j) {
        auto RRset = BFSOnRandomGraph(G);

        size_t WR = 0;
        for (auto vertex : RRset) {
          WR += G.in_degree(vertex);
        }
        // Equation (8) of the paper.
        double KR = 1 - pow(1 - WR / G.size(), k);
        sum += KR;
      }
    }

    if ((sum / c_i) > (1 / i)) {
      KPTStar = G.scale() * sum / (c_i * 2);
      break;
    }
  }

  // Try to refine the bound computing KPT' with Algorithm 3
  size_t KPTPrime = 1;

  // KPT+ = max{KPT*, KPT'}
  size_t KPTPlus = std::max(KPTStar, KPTPrime);

  // Compute lambda from equation (4)
  size_t l = 1;
  double lambda =
      (8 + 2 * epsilon) * G.scale() *
      (log10(l * G.scale()) +
       log10(boost::math::binomial_coefficient<double>(G.scale(), k)) +
       log10(2.0)) *
      pow(epsilon, -2);

  // Compute theta
  size_t theta = lambda / KPTPlus;
  return theta;
}

//! \brief Generate Random RR sets.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \param theta The number of random RR set to be generated
//! \return A set of theta random RR set.
template <typename GraphTy>
std::vector<std::set<typename GraphTy::vertex_type>> generateRandomRRSet(
    GraphTy &G, size_t theta) {
  std::vector<std::set<typename GraphTy::vertex_type>> result;

#pragma omp paralell for schedule(static)
  for (size_t i = 0; i < theta; ++i) {
    auto influenced_set = BFSOnRandomGraph(G);

#pragma omp critical
    result.emplace_back(std::move(influenced_set));
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
typename GraphTy::vertex_type GetMostInfluential(GraphTy &G, RRRSetList &R) {
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
#pragma omp single
    {
      for (auto v : G) {
#pragma omp task
        {
          MostInfluential lMI(v, 0);
          for (auto itr = R.begin(), end = R.end(); itr != end; ++itr) {
            if (itr->find(v) != itr->end()) ++lMI.count;
          }

#pragma omp critical
          MI = std::max(MI, lMI);
        }
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
                                 RRRSetList &R) {
  RRRSetList result;

#pragma omp parallel for schedule(static)
  for (auto itr = R.begin(); itr < R.end(); ++itr) {
    if (itr->find(v) != itr->end()) continue;

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
std::set<typename GraphTy::vertex_type> influence_maximization(
    GraphTy &G, size_t k, double epsilon, const tim_tag &&) {
  // Estimate the number of Random Reverse Reacheable Sets needed
  // Algorithm 2 in Tang Y. et all
  size_t theta = thetaEstimation(G, k, epsilon);

  // - Random Reverse Reacheable Set initialize to the empty set
  using RRRSet = std::set<typename GraphTy::vertex_type>;
  std::vector<RRRSet> R = generateRandomRRSet(G, theta);

  // - Initialize the seed set to the empty set
  std::set<typename GraphTy::vertex_type> seedSet;
  while (seedSet.size() < k && !R.empty()) {
    // 1 - Find the most influential vertex v
    typename GraphTy::vertex_type v = GetMostInfluential(G, R);

    // 2 - Add v to seedSet
    seedSet.insert(v);

    // 3 - Remove all the RRRSet that includes v
    R = std::move(ReduceRandomRRSetList<GraphTy>(v, R));
  }
  return seedSet;
}

}  // namespace im

#endif /* IM_TIM_H */
