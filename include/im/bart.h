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

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \return A random RR set.
template <
  typename GraphTy,
  typename PRNGeneratorTy,
  typename VisitedSetTy = std::unordered_set<typename GraphTy::vertex_type>
  >
std::vector<uint64_t> BlockBFSOnRandomGraph(GraphTy &G, PRNGeneratorTy &generator) {
  // This will be a slightly modified BFS.
  std::uniform_int_distribution<size_t> distribution(0, G.scale() - 1);
  std::uniform_real_distribution<double> value(0.0, 1.0);

  std::vector<uint64_t> result(G.scale(), 0);

  for (size_t j = 0; j < 64; ++j) {
    typename GraphTy::vertex_type root = distribution(generator);
    std::vector<typename GraphTy::vertex_type> queue(G.scale());

    auto start = std::begin(queue);
    auto end = std::begin(queue) + 1;
    auto next = std::begin(queue) + 1;

    for (auto & v : result)
      v <<= 1;

    *start = root;
    VisitedSetTy visited;
    visited.insert(root);

    while (start != end) {
      for(auto itr = start; itr != end; ++itr) {
        auto vertex = *itr;
        for (auto neighbor : G.in_neighbors(vertex)) {
          if (visited.find(neighbor.v) == visited.end() &&
              value(generator) <= neighbor.attribute) {
            *next = neighbor.v;
            visited.insert(neighbor.v);
            result[neighbor.v] |= 1;
            ++next;
          }
        }
      }

      start = end;
      end = next;
    }
  }

  return result;
}


//! \brief Generate Random RR sets.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \param theta The number of random RR set to be generated
//! \return A set of theta random RR set.
template <typename GraphTy>
std::vector<std::vector<uint64_t>>
generateRandomRRSet(
    GraphTy &G, size_t theta, double p, const bart_tag &) {
  std::vector<std::vector<uint64_t>> result(G.scale(), std::vector<uint64_t>(theta/64, 0));

#pragma omp parallel
  {
    size_t size = omp_get_num_threads();
    size_t rank = omp_get_thread_num();

    size_t blocks = theta / 64;

    std::default_random_engine generator;
    generator.discard(rank * (theta * G.size() / size));

    for (size_t i = rank * blocks/size; i < (rank + 1) * blocks/size; ++i) {
      auto block_result = BlockBFSOnRandomGraph(G, generator);

      for (size_t j = 0; j < G.scale(); ++j)
        result[j][i] = block_result[j];
    }
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
typename GraphTy::vertex_type GetMostInfluential(GraphTy &G, RRRSetList &R, std::vector<uint64_t> &mask, const bart_tag &) {
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

#pragma omp parallel
  {
    MostInfluential lMI(0,0);

#pragma omp for schedule(static)
    for (size_t i = 0; i < R.size(); ++i) {
      size_t count = 0;
      for (size_t j = 0; j < R[i].size(); ++j) {
        size_t v = mask[j] ^ R[i][j];
        count += __builtin_popcount(v);
      }

      if (count == 0) R[i].resize(0);

      lMI = std::max(lMI, MostInfluential(i, count));
    }

#pragma omp critical
    MI = std::max(MI, lMI);
  }

  return MI.vertex;
}

template <typename GraphTy, typename RRRSetList>
bool UpdateMask(typename GraphTy::vertex_type v,
                RRRSetList &R, std::vector<uint64_t> &mask, const bart_tag &) {
  size_t used = 0;
#pragma omp parallel for schedule(static) reduction(+:used)
  for (size_t i = 0; i < mask.size(); ++i) {
    mask[i] |= R[v][i];
    used += __builtin_popcount(mask[i]);
  }

  return used == (mask.size() * 64);
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

  auto roundTheta =
      [](size_t theta) -> uint64_t { 
        constexpr size_t mask = 63;
        theta += ((theta & mask) ^ mask) + 1;
        return theta;
      };
  
  // make theta a multiple of 64
  theta = roundTheta(theta);

  std::cout << "Theta : " << theta << std::endl;

  // - Random Reverse Reacheable Set initialize to the empty set
  using RRRSet = std::vector<uint64_t>;
  std::vector<uint64_t> mask(theta, 0);
  std::vector<RRRSet> R = std::move(generateRandomRRSet(G, theta, p, tag));

  std::cout << "Generated RRR" << std::endl;

  // - Initialize the seed set to the empty set
  std::unordered_set<typename GraphTy::vertex_type> seedSet;

  bool emptyRRR = false;
  while (seedSet.size() < k && !emptyRRR) {
    // 1 - Find the most influential vertex v
    typename GraphTy::vertex_type v = GetMostInfluential(G, R, mask, tag);

    // 2 - Add v to seedSet
    seedSet.insert(v);

    // 3 - Remove all the RRRSet that includes v
    emptyRRR = UpdateMask<GraphTy, std::vector<RRRSet>>(v, R, mask, tag);
  }

  return seedSet;
}

}

#endif /* IM_BART_H */
