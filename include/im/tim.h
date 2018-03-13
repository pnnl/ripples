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

#include <iostream>
#include <cmath>
#include <cstddef>
#include <set>
#include <random>

#include "boost/math/special_functions/binomial.hpp"

namespace im {

struct tim_tag {};

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \param G The graph instance.
//! \return A random RR set.
template <typename GraphTy>
std::set<typename GraphTy::vertex_type> BFSOnRandomGraph(GraphTy &G) {
  // This will be a slightly modified BFS.
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_int_distribution<size_t> distribution(0, G.scale() - 1);
  typename GraphTy::vertex_type root = distribution(generator);

  std::uniform_real_distribution<double> value(0.0, 1.0);

  std::vector<typename GraphTy::vertex_type> queue(G.scale());
  auto start = std::begin(queue);
  auto end = std::begin(queue) + 1;
  auto next = std::begin(queue) + 1;

  *start = root;
  std::set<typename GraphTy::vertex_type> result;
  result.insert(root);

  while (start != end) {
    for(auto itr = start; itr != end; ++itr) {
      auto vertex = *itr;
      for (auto neighbor : G.in_neighbors(vertex)) {
        if (result.find(neighbor.v) == result.end() &&
            value(generator) > neighbor.attribute) {
          *next = neighbor.v;
          result.insert(neighbor.v);
          ++next;
        }
      }
    }

    start = end;
    end = next;
  }

  return result;
}

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
    for (size_t j = 0; j < c_i; ++j) {
      auto RRset = BFSOnRandomGraph(G);

      size_t WR = 0;
      for (auto vertex : RRset) {
        WR += G.in_degree(vertex);
      }
      // Equation (8) of the paper.
      double KR = 1 - pow(1 - WR / G.size(), k);
      sum += KR;
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
  std::cout << "theta is " << theta << std::endl;
  for (size_t i = 0; i < theta; ++i) {
    auto influenced_set = BFSOnRandomGraph(G);
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
  typename GraphTy::vertex_type best_vertex;

  size_t best_count = 0;
  for (auto v : G) {
    size_t count = 0;
    for (auto itr = R.begin(), end = R.end(); itr != end; ++itr) {
      if (itr->find(v) != itr->end()) ++count;
    }

    if (count > best_count) {
      best_count = count;
      best_vertex = v;
    }
  }

  return best_vertex;
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
  for (auto itr = R.begin(), end = R.end(); itr != end; ++itr) {
    if (itr->find(v) != itr->end()) continue;
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
  std::cout << "theta Estimated >> "<< G.scale() << std::endl;

  // - Random Reverse Reacheable Set initialize to the empty set
  using RRRSet = std::set<typename GraphTy::vertex_type>;
  std::vector<RRRSet> R = generateRandomRRSet(G, theta);
  std::cout << "Reverse Reacheable Sets Generated >> " << G.scale() << std::endl;

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
