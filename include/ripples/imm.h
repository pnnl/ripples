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

#ifndef RIPPLES_IMM_H
#define RIPPLES_IMM_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/imm_execution_record.h"
#include "ripples/tim.h"
#include "ripples/utility.h"

#include "ripples/streaming_rrr_generator.h"

#define CUDA_PROFILE 0

namespace ripples {

//! Compute ThetaPrime.
//!
//! \tparam execution_tag The execution policy
//!
//! \param x The index of the current iteration.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
template <typename execution_tag>
ssize_t ThetaPrime(ssize_t x, double epsilonPrime, double l, size_t k,
                   size_t num_nodes, execution_tag &&) {
  k = std::min(k, num_nodes/2);
  return (2 + 2. / 3. * epsilonPrime) *
         (l * std::log(num_nodes) + logBinomial(num_nodes, k) +
          std::log(std::log2(num_nodes))) *
         std::pow(2.0, x) / (epsilonPrime * epsilonPrime);
}

//! Compute InvThetaPrime.
//!
//! \tparam execution_tag The execution policy
//!
//! \param rrsize The size of the RR set.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
template <typename execution_tag>
ssize_t InvThetaPrime(ssize_t rrsize, double epsilonPrime, double l, size_t k,
                   size_t num_nodes, execution_tag &&) {
  // Given the RR seed set, compute the index of the iteration
  k = std::min(k, num_nodes/2);
  return std::log2(rrsize * epsilonPrime * epsilonPrime /
                   ((2 + 2. / 3. * epsilonPrime) *
                    (l * std::log(num_nodes) + logBinomial(num_nodes, k) +
                     std::log(std::log2(num_nodes)))));
}

//! Compute Theta.
//!
//! \param epsilon Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param LB The estimate of the lower bound.
//! \param num_nodes The number of nodes in the input graph.
inline size_t Theta(double epsilon, double l, size_t k, double LB,
                    size_t num_nodes) {
  if (LB == 0) return 0;

  k = std::min(k, num_nodes/2);
  double term1 = 0.6321205588285577;  // 1 - 1/e
  double alpha = sqrt(l * std::log(num_nodes) + std::log(2));
  double beta = sqrt(term1 * (logBinomial(num_nodes, k) +
                              l * std::log(num_nodes) + std::log(2)));
  double lamdaStar = 2 * num_nodes * (term1 * alpha + beta) *
                     (term1 * alpha + beta) * pow(epsilon, -2);

  // std::cout << "#### " << lamdaStar << " / " << LB << " = " << lamdaStar / LB << std::endl;
  return lamdaStar / LB;
}

//! Collect a set of Random Reverse Reachable set.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam RRRGeneratorTy The type of the RRR generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param generator The rrr sets generator.
//! \param record Data structure storing timing and event counts.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename RRRSetsTy, typename RRRSetAllocatorTy,
          typename diff_model_tag, typename execution_tag,
          typename MapPtrTy = std::unordered_map<ssize_t, double> *>
void Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, RRRSetsTy &RR,
              RRRSetAllocatorTy &allocator,
              IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag,
              MapPtrTy fMap_ptr = nullptr) {
  using vertex_type = typename GraphTy::vertex_type;
  
  #ifdef ENABLE_METALL_RRRSETS
  assert(fMap_ptr != nullptr);
  metall::container::unordered_map<ssize_t, double> &fMap(*fMap_ptr);
  #endif
  
  
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  auto start = std::chrono::high_resolution_clock::now();
  ssize_t thetaPrimePrev = 0;
  ssize_t x = 1;
  if (RR.size() != 0){
    x = InvThetaPrime(RR.size()/2, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));
    // std::cout << "x_calc = " << x << std::endl;
  }
  for (; x < std::log2(G.num_nodes()); ++x) {
    // std::cout << "x = " << x << std::endl;
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));
    double f = 0;
    auto RRend = RR.end();
    auto timeRRRSets = measure<>::exec_time([&]() {
    if (thetaPrime <= RR.size()){
      #ifdef ENABLE_METALL_RRRSETS
      if(auto search = fMap.find(RR.size()); search != fMap.end()){
        f = search->second;
      }
      #endif
      thetaPrime = RR.size();
      record.ThetaPrimeDeltas.push_back(0);
    }
    else{
      size_t delta = thetaPrime - RR.size();
      spdlog::get("console")->info("Generating Sets: {}", delta);
      record.ThetaPrimeDeltas.push_back(delta);
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));
      auto begin = RR.end() - delta;
      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag));
      RRend = RR.end();
      }
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    auto RRbegin = RR.begin();
    // std::cout << "Top-k distance = " << std::distance(RRbegin, RRend) << std::endl;

    spdlog::get("console")->info("Finding top-k Seeds: {}", thetaPrime);

    auto timeMostInfluential = measure<>::exec_time([&]() {
      if (f == 0){
        const auto &S =
            FindMostInfluentialSet(G, CFG, RRbegin, RRend, record, generator.isGpuEnabled(),
                                  std::forward<execution_tag>(ex_tag));

        f = S.first;
      }
    });

    // spdlog::get("console")->info("F = {}", f);

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      // std::cout << "Fraction " << f << std::endl;
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;
  spdlog::get("console")->info("Theta {}", theta);

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      spdlog::get("console")->info("Generating Final Sets: {}", final_delta);
      RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag));
    }
  });

  spdlog::get("console")->info("Done Sampling, Saving RR Sets and Returning");

  // return RR;
}

//! Collect a set of Random Reverse Reachable set.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam RRRGeneratorTy The type of the RRR generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param generator The rrr sets generator.
//! \param record Data structure storing timing and event counts.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename diff_model_tag, typename execution_tag>
auto Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  #if defined ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(CFG.rr_dir.c_str(), 0);
  RRRsets<GraphTy> RR;
  #elif defined ENABLE_METALL_RRRSETS
  bool exists = metall::manager::consistent(CFG.rr_dir.c_str());
  metall::manager manager =
      (exists ? metall::manager(metall::open_only, CFG.rr_dir.c_str())
              : metall::manager(metall::create_only, CFG.rr_dir.c_str()));
  // RRRsetAllocator<vertex_type> 
  RRRsetsAllocator<GraphTy> allocator =  manager.get_allocator();
  RRRsets<GraphTy> *RR_ptr;

  metall::container::unordered_map<ssize_t, double> *fMap_ptr;

  if (exists) {
    // std::cout << "Metall: " << CFG.rr_dir << " exists. Reloading." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} exists. Reloading...", CFG.rr_dir);
    RR_ptr = manager.find<RRRsets<GraphTy>>("rrrsets").first;
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.find<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str()).first;
    spdlog::get("console")->info("Reloading complete!");
  } else {
    // std::cout << "Metall: " << CFG.rr_dir << " does not exist. Creating New." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} does not exist. Creating...", CFG.rr_dir);
    RR_ptr = manager.construct<RRRsets<GraphTy>>("rrrsets")(RRRsets<GraphTy>(allocator));
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.construct<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str())(manager.get_allocator());
    spdlog::get("console")->info("Creation Complete!!");
  }
  RRRsets<GraphTy> &RR(*RR_ptr);
  metall::container::unordered_map<ssize_t, double> &fMap(*fMap_ptr);
  #else
  RRRsetAllocator<vertex_type> allocator;
  RRRsets<GraphTy> RR;
  #endif

  auto start = std::chrono::high_resolution_clock::now(); 
  ssize_t thetaPrimePrev = 0;
  ssize_t x = 1;
  if (RR.size() != 0){
    x = InvThetaPrime(RR.size()/2, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));
    // std::cout << "x_calc = " << x << std::endl;
  }
  for (; x < std::log2(G.num_nodes()); ++x) {
    // std::cout << "x = " << x << std::endl;
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));
    double f = 0;
    auto RRend = RR.end();
    auto timeRRRSets = measure<>::exec_time([&]() {
    if (thetaPrime <= RR.size()){
      #ifdef ENABLE_METALL_RRRSETS
      if(auto search = fMap.find(RR.size()); search != fMap.end()){
        f = search->second;
      }
      #endif
      thetaPrime = RR.size();
      record.ThetaPrimeDeltas.push_back(0);
    }
    else{
      size_t delta = thetaPrime - RR.size();
      spdlog::get("console")->info("Generating Sets: {}", delta);
      record.ThetaPrimeDeltas.push_back(delta);
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));
      auto begin = RR.end() - delta;
      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag));
      RRend = RR.end();
      }
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    auto RRbegin = RR.begin();
    // std::cout << "Top-k distance = " << std::distance(RRbegin, RRend) << std::endl;

    spdlog::get("console")->info("Finding top-k Seeds: {}", thetaPrime);

    auto timeMostInfluential = measure<>::exec_time([&]() {
      if (f == 0){
        const auto &S =
            FindMostInfluentialSet(G, CFG, RRbegin, RRend, record, generator.isGpuEnabled(),
                                  std::forward<execution_tag>(ex_tag));

        f = S.first;
      }
    });

    // spdlog::get("console")->info("F = {}", f);

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      // std::cout << "Fraction " << f << std::endl;
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;
  spdlog::get("console")->info("Theta {}", theta);

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      spdlog::get("console")->info("Generating Final Sets: {}", final_delta);
      RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag));
    }
  });

  spdlog::get("console")->info("Done Sampling, Saving RR Sets and Returning");

  return RR;
}

template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename RRRSetsTy, typename RRRSetAllocatorTy,
          typename diff_model_tag,
          typename MapPtrTy = std::unordered_map<ssize_t, double> *>
void Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, RRRSetsTy &RR,
              RRRSetAllocatorTy &allocator,
              IMMExecutionRecord &record,
              diff_model_tag &&model_tag, sequential_tag &&ex_tag,
              MapPtrTy fMap_ptr = nullptr) {
  using vertex_type = typename GraphTy::vertex_type;

  #ifdef ENABLE_METALL_RRRSETS
  metall::container::unordered_map<ssize_t, double> &fMap(*fMap_ptr);
  #endif

  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  auto start = std::chrono::high_resolution_clock::now();
  ssize_t thetaPrimePrev = 0;
  ssize_t x = 1;
  if (RR.size() != 0){
    x = InvThetaPrime(RR.size()/2, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<sequential_tag>(ex_tag));
    // std::cout << "x_calc = " << x << std::endl;
  }
  for (; x < std::log2(G.num_nodes()); ++x) {
    // std::cout << "x = " << x << std::endl;
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<sequential_tag>(ex_tag));
    double f = 0;
    auto RRend = RR.end();
    auto timeRRRSets = measure<>::exec_time([&]() {
    if (thetaPrime <= RR.size()){
      #ifdef ENABLE_METALL_RRRSETS
      if(auto search = fMap.find(RR.size()); search != fMap.end()){
        f = search->second;
      }
      #endif
      thetaPrime = RR.size();
      record.ThetaPrimeDeltas.push_back(0);
    }
    else{
      size_t delta = thetaPrime - RR.size();
      spdlog::get("console")->info("Generating Sets: {}", delta);
      record.ThetaPrimeDeltas.push_back(delta);
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));
      auto begin = RR.end() - delta;
      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<sequential_tag>(ex_tag));
      RRend = RR.end();
      }
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    auto RRbegin = RR.begin();
    // std::cout << "Top-k distance = " << std::distance(RRbegin, RRend) << std::endl;

      auto timeMostInfluential = measure<>::exec_time([&]() {
      if (f == 0){
        const auto &S = FindMostInfluentialSet(
            G, CFG, RRbegin, RRend, record, false, std::forward<sequential_tag>(ex_tag));

        f = S.first;
      }
    });

    // spdlog::get("console")->info("F = {}", f);

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<sequential_tag>(ex_tag));
    }
  });

  // return RR;
}

template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename diff_model_tag>
auto Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, sequential_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  #if defined ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(CFG.rr_dir.c_str(), 0);
  RRRsets<GraphTy> RR;
  #elif defined ENABLE_METALL_RRRSETS
  bool exists = metall::manager::consistent(CFG.rr_dir.c_str());
  metall::manager manager =
      (exists ? metall::manager(metall::open_only, CFG.rr_dir.c_str())
              : metall::manager(metall::create_only, CFG.rr_dir.c_str()));
  // RRRsetAllocator<vertex_type> 
  RRRsetsAllocator<GraphTy> allocator =  manager.get_allocator();
  RRRsets<GraphTy> *RR_ptr;

  metall::container::unordered_map<ssize_t, double> *fMap_ptr;

  if (exists) {
    // std::cout << "Metall: " << CFG.rr_dir << " exists. Reloading." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} exists. Reloading...", CFG.rr_dir);
    RR_ptr = manager.find<RRRsets<GraphTy>>("rrrsets").first;
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.find<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str()).first;
    spdlog::get("console")->info("Reloading complete!");
  } else {
    // std::cout << "Metall: " << CFG.rr_dir << " does not exist. Creating New." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} does not exist. Creating...", CFG.rr_dir);
    RR_ptr = manager.construct<RRRsets<GraphTy>>("rrrsets")(RRRsets<GraphTy>(allocator));
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.construct<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str())(manager.get_allocator());
    spdlog::get("console")->info("Creation Complete!!");
  }
  RRRsets<GraphTy> &RR(*RR_ptr);
  metall::container::unordered_map<ssize_t, double> &fMap(*fMap_ptr);
  #else
  RRRsetAllocator<vertex_type> allocator;
  RRRsets<GraphTy> RR;
  #endif

  auto start = std::chrono::high_resolution_clock::now();
  ssize_t thetaPrimePrev = 0;
  ssize_t x = 1;
  if (RR.size() != 0){
    x = InvThetaPrime(RR.size()/2, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<sequential_tag>(ex_tag));
    // std::cout << "x_calc = " << x << std::endl;
  }
  for (; x < std::log2(G.num_nodes()); ++x) {
    // std::cout << "x = " << x << std::endl;
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<sequential_tag>(ex_tag));
    double f = 0;
    auto RRend = RR.end();
    auto timeRRRSets = measure<>::exec_time([&]() {
    if (thetaPrime <= RR.size()){
      #ifdef ENABLE_METALL_RRRSETS
      if(auto search = fMap.find(RR.size()); search != fMap.end()){
        f = search->second;
      }
      #endif
      thetaPrime = RR.size();
      record.ThetaPrimeDeltas.push_back(0);
    }
    else{
      size_t delta = thetaPrime - RR.size();
      spdlog::get("console")->info("Generating Sets: {}", delta);
      record.ThetaPrimeDeltas.push_back(delta);
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));
      auto begin = RR.end() - delta;
      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<sequential_tag>(ex_tag));
      RRend = RR.end();
      }
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    auto RRbegin = RR.begin();
    // std::cout << "Top-k distance = " << std::distance(RRbegin, RRend) << std::endl;

      auto timeMostInfluential = measure<>::exec_time([&]() {
      if (f == 0){
        const auto &S = FindMostInfluentialSet(
            G, CFG, RRbegin, RRend, record, false, std::forward<sequential_tag>(ex_tag));

        f = S.first;
      }
    });

    // spdlog::get("console")->info("F = {}", f);

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<sequential_tag>(ex_tag));
    }
  });

  return RR;
}

//! The IMM algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ConfTy The configuration type.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param CFG The configuration.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename PRNG,
          typename diff_model_tag>
std::vector<typename GraphTy::vertex_type> IMM(const GraphTy &G, const ConfTy &CFG, double l, PRNG &gen,
         IMMExecutionRecord &record, diff_model_tag &&model_tag,
         sequential_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  // Initialize RR set & allocator
  #if defined ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(CFG.rr_dir.c_str(), 0);
  RRRsets<GraphTy> RR;
  std::unordered_map<ssize_t, double> *fMap_ptr = nullptr;
  #elif defined ENABLE_METALL_RRRSETS
  bool exists = metall::manager::consistent(CFG.rr_dir.c_str());
  metall::manager manager =
      (exists ? metall::manager(metall::open_only, CFG.rr_dir.c_str())
              : metall::manager(metall::create_only, CFG.rr_dir.c_str()));
  // RRRsetAllocator<vertex_type> 
  RRRsetsAllocator<GraphTy> allocator =  manager.get_allocator();
  RRRsets<GraphTy> *RR_ptr;

  metall::container::unordered_map<ssize_t, double> *fMap_ptr;

  if (exists) {
    // std::cout << "Metall: " << CFG.rr_dir << " exists. Reloading." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} exists. Reloading...", CFG.rr_dir);
    RR_ptr = manager.find<RRRsets<GraphTy>>("rrrsets").first;
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.find<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str()).first;
    spdlog::get("console")->info("Reloading complete!");
  } else {
    // std::cout << "Metall: " << CFG.rr_dir << " does not exist. Creating New." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} does not exist. Creating...", CFG.rr_dir);
    RR_ptr = manager.construct<RRRsets<GraphTy>>("rrrsets")(RRRsets<GraphTy>(allocator));
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.construct<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str())(manager.get_allocator());
    spdlog::get("console")->info("Creation Complete!!");
  }
  RRRsets<GraphTy> &RR(*RR_ptr);
  #else
  RRRsetAllocator<vertex_type> allocator;
  RRRsets<GraphTy> RR;
  std::unordered_map<ssize_t, double> *fMap_ptr = nullptr;
  #endif

  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  std::vector<trng::lcg64> generator(1, gen);

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  Sampling(G, CFG, l, generator, RR, allocator, record,
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<sequential_tag>(ex_tag),
                    fMap_ptr);

#if CUDA_PROFILE
  auto logst = spdlog::stdout_color_st("IMM-profile");
  std::vector<size_t> rrr_sizes;
  for (auto &rrr_set : RR) rrr_sizes.push_back(rrr_set.size());
  print_profile_counter(logst, rrr_sizes, "RRR sizes");
#endif

  spdlog::get("console")->info("Finding Final Most Influential Sets");

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S = FindMostInfluentialSet(G, CFG, RR.begin(), RR.begin() + record.Theta, record, false,
                                         std::forward<sequential_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  spdlog::get("console")->info("Found Most Influential Sets!");

  #ifdef ENABLE_METALL_RRRSETS
    assert(fMap_ptr != nullptr);
    metall::container::unordered_map<ssize_t, double> &fMap(*fMap_ptr);
    fMap[RR.size()] = S.first;
  #endif
  return S.second;
}

//! The IMM algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ConfTy The configuration type
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param CFG The configuration.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
std::vector<typename GraphTy::vertex_type> IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
                                               IMMExecutionRecord& record, diff_model_tag &&model_tag, omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  // Initialize RR set & allocator
  #if defined ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(CFG.rr_dir.c_str(), 0);
  RRRsets<GraphTy> RR;
  std::unordered_map<ssize_t, double> *fMap_ptr = nullptr;
  #elif defined ENABLE_METALL_RRRSETS
  bool exists = metall::manager::consistent(CFG.rr_dir.c_str());
  metall::manager manager =
      (exists ? metall::manager(metall::open_only, CFG.rr_dir.c_str())
              : metall::manager(metall::create_only, CFG.rr_dir.c_str()));
  // RRRsetAllocator<vertex_type> 
  RRRsetsAllocator<GraphTy> allocator =  manager.get_allocator();
  RRRsets<GraphTy> *RR_ptr;

  metall::container::unordered_map<ssize_t, double> *fMap_ptr;

  if (exists) {
    // std::cout << "Metall: " << CFG.rr_dir << " exists. Reloading." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} exists. Reloading...", CFG.rr_dir);
    RR_ptr = manager.find<RRRsets<GraphTy>>("rrrsets").first;
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.find<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str()).first;
    spdlog::get("console")->info("Reloading complete!");
  } else {
    // std::cout << "Metall: " << CFG.rr_dir << " does not exist. Creating New." << std::endl;
    spdlog::get("console")->info("RR Datastore at {} does not exist. Creating...", CFG.rr_dir);
    RR_ptr = manager.construct<RRRsets<GraphTy>>("rrrsets")(RRRsets<GraphTy>(allocator));
    auto fMap_str = std::string("fMap_") + std::to_string(CFG.k);
    fMap_ptr = manager.construct<metall::container::unordered_map<ssize_t, double>>(fMap_str.c_str())(manager.get_allocator());
    spdlog::get("console")->info("Creation Complete!!");
  }
  RRRsets<GraphTy> &RR(*RR_ptr);
  #else
  RRRsetAllocator<vertex_type> allocator;
  RRRsets<GraphTy> RR;
  std::unordered_map<ssize_t, double> *fMap_ptr = nullptr;
  #endif

  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  Sampling(G, CFG, l, gen, RR, allocator, record, std::forward<diff_model_tag>(model_tag),
               std::forward<omp_parallel_tag>(ex_tag), fMap_ptr);

  spdlog::get("console")->info("Finding Final Most Influential Sets");

#if CUDA_PROFILE
  auto logst = spdlog::stdout_color_st("IMM-profile");
  std::vector<size_t> rrr_sizes;
  size_t sizeBytes = 0;
  for (auto &rrr_set : RR) {
    rrr_sizes.push_back(rrr_set.size());
    sizeBytes += rrr_set.size() * sizeof(rrr_set[0]);
  }
  record.RRRSetSize = sizeBytes;
  print_profile_counter(logst, rrr_sizes, "RRR sizes");
#endif

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S =
      FindMostInfluentialSet(G, CFG, RR.begin(), RR.begin() + record.Theta, record, gen.isGpuEnabled(),
                             std::forward<omp_parallel_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  spdlog::get("console")->info("Found Most Influential Sets!");

  start = std::chrono::high_resolution_clock::now();
  size_t total_size = 0;
#pragma omp parallel for reduction(+:total_size)
  for (size_t i = 0; i < RR.size(); ++i) {
    total_size += RR[i].size() * sizeof(vertex_type);
  }
  record.RRRSetSize = total_size;
  end = std::chrono::high_resolution_clock::now();
  record.Total = end - start;

  #ifdef ENABLE_METALL_RRRSETS
    assert(fMap_ptr != nullptr);
    metall::container::unordered_map<ssize_t, double> &fMap(*fMap_ptr);
    fMap[RR.size()] = S.first;
  #endif
  return S.second;
}

}  // namespace ripples

#endif  // RIPPLES_IMM_H
