//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_IMM_H
#define IM_IMM_H

#include <cmath>
#include <cstddef>
#include <vector>

#ifdef HAVE_MPI
#include "mpi.h"
#endif

#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "im/tim.h"
#include "im/utility.h"

namespace im {

struct IMMConfiguration : public TIMConfiguration {};

struct IMMExecutionRecord {
  size_t NumThreads;
  size_t Theta;
  std::chrono::duration<double, std::milli> ThetaEstimation;
  std::chrono::duration<double, std::milli> GenerateRRRSets;
  std::chrono::duration<double, std::milli> FindMostInfluentialSet;
  std::chrono::duration<double, std::milli> Total;
};


inline double logBinomial(size_t n, size_t k) {
  return n * log(n) - k * log(k) - (n - k) * log(n - k);
}


template <typename execution_tag>
ssize_t ThetaPrime(ssize_t x, double epsilonPrime, double l,
                   size_t k, size_t num_nodes, execution_tag &&) {
  return (2 + 2. / 3. * epsilonPrime) *
      (l * std::log(num_nodes) + logBinomial(num_nodes, k) +
       std::log(std::log2(num_nodes))) *
      std::pow(2.0, x) / (epsilonPrime * epsilonPrime);
}


#ifdef HAVE_MPI
inline size_t ThetaPrime(ssize_t x, double epsilonPrime, double l,
                         size_t k, size_t num_nodes, mpi_omp_parallel_tag &&) {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  return ThetaPrime(x, epsilonPrime, l, k, num_nodes, omp_parallel_tag{})
      / world_size;
}
#endif


inline size_t Theta(double epsilon, double l, size_t k, double LB,
                    size_t num_nodes) {
  double term1 = 0.6321205588285577;  // 1 - 1/e
  double alpha = sqrt(l * std::log(num_nodes) + std::log(2));
  double beta = sqrt(term1 * (logBinomial(num_nodes, k) +
                              l * std::log(num_nodes) + std::log(2)));
  double lamdaStar = 2 * num_nodes * (term1 * alpha + beta) *
                     (term1 * alpha + beta) * pow(epsilon, -2);

  return lamdaStar / LB;
}


template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag,
          typename execution_tag>
auto Sampling(const GraphTy &G, std::size_t k, double epsilon, double l,
              PRNGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  std::vector<RRRset<GraphTy>> RR;

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));

    auto deltaRR = GenerateRRRSets(G, thetaPrime - RR.size(), generator,
                                   std::forward<diff_model_tag>(model_tag),
                                   std::forward<execution_tag>(ex_tag));

    RR.insert(RR.end(), std::make_move_iterator(deltaRR.begin()),
              std::make_move_iterator(deltaRR.end()));

    const auto &S =
        FindMostInfluentialSet(G, k, RR, std::forward<execution_tag>(ex_tag));
    double f = double(S.first) / RR.size();

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimation = end - start;

  record.Theta = theta;

  start = std::chrono::high_resolution_clock::now();
  if (theta > RR.size()) {
    auto deltaRR = GenerateRRRSets(G, theta - RR.size(), generator,
                                   std::forward<diff_model_tag>(model_tag),
                                   std::forward<execution_tag>(ex_tag));

    RR.insert(RR.end(), std::make_move_iterator(deltaRR.begin()),
              std::make_move_iterator(deltaRR.end()));
  }
  end = std::chrono::high_resolution_clock::now();

  record.GenerateRRRSets = end - start;

  return RR;
}

template <typename GraphTy, typename diff_model_tag, typename PRNG,
          typename execution_tag>
auto IMM(const GraphTy &G, std::size_t k, double epsilon, double l, PRNG &gen,
         diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  IMMExecutionRecord record;

  size_t max_num_threads(1);

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp single
    max_num_threads = omp_get_max_threads();
  }

  std::vector<trng::lcg64> generator(max_num_threads, gen);

  if (std::is_same<execution_tag, omp_parallel_tag>::value) {
#pragma omp parallel
    {
      generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                            omp_get_thread_num());
    }
  }

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  const auto &R = Sampling(G, k, epsilon, l, generator, record,
                           std::forward<diff_model_tag>(model_tag),
                           std::forward<execution_tag>(ex_tag));

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S =
      FindMostInfluentialSet(G, k, R, std::forward<execution_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  return std::make_pair(S.second, record);
}

#ifdef HAVE_MPI
template <typename GraphTy, typename diff_model_tag, typename PRNG>
auto IMM(const GraphTy &G, std::size_t k, double epsilon, double l, PRNG &gen,
         diff_model_tag &&model_tag, im::mpi_omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  IMMExecutionRecord record;

  size_t max_num_threads(1);

#pragma omp single
  max_num_threads = omp_get_max_threads();

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  gen.split(world_size, world_rank);

  std::vector<trng::lcg64> generator(max_num_threads, gen);

#pragma omp parallel
  {
    generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                          omp_get_thread_num());
  }

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  const auto &R = Sampling(G, k, epsilon, l, generator, record,
                           std::forward<diff_model_tag>(model_tag),
                           std::forward<execution_tag>(ex_tag));

  auto start = std::chrono::high_resolution_clock::now();
  const auto &S =
      FindMostInfluentialSet(G, k, R, std::forward<execution_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  return std::make_pair(S.second, record);
}
#endif

}  // namespace im

#endif /* IM_IMM_H */
