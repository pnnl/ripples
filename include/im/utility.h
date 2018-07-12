//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_UTILITY_H
#define IM_UTILITY_H

#include <chrono>

namespace im {

struct sequential_tag {};
struct omp_parallel_tag {};
struct cxx_parallel_tag {};

struct ExecutionRecord {
  size_t NumThreads;
  std::chrono::duration<double, std::milli> KptEstimation;
  std::chrono::duration<double, std::milli> KptRefinement;
  std::chrono::duration<double, std::milli> GenerateRRRSets;
  std::chrono::duration<double, std::milli> FindMostInfluentialSet;
  std::chrono::duration<double, std::milli> Total;

  template <typename Ostream>
  friend Ostream & operator<<(Ostream &O, const ExecutionRecord &R) {
    O << "{ "
      << "\"NumThreads\" : " << R.NumThreads << ", "
      << "\"KptEstimation\" : " << R.KptEstimation.count() << ", "
      << "\"KptRefinement\" : " << R.KptRefinement.count() << ", "
      << "\"GenerateRRRSets\" : " << R.GenerateRRRSets.count() << ", "
      << "\"FindMostInfluentialSet\" : " << R.FindMostInfluentialSet.count() << ", "
      << "\"Total\" : " << R.Total.count()
      << " }";
    return O;
  }
};

}  // namespace im

#endif  // IM_UTILITY_H
