//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_UTILITY_H
#define IM_UTILITY_H

#include <chrono>
#include <utility>

#include "nlohmann/json.hpp"

namespace im {

//! Execution tag for the sequential algorithms.
struct sequential_tag {};
//! Execution tag for the OpenMP parallel algorithms.
struct omp_parallel_tag {};
//! Execution tag for the CUDA parallel algorithms.
struct cuda_parallel_tag {};
//! Execution tag for the MPI+OpenMP parallel algorithms.
struct mpi_omp_parallel_tag {};

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
  static auto exec_time(F&& function, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<decltype(function)>(function)(std::forward<Args>(args)...);
    return std::chrono::high_resolution_clock::now() - start;
  }
};

}  // namespace im

namespace nlohmann {

template <typename T1, typename T2>
void to_json(nlohmann::json &j, const std::chrono::duration<T1, T2> & d) {
  j = d.count();
}

}

#endif  // IM_UTILITY_H
