//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_UTILITY_H
#define IM_UTILITY_H

namespace im {

struct sequential_tag {};
struct omp_parallel_tag {};
struct cuda_sequential_tag {};
struct mpi_omp_parallel_tag {};
struct cxx_parallel_tag {};

}  // namespace im

#endif  // IM_UTILITY_H
