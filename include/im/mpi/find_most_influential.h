//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_MPI_FIND_MOST_INFLUENTIAL_H
#define IM_MPI_FIND_MOST_INFLUENTIAL_H

#include "im/find_most_influential.h"
#include "im/utility.h"

namespace im {

//! \brief Select k seeds starting from the a list of Random Reverse
//! Reachability Sets.
//!
//! \tparam GraphTy The graph type.
//! \tparam RRRset The type storing Random Reverse Reachability Sets.
//! \tparam execution_tag The execution policy.
//!
//! \param G The input graph.
//! \param k The size of the seed set.
//! \param RRRsets A vector of Random Reverse Reachability sets.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.
template <typename GraphTy, typename RRRset>
auto FindMostInfluentialSet(const GraphTy &G, size_t k,
                            const std::vector<RRRset> &RRRsets,
                            mpi_omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;

  std::vector<typename GraphTy::vertex_type> result;
  result.reserve(k);

  size_t uncovered = RRRsets.size();

  return std::make_pair(RRRsets.size() - uncovered, result);
}

}  // namespace im

#endif  // IM_MPI_FIND_MOST_INFLUENTIAL_H
