//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_INFLUENCE_MAXIMIZATION_H
#define IM_INFLUENCE_MAXIMIZATION_H

#include <set>
#include "im/graph.h"

namespace im {

//! \brief The influence maximization algorithm
//!
//! The implementation uses a method_tag to select the implementation of the
//! method to be used.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam method_tag The type tag selecting the implementation.
//!
//! \param G The social network.
//! \param k The size of the seed set for the influence maximization.
template <typename GraphTy, typename method_tag>
std::set<typename GraphTy::vertex_type>
influence_maximization(
    const GraphTy &G, size_t k, double epsilon, const method_tag &);

}  // namespae im

#include "im/tim.h"
#include "im/bart.h"

#endif /* IM_INFLUENCE_MAXIMIZATION_H */
