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

#endif /* IM_INFLUENCE_MAXIMIZATION_H */
