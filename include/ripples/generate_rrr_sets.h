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

#ifndef RIPPLES_GENERATE_RRR_SETS_H
#define RIPPLES_GENERATE_RRR_SETS_H

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "omp.h"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/imm_execution_record.h"
#include "ripples/utility.h"
#include "ripples/rrr_sets.h"
#include "ripples/add_rrrset.h"
#include "ripples/streaming_rrr_generator.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

namespace ripples {

//! \brief Generate Random Reverse Reachability Sets - sequential.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam ItrTy A random access iterator type.
//! \tparam ExecRecordTy The type of the execution record
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param generator The random numeber generator.
//! \param begin The start of the sequence where to store RRR sets.
//! \param end The end of the sequence where to store RRR sets.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag>
void GenerateRRRSets(GraphTy &G, PRNGeneratorTy &generator,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&model_tag,
                     sequential_tag &&ex_tag) {
  trng::uniform_int_dist start(0, G.num_nodes());

  for (auto itr = begin; itr < end; ++itr) {
    typename GraphTy::vertex_type r = start(generator[0]);
    AddRRRSet(G, r, generator[0], *itr,
              std::forward<diff_model_tag>(model_tag));
  }
}

//! \brief Generate Random Reverse Reachability Sets - CUDA.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam ItrTy A random access iterator type.
//! \tparam ExecRecordTy The type of the execution record
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param generator The random numeber generator.
//! \param begin The start of the sequence where to store RRR sets.
//! \param end The end of the sequence where to store RRR sets.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag>
void GenerateRRRSets(const GraphTy &G,
                     StreamingRRRGenerator<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag> &se,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&,
                     omp_parallel_tag &&) {
  se.generate(begin, end);
}

}  // namespace ripples

#endif  // RIPPLES_GENERATE_RRR_SETS_H
