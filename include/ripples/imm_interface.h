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

#ifndef RIPPLES_IMM_INTERFACE_H
#define RIPPLES_IMM_INTERFACE_H

#include "ripples/imm_execution_record.h"

namespace ripples {

using dest_type = ripples::WeightedDestination<uint32_t, float>;
using GraphFwd =
  ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>>;
using GraphBwd =
  ripples::Graph<uint32_t, dest_type, ripples::BackwardDirection<uint32_t>>;

using LTStreamingGenerator =
	  ripples::StreamingRRRGenerator<
	    GraphBwd, trng::lcg64,
	  typename ripples::RRRsets<GraphBwd>::iterator,
	    ripples::linear_threshold_tag>;
using ICStreamingGenerator =
	  ripples::StreamingRRRGenerator<
	    GraphBwd, trng::lcg64,
	  typename ripples::RRRsets<GraphBwd>::iterator,
	    ripples::independent_cascade_tag>;

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
extern std::vector<typename GraphTy::vertex_type> IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
                diff_model_tag &&model_tag, sequential_tag &&ex_tag);

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
extern std::vector<typename GraphTy::vertex_type> IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
                diff_model_tag &&model_tag, omp_parallel_tag &&ex_tag);

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
extern std::vector<typename GraphTy::vertex_type> IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
                IMMExecutionRecord&, diff_model_tag &&model_tag, sequential_tag &&ex_tag);

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
extern std::vector<typename GraphTy::vertex_type> IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
                IMMExecutionRecord &, diff_model_tag &&model_tag, omp_parallel_tag &&ex_tag);
}

#endif
