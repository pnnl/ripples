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

#ifndef RIPPLES_CUDA_FIND_MOST_INFLUENTIAL_H
#define RIPPLES_CUDA_FIND_MOST_INFLUENTIAL_H

#include "ripples/cuda/cuda_utils.h"

#include <utility>
#include <cstddef>
#include <cstdint>

namespace ripples {

std::pair<uint32_t, size_t> CudaMaxElement(uint32_t * b, size_t N);

void
CudaUpdateCounters(cudaStream_t compute_stream,
                   size_t batch_size, uint32_t *d_rr_vertices,
                   uint32_t * d_rr_edges, uint32_t * d_mask,
                   uint32_t * d_Counters, size_t num_nodes,
                   uint32_t last_seed);

void
CudaUpdateCounters(size_t batch_size, uint32_t *d_rr_vertices,
                   uint32_t * d_rr_edges, uint32_t * d_mask,
                   uint32_t * d_Counters, size_t num_nodes,
                   uint32_t last_seed);


void CudaCountOccurrencies(
    uint32_t * d_Counters, uint32_t * d_rrr_sets,
    size_t rrr_sets_size, size_t num_nodes);

void CudaCountOccurrencies(
    uint32_t * d_Counters, uint32_t * d_rrr_sets,
    size_t rrr_sets_size, size_t num_nodes, cudaStream_t S);

void CudaReduceCounters(uint32_t * src, uint32_t * dest, size_t N);
void CudaReduceCounters(cudaStream_t S, uint32_t * src, uint32_t * dest, size_t N);

size_t CountZeros(char * d_rr_mask, size_t N);
size_t CountOnes(char * d_rr_mask, size_t N);

}  // namespace ripples

#endif /* RIPPLES_CUDA_FIND_MOST_INFLUENTIAL_H */
