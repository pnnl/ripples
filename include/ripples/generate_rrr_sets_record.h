//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024, Battelle Memorial Institute
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

#ifndef RIPPLES_GENERATE_RRR_SETS_RECORD_H
#define RIPPLES_GENERATE_RRR_SETS_RECORD_H

#include <chrono>
#include <vector>

namespace ripples {

//! Generate RRR sets Record.
struct GenerateRRRSetsRecord {
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  using ex_time_ns = std::chrono::nanoseconds;

  struct cpu_walk_prof {
    size_t NumSets;
    ex_time_ms Total;

    cpu_walk_prof() : NumSets(), Total() {}
  };

  struct gpu_walk_prof {
    size_t NumSets;
    ex_time_ms Total;
    ex_time_ns Kernel, D2H, Post;

    gpu_walk_prof() : NumSets(), Total(), Kernel(), D2H(), Post() {}
  };

  struct walk_iteration_prof {
    std::vector<cpu_walk_prof> CPUWalks{};
    std::vector<gpu_walk_prof> GPUWalks{};
    size_t NumSets{0};
    ex_time_ms Total{0};

    walk_iteration_prof() : CPUWalks(), GPUWalks(), NumSets(), Total() {}
  };

  GenerateRRRSetsRecord()
      : CPUBatchSize(64),
        GPUBatchSize(64),
        WalkIterations(),
        Microbenchmarking() {}

  //! CPU Batch Size
  size_t CPUBatchSize;
  //! GPU Batch Size
  size_t GPUBatchSize;
  //! Iterations breakdown
  std::vector<walk_iteration_prof> WalkIterations;
  //! Total microbenchmarking time.
  ex_time_ms Microbenchmarking;
};

}  // namespace ripples

#endif
