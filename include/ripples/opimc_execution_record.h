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

#ifndef RIPPLES_OPIMC_EXECUTION_RECORD_H
#define RIPPLES_OPIMC_EXECUTION_RECORD_H

#include <chrono>
#include <vector>

#include "ripples/find_most_influential_record.h"
#include "ripples/generate_rrr_sets_record.h"

namespace ripples {

//! OPIMC execution record.
struct OPIMCExecutionRecord : public GenerateRRRSetsRecord,
                              public FindMostInfluentialSetRecord {
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  using ex_time_ns = std::chrono::nanoseconds;

  OPIMCExecutionRecord()
      : GenerateRRRSetsRecord(),
        FindMostInfluentialSetRecord(),
        NumThreads(),
        Counting(),
        Pivoting(),
        GenerateRRRSets(),
        FindMostInfluentialSet(),
        Total(),
        RRRSetSize() {}

  //! Number of threads used during the execution.
  size_t NumThreads;
  //! Execution time of the Theta estimation phase.
  ex_time_ms ThetaEstimationTotal;
  std::vector<ex_time_ms> Counting;
  std::vector<ex_time_ms> Pivoting;
  //! Execution time of the RRR sets generation phase.
  ex_time_ms GenerateRRRSets;
  //! Execution time of the maximum coverage phase.
  ex_time_ms FindMostInfluentialSet;
  //! Total execution time.
  ex_time_ms Total;
  size_t RRRSetSize;
};

}  // namespace ripples

#endif  // RIPPLES_OPIMC_EXECUTION_RECORD_H
