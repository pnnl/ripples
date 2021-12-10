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

#ifndef RIPPLES_IMM_CONFIGURATION_H
#define RIPPLES_IMM_CONDIGURATION_H

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>

#include "ripples/configuration.h"
#include "ripples/tim.h"

namespace ripples {

//! The IMM algorithm configuration descriptor.
struct IMMConfiguration : public TIMConfiguration {
  size_t streaming_workers{0};
  size_t streaming_gpu_workers{0};
  size_t seed_select_max_workers{std::numeric_limits<size_t>::max()};
  size_t seed_select_max_gpu_workers{0};
  std::string gpu_mapping_string{""};
  std::unordered_map<size_t, size_t> worker_to_gpu;

  //! \brief Add command line options to configure IMM.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    TIMConfiguration::addCmdOptions(app);
    app.add_option(
           "--streaming-gpu-workers", streaming_gpu_workers,
           "The number of GPU workers for the CPU+GPU streaming engine.")
        ->group("Streaming-Engine Options");
    app.add_option("--streaming-gpu-mapping", gpu_mapping_string,
                   "A comma-separated set of OpenMP numbers for GPU workers.")
        ->group("Streaming-Engine Options");
    app.add_option("--seed-select-max-workers", seed_select_max_workers,
                   "The max number of workers for seed selection.")
        ->group("Streaming-Engine Options");
    app.add_option("--seed-select-max-gpu-workers", seed_select_max_gpu_workers,
                   "The max number of GPU workers for seed selection.")
        ->group("Streaming-Engine Options");
  }
};

//! Retrieve the configuration parsed from command line.
//! \return the configuration parsed from command line.
ToolConfiguration<ripples::IMMConfiguration> configuration();

}

#endif
