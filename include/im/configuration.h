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

#ifndef IM_CONFIGURATION_H
#define IM_CONFIGURATION_H

#include <cstddef>
#include <random>
#include <string>

namespace im {

//! \brief The command line configuration
struct Configuration {
  std::string IFileName;  //!< The input file name
  size_t k;               //!< The size of the seedset
  double epsilon;         //!< The epsilon of the IM algorithm

  std::default_random_engine generator;  //!< The random number generator.
};

extern Configuration CFG;

};  // namespace im

#endif  // IM_CONFIGURATION_H
