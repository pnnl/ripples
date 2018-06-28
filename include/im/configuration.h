//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
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
  std::string IFileName{""};                //!< The input file name
  std::string LogFile{"log.log"};           //!< The file name of the log
  size_t k{10};                             //!< The size of the seedset
  double epsilon{0.15};                     //!< The epsilon of the IM algorithm
  bool parallel{false};
};

};  // namespace im

#endif  // IM_CONFIGURATION_H
