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

enum class Algorithm { TIM };

//! \brief The command line configuration
struct Configuration {
  std::string IFileName{""};                //!< The input file name
  size_t k{10};                             //!< The size of the seedset
  double epsilon{0.15};                     //!< The epsilon of the IM algorithm
  Algorithm algo{Algorithm::TIM};           //!< The algorithm to be used.
  std::default_random_engine generator{0};  //!< The random number generator.
};

};  // namespace im

#endif  // IM_CONFIGURATION_H
