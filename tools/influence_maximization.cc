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

#include <iostream>
#include <string>

#include "boost/program_options.hpp"

namespace im {

//! \brief The command line configuration
struct Configuration {
  std::string IFileName;  //!< The input file name
};

Configuration ParseCmdOptions(int argc, char **argv) {
  Configuration CFG;

  namespace po = boost::program_options;

  po::options_description description("Options");
  description.add_options()("help,h", "Print this help message")(
      "input-graph,i", po::value<std::string>(&CFG.IFileName)->required(),
      "The input file with the edge-list.");

  po::variables_map VM;
  try {
    po::store(po::parse_command_line(argc, argv, description), VM);

    if (VM.count("help")) {
      std::cout << argv[0] << " [options]" << std::endl;
      std::cout << description << std::endl;
      exit(0);
    }

    po::notify(VM);
  } catch (po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << description << std::endl;
    exit(-1);
  }

  return CFG;
}

}  // namespace im

int main(int argc, char **argv) {
  im::Configuration CFG = im::ParseCmdOptions(argc, argv);
  return 0;
}
