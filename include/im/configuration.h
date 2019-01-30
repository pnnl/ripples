//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CONFIGURATION_H
#define IM_CONFIGURATION_H

#include <string>

#include "CLI11/CLI11.hpp"

namespace im {

//! \brief The command line configuration
template <typename AlgorithmConfiguration>
struct ToolConfiguration : public AlgorithmConfiguration {
  std::string IFileName{""};       //!< The input file name
  std::string LogFile{"log.log"};  //!< The file name of the log
  bool weighted{false};
  bool undirected{false};
  bool OMPStrongScaling{false};
  bool reload{false};

  void ParseCmdOptions(int argc, char **argv) {
    CLI::App app;
    app.add_option("-i,--input-graph", IFileName,
                   "The input file with the edge-list.")
        ->required();
    app.add_flag("--reload-binary", reload,
                 "Reload a graph from binary input");
    app.add_flag("-u,--undirected", undirected,
                 "The input graph is undirected");
    app.add_flag("-w,--weighted", weighted, "The input graph is weighted");
    app.add_option("-l,--log", LogFile, "The file name of the log.");
    app.add_flag("--omp_strong_scaling", OMPStrongScaling,
                 "Trigger strong scaling experiments");

    AlgorithmConfiguration::addCmdOptions(app);

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      exit(app.exit(e));
    }
  }
};

};  // namespace im

#endif  // IM_CONFIGURATION_H
