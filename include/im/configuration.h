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

struct GraphInputConfiguration {
  std::string IFileName{""};       //!< The input file name
  bool weighted{false};
  bool undirected{false};
  bool reload{false};

  void addCmdOptions(CLI::App & app) {
    app.add_option("-i,--input-graph", IFileName,
                   "The input file with the edge-list.")
        ->group("Input Options")
        ->required();
    app.add_flag("--reload-binary", reload,
                 "Reload a graph from binary input")
        ->group("Input Options");
    app.add_flag("-u,--undirected", undirected,
                 "The input graph is undirected")
        ->group("Input Options");
    app.add_flag("-w,--weighted", weighted, "The input graph is weighted")
        ->group("Input Options");
  }
};


struct OutputConfiguration {
  std::string OutputFile{"output.json"};  //!< The file name of the log

  void addCmdOptions(CLI::App & app) {
    app.add_option("-o,--output", OutputFile, "The file name of the log.")
        ->group("Output Options");
  }
};


//! \brief The command line configuration
template <
  typename AlgorithmConfiguration,
  typename OutputConfiguration = OutputConfiguration,
  typename InputConfiguration = GraphInputConfiguration
  >
struct ToolConfiguration :
      public InputConfiguration,
      public AlgorithmConfiguration,
      public OutputConfiguration {

  void ParseCmdOptions(int argc, char **argv) {
    CLI::App app;

    InputConfiguration::addCmdOptions(app);
    AlgorithmConfiguration::addCmdOptions(app);
    OutputConfiguration::addCmdOptions(app);

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      exit(app.exit(e));
    }
  }
};

};  // namespace im

#endif  // IM_CONFIGURATION_H
