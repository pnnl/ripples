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

//! \brief Descriptor for the input Graph
//!
//! The GraphInputConfiguration stores the command-line input describing the
//! input graphs.
struct GraphInputConfiguration {
  std::string IFileName{""};       //!< The input file name
  bool weighted{false};            //!< is Graph weighted?
  bool undirected{false};          //!< is Graph undirected?
  bool reload{false};              //!< are we reloading a binary dump?

  //! \brief Add command line options for the input grah.
  //!
  //! \parm app The command-line parser object.
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


//! \brief Descriptor for the output of the tool.
//!
//! The OutputConfiguration stores the command-line input describing the output
//! of the tool.
struct OutputConfiguration {
  std::string OutputFile{"output.json"};  //!< The file name of the log

  //! \brief Add command line options for the output of the tool.
  //!
  //! \parm app The command-line parser object.
  void addCmdOptions(CLI::App & app) {
    app.add_option("-o,--output", OutputFile, "The file name of the log.")
        ->group("Output Options");
  }
};


//! \brief Command-line configuration descriptor
//!
//! The class describes the input coming from the command-line interface.
//!
//! \tparam AlgorithmConfiguration Configuration descriptor of the algorithm to
//! run.
//! \tparam OutputConfiguration Configuration descriptor for the output of the
//! tool.
//! \tparm InputConfiguration Configuration descriptor for the input of the
//! tool.
template <
  typename AlgorithmConfiguration,
  typename OutputConfiguration = OutputConfiguration,
  typename InputConfiguration = GraphInputConfiguration
  >
struct ToolConfiguration :
      public InputConfiguration,
      public AlgorithmConfiguration,
      public OutputConfiguration {

  //! \brief Parse command-line options.
  //!
  //! \param argc Length of the command-line options vector.
  //! \param argv Command-line options vector.
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
