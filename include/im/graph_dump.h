//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//


#ifndef IM_GRAPH_DUMP_H
#define IM_GRAPH_DUMP_H

#include <iostream>


namespace im {

template <typename GraphTy, typename OStream>
void dumpGraph(const GraphTy & G, OStream & OS) {
  for (auto i = 0; i < G.num_nodes(); ++i) {
    for (auto j : G.neighbors(i)) {
      OS << G.convertID(i) << '\t' << G.convertID(j.vertex)
         << '\t' << j.weight << std::endl;
    }
  }
}

}  // namespace im

#endif /* IM_GRAPH_DUMP_H */
