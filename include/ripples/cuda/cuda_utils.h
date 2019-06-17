//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_UTILS_H
#define IM_CUDA_CUDA_UTILS_H

#include "ripples/graph.h"

namespace ripples {

//
// debug utilities
//
template <typename graph_t>
void print_graph(const graph_t &g) {
  printf("*** graph BEGIN ***\n");
  for (typename graph_t::vertex_type i = 0; i < g.num_nodes(); ++i) {
    printf("%d\t:", i);
    for (auto &n : g.neighbors(i)) printf("%d\t", n.vertex);
    printf("\n");
  }
  printf("*** graph END ***\n");
}

void cuda_check(cudaError_t err, const char *fname, int line) {
  if (err != cudaSuccess) {
    CUDA_LOG("> CUDA error @%s:%d: name=%s msg='%s'\n", fname, line,
             cudaGetErrorName(err), cudaGetErrorString(err));
    assert(false);
  }
}

void cuda_check(const char *fname, int line) {
  cuda_check(cudaGetLastError(), fname, line);
}
}  // namespace ripples

#endif  // IM_CUDA_CUDA_UTILS_H
