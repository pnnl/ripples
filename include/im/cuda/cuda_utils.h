//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_UTILS_H
#define IM_CUDA_CUDA_UTILS_H

#include "im/graph.h"

namespace im {

#define CUDA_DBG 0
#define CUDA_CHECK 0

#if CUDA_DBG
#define CUDA_LOG(...) printf(__VA_ARGS__)
#else
#define CUDA_LOG(...)
#endif

//
// check utilities
//
template <typename graph_t>
bool reaches(const graph_t &g, typename graph_t::vertex_type src,
             typename graph_t::vertex_type dst) {
  assert(src != dst);
  for (auto &n : g.neighbors(src))
    if (n.vertex == dst) return true;
  return false;
}

template <typename rrr_t, typename graph_t>
bool check_lt_from(const rrr_t &r,
                   const typename rrr_t::const_iterator &root_it,
                   const graph_t &g) {
  if (r.size() == 1) return root_it == r.begin();
  auto wr = r;
  auto root = *root_it;
  wr.erase(wr.begin() + std::distance(r.begin(), root_it));
  for (auto it = wr.begin(); it != wr.end(); ++it)
    if (reaches(g, root, *it) && check_lt_from(wr, it, g)) return true;
  return false;
}

#if CUDA_CHECK
template <typename rrr_t, typename graph_t>
void check_lt(const rrr_t &r, const graph_t &g, size_t id) {
  printf("> checking set %d: ", id);
  for (auto &v : r) printf("%d ", v);
  printf("\n");

  bool res = false;
  for (auto it = r.begin(); it != r.end(); ++it) {
    if (check_lt_from(r, it, g)) {
      res = true;
      break;
    }
  }

  if (!res) {
    printf("> check FAILED\n");
    exit(1);
  }
}
#else
template <typename... Args>
void check_lt(Args &&...) {}
#endif

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
}  // namespace im

#endif  // IM_CUDA_CUDA_UTILS_H
