//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_COMMON_CUH
#define IM_CUDA_CUDA_COMMON_CUH

#include "im/graph.h"

template<typename GraphTy>
struct cuda_graph_t {
	using vertex_type = typename GraphTy::vertex_type;
	using edge_weight_type = typename GraphTy::edge_weight_type;
};

template<typename GraphTy>
cuda_graph_t<GraphTy> *cuda_h2d_graph(GraphTy &G) {
	cuda_graph_t<GraphTy> *res;
	cudaMalloc(&res, sizeof(cuda_graph_t<GraphTy> ));
	return res;
}

template<typename GraphTy>
void cuda_destroy_graph(cuda_graph_t<GraphTy> *G) {
	cudaFree(G);
}

#endif  // IM_CUDA_CUDA_COMMON_CUH
