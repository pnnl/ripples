//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_CUDA_CUDA_COMMON_CUH
#define IM_CUDA_CUDA_COMMON_CUH

#include "im/graph.h"

template<typename host_GraphTy>
struct cuda_graph_t {
	using vertex_type = typename host_GraphTy::vertex_type;
	using edge_weight_type = typename host_GraphTy::edge_weight_type;
	struct destination_type {
		vertex_type vertex;
		edge_weight_type weight;
	};

	cuda_graph_t(const host_GraphTy &hg) {
		// prepare edges
		cudaMalloc(&d_edges_, hg.num_edges() * sizeof(destination_type));

		// prepare index
		cudaMalloc(&d_index_,
				(hg.num_nodes() + 1) * sizeof(destination_type *));

		// copy index
		auto ds = d_edges_;
		cudaMemcpy(d_index_, &d_edges_, sizeof(destination_type *),
				cudaMemcpyHostToDevice);
		for (size_t i = 1; i <= hg.num_nodes(); ++i) {
			ds += hg.degree(i - 1);
			cudaMemcpy(d_index_ + i, &ds, sizeof(destination_type *),
					cudaMemcpyHostToDevice);
		}

		// copy edges
		cudaMemcpy(d_edges_, hg.neighbors(0).begin(),
				hg.num_edges() * sizeof(destination_type),
				cudaMemcpyHostToDevice);
	}

	~cuda_graph_t() {
		cudaFree(d_edges_);
		cudaFree(d_index_);
	}

	//device pointers
	destination_type **d_index_, *d_edges_;
};

#endif  // IM_CUDA_CUDA_COMMON_CUH
