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
		num_edges_ = hg.num_edges();
		cudaMalloc(&edges_, num_edges_ * sizeof(destination_type));

		// prepare index
		num_nodes_ = hg.num_nodes();
		cudaMalloc(&index_, (num_nodes_ + 1) * sizeof(destination_type *));

		// copy index
		auto ds = edges_;
		cudaMemcpy(index_, &edges_, sizeof(destination_type *),
				cudaMemcpyHostToDevice);
		for (size_t i = 1; i <= hg.num_nodes(); ++i) {
			ds += hg.degree(i - 1);
			cudaMemcpy(index_ + i, &ds, sizeof(destination_type *),
					cudaMemcpyHostToDevice);
		}

		// copy edges
		cudaMemcpy(edges_, hg.neighbors(0).begin(),
				hg.num_edges() * sizeof(destination_type),
				cudaMemcpyHostToDevice);
	}

	~cuda_graph_t() {
		cudaFree(edges_);
		cudaFree(index_);
	}

	//device pointers
	destination_type **index_, *edges_;

	// host meta-data
	size_t num_nodes_, num_edges_;
};

#endif  // IM_CUDA_CUDA_COMMON_CUH
