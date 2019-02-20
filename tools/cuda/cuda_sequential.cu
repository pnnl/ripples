//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include "im/cuda/cuda_common.cuh"
#include "im/cuda/cuda_sequential.h"

namespace im {

template<typename DeviceGraphTy>
__global__
void seq_bfs_kernel(typename DeviceGraphTy::destination_type **index,
		size_t num_nodes, typename DeviceGraphTy::destination_type *edges,
		typename DeviceGraphTy::vertex_type *dres, size_t *dres_num) {
	*dres_num = 0;
}

//template<typename res_t, typename GraphTy, typename PRNGeneratorTy,
//		typename diff_model_tag>
void cuda_sequential_impl(cuda_res_t &rrr_sets, const cuda_GraphTy &G,
		size_t theta, cuda_PRNGeneratorTy &generator,
		im::linear_threshold_tag &&model_tag) {
	using vertex_type = typename cuda_GraphTy::vertex_type;

	// allocate host memory for results
	for (auto &set : rrr_sets)
		set.reserve(G.num_nodes());

	// copy graph to device
	cuda_graph_t<cuda_GraphTy> cuda_graph(G);

	// allocate device memory for results
	vertex_type *dres;
	size_t *dres_num;
	cudaMalloc(&dres, G.num_nodes() * sizeof(vertex_type));
	cudaMalloc(&dres_num, sizeof(size_t));

	for (auto &set : rrr_sets) {
		// run kernel
		seq_bfs_kernel<cuda_graph_t<cuda_GraphTy>> <<<1, 1>>>(cuda_graph.index_,
				cuda_graph.num_nodes_, cuda_graph.edges_, dres, dres_num);

		// copy results back to host and convert to host representation
		size_t res_num;
		cudaMemcpy(&res_num, dres_num, sizeof(size_t), cudaMemcpyDeviceToHost);

		set.resize(res_num);
		cudaMemcpy(set.data(), dres, res_num * sizeof(vertex_type),
				cudaMemcpyDeviceToHost);
	}
}

void cuda_sequential(cuda_res_t &r, const cuda_GraphTy &g, size_t s,
		cuda_PRNGeneratorTy& p, im::linear_threshold_tag&&t) {
	cuda_sequential_impl(r, g, s, p,
			std::forward < im::linear_threshold_tag > (t));
}

} // namespace im
