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
void seq_bfs_kernel(DeviceGraphTy *G, typename DeviceGraphTy::vertex_type *dres,
		size_t *dres_num) {
	*dres_num = 0;
}

//template<typename res_t, typename GraphTy, typename PRNGeneratorTy,
//		typename diff_model_tag>
void cuda_sequential_impl(cuda_res_t &rrr_sets, const cuda_GraphTy &G, size_t theta,
		cuda_PRNGeneratorTy &generator, im::linear_threshold_tag &&model_tag) {
	using vertex_type = typename cuda_GraphTy::vertex_type;

	// allocate host memory for results
	for (auto &set : rrr_sets)
		set.reserve(G.num_nodes());

	// copy graph to device
	auto device_graph = cuda_h2d_graph(G);

	// allocate device memory for results
	vertex_type *dres;
	size_t *dres_num;
	cudaMalloc(&dres, G.num_nodes() * sizeof(vertex_type));
	cudaMalloc(&dres_num, sizeof(size_t));

	//size_t i = 0;
	for (auto &set : rrr_sets) {
		// run kernel
		seq_bfs_kernel<<<1, 1>>>(device_graph, dres, dres_num);
		auto err = cudaGetLastError();
		if (cudaSuccess != err) {
			printf("CUDA-kernel error: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		// copy results back to host and convert to host representation
		size_t res_num;
		cudaMemcpy(&res_num, dres_num, sizeof(size_t), cudaMemcpyDeviceToHost);

		set.resize(res_num);
		cudaMemcpy(set.data(), dres, res_num * sizeof(vertex_type),
				cudaMemcpyDeviceToHost);
	}

	cuda_destroy_graph(device_graph);
}

void cuda_sequential(cuda_res_t &r, const cuda_GraphTy &g, size_t s,
		cuda_PRNGeneratorTy& p, im::linear_threshold_tag&&t) {
	cuda_sequential_impl(r, g, s, p,
			std::forward < im::linear_threshold_tag > (t));
}

//void cuda_sequential(cuda_res_t &, const cuda_GraphTy &, size_t,
//		cuda_PRNGeneratorTy&, im::independent_cascade_tag&&) {
//	cuda_sequential_impl(r, g, s, p,
//			std::forward < im::independent_cascade_tag > (t));
//}

} // namespace im
