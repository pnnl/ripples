//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <curand_kernel.h>

#include "im/cuda/cuda_common.cuh"
#include "im/cuda/cuda_generate_rrr_sets.h"

#define CUDA_BATCHED 1

namespace im {

constexpr size_t BATCH_SIZE = 1024;
static unsigned long long d_rng_seed = 666;

extern __shared__ uint8_t shmem_lt_per_thread[];
template<typename DeviceGraphTy>
__global__
void kernel_lt_per_thread(typename DeviceGraphTy::destination_type **index,
		size_t num_nodes, curandState *d_rng_states, uint8_t *d_res_masks) {
	using destination_type = typename DeviceGraphTy::destination_type;
	using vertex_type = typename DeviceGraphTy::vertex_type;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t *d_res_mask = d_res_masks + tid * num_nodes;

	//init res memory
	uint8_t *shmem_res_mask = shmem_lt_per_thread + threadIdx.x * num_nodes;
	memset(shmem_res_mask, 0, num_nodes * sizeof(uint8_t));

	// cache rng state
	auto rng_state = d_rng_states + tid;

	// select source node
	vertex_type src = tid % num_nodes;

	float x, acc;
	destination_type *first, *last;
	while (src != num_nodes) {
		shmem_res_mask[src] = 1;
		first = index[src];
		last = index[src + 1];
		x = curand(rng_state);
		acc = 0;
		src = num_nodes;
		while (first != last) {
			if (x < (acc += first->weight)) {
				src = first->vertex;
				break;
			}
			++first;
		}
	}

	// write back results to global memory
	memcpy(d_res_mask, shmem_res_mask, num_nodes * sizeof(uint8_t));
}

__global__
void kernel_rng_setup(curandState *d_rng_states, unsigned long long seed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, d_rng_states + tid);
}

//template<typename res_t, typename GraphTy, typename PRNGeneratorTy,
//		typename diff_model_tag>
void CudaGenerateRRRSets(cuda_res_t &rrr_sets, const cuda_GraphTy &G,
		size_t theta, cuda_PRNGeneratorTy &generator,
		im::linear_threshold_tag &&model_tag) {
	using vertex_type = typename cuda_GraphTy::vertex_type;

	// copy graph to device
	cuda_graph_t<cuda_GraphTy> cuda_graph(G);

#if CUDA_BATCHED
	// allocate memory for result-masks
	uint8_t *res_masks, *d_res_masks;
	res_masks = (uint8_t *) malloc(
			BATCH_SIZE * G.num_nodes() * sizeof(uint8_t));
	cudaMalloc(&d_res_masks, BATCH_SIZE * G.num_nodes() * sizeof(uint8_t));

	//init rng
	curandState *d_rng_states;
	cudaMalloc(&d_rng_states, BATCH_SIZE * sizeof(curandState));
	kernel_rng_setup<<<128, BATCH_SIZE / 128>>>(d_rng_states, d_rng_seed++);

	for (size_t bf = 0; bf < rrr_sets.size(); bf += BATCH_SIZE) {
		//execute a batch
		kernel_lt_per_thread<cuda_graph_t<cuda_GraphTy>> <<<128,
				BATCH_SIZE / 128, G.num_nodes() * 128 * sizeof(uint8_t)>>>(
				cuda_graph.d_index_, G.num_nodes(), d_rng_states, d_res_masks);

		// copy masks back to host
		cudaMemcpy(res_masks, d_res_masks,
				BATCH_SIZE * G.num_nodes() * sizeof(uint8_t),
				cudaMemcpyDeviceToHost);

		// convert masks to results
		for (size_t i = 0; i < BATCH_SIZE; ++i) {
			auto set_idx = bf + i;
			if (set_idx < rrr_sets.size()) {
				for (size_t j = 0; j < G.num_nodes(); ++j)
					if (res_masks[i * G.num_nodes() + j])
						rrr_sets[bf + i].push_back(vertex_type(j));

				//check - TODO

			} else
				break;
		}
	}

	// cleanup
	free(res_masks);
	cudaFree(d_res_masks);
	cudaFree(d_rng_states);

#endif
}

void CudaGenerateRRRSets(cuda_res_t &rrr_sets, const cuda_GraphTy &G,
		size_t theta, cuda_PRNGeneratorTy &generator,
		im::independent_cascade_tag &&model_tag) {
	assert(false);
}

} // namespace im
