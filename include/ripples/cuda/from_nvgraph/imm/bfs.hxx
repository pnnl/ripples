/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <climits>

#include <cuda_runtime.h>

#include "ripples/graph.h"

// Used in nvgraph/nvgraph.h

#define TRAVERSAL_DEFAULT_ALPHA 15

#define TRAVERSAL_DEFAULT_BETA 18

#include "ripples/cuda/from_nvgraph/nvgraph_error.hxx"

namespace nvgraph

{

template <typename IndexType, typename PRNGeneratorTy>

class Bfs

{
 private:
  IndexType n, nnz;

  IndexType *row_offsets;

  IndexType *col_indices;

  float *weights;

  bool directed;
  bool deterministic;

  // edgemask, distances, predecessors are set/read by users - using Vectors

  bool useEdgeMask;

  bool computeDistances;

  bool computePredecessors;

  IndexType *distances;

  IndexType *predecessors;

  int *edge_mask;

  // Working data

  // For complete description of each, go to bfs.cu

  IndexType nisolated;

  IndexType *frontier, *new_frontier;

  IndexType *original_frontier;

  IndexType vertices_bmap_size;

  int *visited_bmap, *isolated_bmap;

  IndexType *vertex_degree;

  IndexType *buffer_np1_1, *buffer_np1_2;

  IndexType *frontier_vertex_degree;

  IndexType *exclusive_sum_frontier_vertex_degree;

  IndexType *unvisited_queue;

  IndexType *left_unvisited_queue;

  IndexType *exclusive_sum_frontier_vertex_buckets_offsets;

  IndexType *d_counters_pad;

  IndexType *d_new_frontier_cnt;

  IndexType *d_mu;

  IndexType *d_unvisited_cnt;

  IndexType *d_left_unvisited_cnt;

  void *d_cub_exclusive_sum_storage;

  size_t cub_exclusive_sum_storage_bytes;

  // Parameters for direction optimizing

  IndexType alpha, beta;

  cudaStream_t stream;

  PRNGeneratorTy *d_trng_state_{nullptr};
  size_t rng_offset_{0};
  size_t num_rngs_;

  // resets pointers defined by d_counters_pad (see implem)

  void resetDevicePointers();

  NVGRAPH_ERROR setup();

  void clean();

  const IndexType dyn_max_blocks;

 public:
  virtual ~Bfs(void) { clean(); };

  Bfs(IndexType _n, IndexType _nnz, IndexType *_row_offsets,
      IndexType *_col_indices, float *_weights, bool _directed,
      IndexType _alpha, IndexType _beta, IndexType _dyn_max_blocks,
      cudaStream_t _stream = 0)
      : n(_n),
        nnz(_nnz),
        row_offsets(_row_offsets),
        col_indices(_col_indices),
        weights(_weights),
        directed(_directed),
        alpha(_alpha),
        beta(_beta),
        dyn_max_blocks(_dyn_max_blocks),
        stream(_stream)
  {
    num_rngs_ = dyn_max_blocks * traverse_block_size();
    setup();
  }

  NVGRAPH_ERROR
  configure(IndexType *distances, IndexType *predecessors, int *edge_mask);

  NVGRAPH_ERROR traverse(IndexType source_vertex);

  // Used only for benchmarks

  NVGRAPH_ERROR traverse(IndexType *source_vertices, IndexType nsources);

  static IndexType traverse_block_size();
  void rng(PRNGeneratorTy *d_trng_state) {
	d_trng_state_ = d_trng_state;
  }
};

}  // end namespace nvgraph
