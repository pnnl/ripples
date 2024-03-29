/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 * Copyright (c) 2020, Battelle Memorial Institute.
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

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>

#include "ripples/graph.h"
#include "ripples/gpu/generate_rrr_sets.h"
#include "ripples/cuda/from_nvgraph/imm/bfs.hxx"
#include "ripples/cuda/from_nvgraph/nvgraph_error.hxx"

#include "bfs_kernels.cu"

using namespace bfs_kernels;

namespace nvgraph {
  enum BFS_ALGO_STATE {
    TOPDOWN, BOTTOMUP
  };

  template<typename IndexType, typename PRNGeneratorTy>
  NVGRAPH_ERROR Bfs<IndexType, PRNGeneratorTy>::setup() {

    // Determinism flag, false by default
    deterministic = false;

    //Working data
    //Each vertex can be in the frontier at most once
    cudaMalloc(&frontier, n * sizeof(IndexType));
    cudaCheckError();

    //We will update frontier during the execution
    //We need the orig to reset frontier, or cudaFree
    original_frontier = frontier;

    //size of bitmaps for vertices
    vertices_bmap_size = (n / (8 * sizeof(int)) + 1);
    //ith bit of visited_bmap is set <=> ith vertex is visited
    cudaMalloc(&visited_bmap, sizeof(int) * vertices_bmap_size);
    cudaCheckError();

    //ith bit of isolated_bmap is set <=> degree of ith vertex = 0
    cudaMalloc(&isolated_bmap, sizeof(int) * vertices_bmap_size);
    cudaCheckError();

    //vertices_degree[i] = degree of vertex i
    cudaMalloc(&vertex_degree, sizeof(IndexType) * n);
    cudaCheckError();

    //Cub working data
    cub_exclusive_sum_alloc(n + 1, d_cub_exclusive_sum_storage, cub_exclusive_sum_storage_bytes);

    //We will need (n+1) ints buffer for two differents things (bottom up or top down) - sharing it since those uses are mutually exclusive
    cudaMalloc(&buffer_np1_1, (n + 1) * sizeof(IndexType));
    cudaCheckError();

    cudaMalloc(&buffer_np1_2, (n + 1) * sizeof(IndexType));
    cudaCheckError();

    //Using buffers : top down

    //frontier_vertex_degree[i] is the degree of vertex frontier[i]
    frontier_vertex_degree = buffer_np1_1;
    //exclusive sum of frontier_vertex_degree
    exclusive_sum_frontier_vertex_degree = buffer_np1_2;

    //Using buffers : bottom up

    //contains list of unvisited vertices
    unvisited_queue = buffer_np1_1;
    //size of the "last" unvisited queue : size_last_unvisited_queue
    //refers to the size of unvisited_queue
    //which may not be up to date (the queue may contains vertices that are now visited)

    //We may leave vertices unvisited after bottom up main kernels - storing them here
    left_unvisited_queue = buffer_np1_2;

    //We use buckets of edges (32 edges per bucket for now, see exact macro in bfs_kernels). frontier_vertex_degree_buckets_offsets[i] is the index k such as frontier[k] is the source of the first edge of the bucket
    //See top down kernels for more details
    cudaMalloc(&exclusive_sum_frontier_vertex_buckets_offsets,
                           ((nnz / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2) * sizeof(IndexType));
    cudaCheckError();

    //Init device-side counters
    //Those counters must be/can be reset at each bfs iteration
    //Keeping them adjacent in memory allow use call only one cudaMemset - launch latency is the current bottleneck
    cudaMalloc(&d_counters_pad, 4 * sizeof(IndexType));
    cudaCheckError();

    d_new_frontier_cnt = &d_counters_pad[0];
    d_mu = &d_counters_pad[1];
    d_unvisited_cnt = &d_counters_pad[2];
    d_left_unvisited_cnt = &d_counters_pad[3];

    //Lets use this int* for the next 3 lines
    //Its dereferenced value is not initialized - so we dont care about what we put in it
    IndexType * d_nisolated = d_new_frontier_cnt;
    cudaMemsetAsync(d_nisolated, 0, sizeof(IndexType), stream);
    cudaCheckError()
    ;

    //Computing isolated_bmap
    //Only dependent on graph - not source vertex - done once
    flag_isolated_vertices(n, isolated_bmap, row_offsets, vertex_degree,
                           d_nisolated, dyn_max_blocks, stream);
    cudaMemcpyAsync(&nisolated, d_nisolated, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
    cudaCheckError()
    ;

    //We need nisolated to be ready to use
    cudaStreamSynchronize(stream);
    cudaCheckError()
    ;

    return NVGRAPH_OK;
  }

  template<typename IndexType, typename PRNGeneratorTy>
  NVGRAPH_ERROR Bfs<IndexType, PRNGeneratorTy>::configure(  IndexType *_distances,
                              IndexType *_predecessors,
                              int *_edge_mask)
                              {
    distances = _distances;
    predecessors = _predecessors;
    edge_mask = _edge_mask;

    useEdgeMask = (edge_mask != NULL);
    computeDistances = (distances != NULL);
    computePredecessors = (predecessors != NULL);

    //We need distances to use bottom up
    if (directed && !computeDistances) {
      cudaMalloc(&distances, n * sizeof(IndexType));
      cudaCheckError();
    }

    return NVGRAPH_OK;
  }

  template<typename IndexType, typename PRNGeneratorTy>
  IndexType Bfs<IndexType, PRNGeneratorTy>::traverse_block_size() {
    return frontier_expand_block_size<IndexType>();
  }

  template<typename IndexType, typename PRNGeneratorTy>
  NVGRAPH_ERROR Bfs<IndexType, PRNGeneratorTy>::traverse(IndexType source_vertex) {

    //Init visited_bmap
    //If the graph is undirected, we not that
    //we will never discover isolated vertices (in degree = out degree = 0)
    //we avoid a lot of work by flagging them now
    //in g500 graphs they represent ~25% of total vertices
    //more than that for wiki and twitter graphs

    if (directed) {
      cudaMemsetAsync(visited_bmap, 0, vertices_bmap_size * sizeof(int), stream);
    } else {
      cudaMemcpyAsync(  visited_bmap,
                  isolated_bmap,
                  vertices_bmap_size * sizeof(int),
                  cudaMemcpyDeviceToDevice,
                  stream);
    }
    cudaCheckError()
    ;

    //If needed, setting all vertices as undiscovered (inf distance)
    //We dont use computeDistances here
    //if the graph is undirected, we may need distances even if
    //computeDistances is false
    if (distances)
      fill_vec(distances, n, vec_t<IndexType>::max, stream);

    //If needed, setting all predecessors to non-existent (-1)
    if (computePredecessors)
    {
      cudaMemsetAsync(predecessors, -1, n * sizeof(IndexType), stream);
      cudaCheckError()
      ;
    }

    //
    //Initial frontier
    //

    frontier = original_frontier;

    if (distances)
    {
      cudaMemsetAsync(&distances[source_vertex], 0, sizeof(IndexType), stream);
      cudaCheckError()
      ;
    }

    //Setting source_vertex as visited
    //There may be bit already set on that bmap (isolated vertices) - if the graph is undirected
    int current_visited_bmap_source_vert = 0;

    if (!directed) {
      cudaMemcpyAsync(&current_visited_bmap_source_vert,
                  &visited_bmap[source_vertex / INT_SIZE],
                  sizeof(int),
                  cudaMemcpyDeviceToHost);
      cudaCheckError()
      ;
      //We need current_visited_bmap_source_vert
      cudaStreamSynchronize(stream);
      cudaCheckError()
      ;
      //We could detect that source is isolated here
    }

    int m = (1 << (source_vertex % INT_SIZE));

    //In that case, source is isolated, done now
    if (!directed && (m & current_visited_bmap_source_vert)) {
      //Init distances and predecessors are done, (cf Streamsync in previous if)
      cudaCheckError()
      ;
      return NVGRAPH_OK;
    }

    m |= current_visited_bmap_source_vert;

    cudaMemcpyAsync(  &visited_bmap[source_vertex / INT_SIZE],
                &m,
                sizeof(int),
                cudaMemcpyHostToDevice,
                stream);
    cudaCheckError()
    ;

    //Adding source_vertex to init frontier
    cudaMemcpyAsync(  &frontier[0],
                &source_vertex,
                sizeof(IndexType),
                cudaMemcpyHostToDevice,
                stream);
    cudaCheckError()
    ;

    //mf : edges in frontier
    //nf : vertices in frontier
    //mu : edges undiscovered
    //nu : nodes undiscovered
    //lvl : current frontier's depth
    IndexType mf, nf, mu, nu;
    IndexType lvl = 1;

    //Frontier has one vertex
    nf = 1;

    //all edges are undiscovered (by def isolated vertices have 0 edges)
    mu = nnz;

    //all non isolated vertices are undiscovered (excepted source vertex, which is in frontier)
    //That number is wrong if source_vertex is also isolated - but it's not important
    nu = n - nisolated - nf;

    //Typical pre-top down workflow. set_frontier_degree + exclusive-scan
    set_frontier_degree(frontier_vertex_degree, frontier, vertex_degree, nf,
                        dyn_max_blocks, stream);
    exclusive_sum(  d_cub_exclusive_sum_storage,
              cub_exclusive_sum_storage_bytes,
              frontier_vertex_degree,
              exclusive_sum_frontier_vertex_degree,
              nf + 1,
              stream);

    cudaMemcpyAsync(  &mf,
                &exclusive_sum_frontier_vertex_degree[nf],
                sizeof(IndexType),
                cudaMemcpyDeviceToHost,
                stream);
    cudaCheckError()
    ;

    //We need mf
    cudaStreamSynchronize(stream);
    cudaCheckError()
    ;

    while (nf > 0) {
      // Each vertices can appear only once in the frontierer array - we know it
      // will fit
      new_frontier = frontier + nf;
      resetDevicePointers();

      // Executing algo
      compute_bucket_offsets(exclusive_sum_frontier_vertex_degree,
                             exclusive_sum_frontier_vertex_buckets_offsets, nf,
                             mf, dyn_max_blocks, stream);

#if CUDA_CHECK
      if (mf) {
        auto block_size = frontier_expand_block_size<IndexType>();
        IndexType max_items_per_thread =
            (mf + dyn_max_blocks * block_size - 1) / (dyn_max_blocks * block_size);
        auto num_blocks = min((mf + max_items_per_thread * block_size - 1) /
                                  (max_items_per_thread * block_size),
                              dyn_max_blocks);
        auto max_num_threads =
          dyn_max_blocks * frontier_expand_block_size<IndexType>();
        assert(block_size * num_blocks <= max_num_threads);
      }
#endif

      frontier_expand(row_offsets, col_indices, weights, frontier, nf, mf, lvl,
                      new_frontier, d_new_frontier_cnt,
                      exclusive_sum_frontier_vertex_degree,
                      exclusive_sum_frontier_vertex_buckets_offsets,
                      visited_bmap, distances, predecessors, edge_mask,
                      isolated_bmap, directed, dyn_max_blocks, stream,
                      deterministic, d_trng_state_, rng_offset_, num_rngs_);

      mu -= mf;

      cudaMemcpyAsync(&nf, d_new_frontier_cnt, sizeof(IndexType),
                      cudaMemcpyDeviceToHost, stream);
      cudaCheckError();

      // We need nf
      cudaStreamSynchronize(stream);
      cudaCheckError();

      if (nf) {
        // Typical pre-top down workflow. set_frontier_degree + exclusive-scan
        set_frontier_degree(frontier_vertex_degree, new_frontier, vertex_degree,
                            nf, dyn_max_blocks, stream);
        exclusive_sum(d_cub_exclusive_sum_storage,
                      cub_exclusive_sum_storage_bytes, frontier_vertex_degree,
                      exclusive_sum_frontier_vertex_degree, nf + 1, stream);
        cudaMemcpyAsync(&mf, &exclusive_sum_frontier_vertex_degree[nf],
                        sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
        cudaCheckError();

        // We need mf
        cudaStreamSynchronize(stream);
        cudaCheckError();
      }

      // Updating undiscovered edges count
      nu -= nf;

      // Using new frontier
      frontier = new_frontier;

      ++lvl;
    }

    cudaCheckError()
    ;
    return NVGRAPH_OK;
  }

  //Just used for benchmarks now
  template<typename IndexType, typename PRNGeneratorTy>
  NVGRAPH_ERROR Bfs<IndexType, PRNGeneratorTy>::traverse(IndexType *source_vertices, IndexType nsources) {
    for (IndexType i = 0; i < nsources; ++i)
      traverse(source_vertices[i]);

    return NVGRAPH_OK;
  }

  template<typename IndexType, typename PRNGeneratorTy>
  void Bfs<IndexType, PRNGeneratorTy>::resetDevicePointers() {
    cudaMemsetAsync(d_counters_pad, 0, 4 * sizeof(IndexType), stream);
    cudaCheckError()
    ;
  }

  template<typename IndexType, typename PRNGeneratorTy>
  void Bfs<IndexType, PRNGeneratorTy>::clean() {
    cudaCheckError()
    ;

    //the vectors have a destructor that takes care of cleaning
    cudaFree(original_frontier);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(visited_bmap);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(isolated_bmap);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(vertex_degree);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(d_cub_exclusive_sum_storage);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(buffer_np1_1);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(buffer_np1_2);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(exclusive_sum_frontier_vertex_buckets_offsets);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.
    cudaFree(d_counters_pad);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.

    //In that case, distances is a working data
    if (directed && !computeDistances)
      cudaFree(distances);//Better to be error checked, but we do not have a policy for error checking yet (in particular for void functions), so I defer error check as future work.

    cudaCheckError()
    ;
  }

  template class Bfs<int, trng::lcg64>;
} // end namespace nvgraph
