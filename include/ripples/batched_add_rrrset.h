//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_BATCHED_ADD_RRRSET_H
#define RIPPLES_BATCHED_ADD_RRRSET_H

#include <cassert>
#include <algorithm>
#include <numeric>
#include <string>
#include <ostream>
#include <map>
#include <unordered_map>

#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"
#include "trng/uniform01_dist.hpp"

#include <sched.h>

#define NEIGHBOR_COLOR

namespace ripples {

struct BFSCPUContext{
  BFSCPUContext(size_t num_nodes) : old_visited_matrix(num_nodes),
                                    new_visited_matrix(num_nodes) {}
  std::vector<uint64_t> old_visited_matrix;
  std::vector<uint64_t> new_visited_matrix;
};

template <typename GraphTy, typename SItrTy, typename OItrTy,
          typename PRNGeneratorTy, typename diff_model_tag>
void BatchedBFS(const GraphTy &G, SItrTy B, SItrTy E, OItrTy O,
                PRNGeneratorTy& generator,
                diff_model_tag &&tag) {
  assert(std::distance(B, E) <= 64 && "Only up to 64 BFS are supported");
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<std::vector<bool>> visited_matrix(std::distance(B, E), std::vector<bool>(G.num_nodes(), false));

  // using frontier_element = std::pair<vertex_type, uint64_t>;
  using frontier_element = vertex_type;
  std::unordered_map<vertex_type, uint64_t> color_map, new_color_map;

  trng::uniform01_dist<float> value;

  uint64_t color = 1ul << 63;
  std::vector<frontier_element> frontier, new_frontier;
  for (auto itr = B; itr < E; ++itr, color /= 2) {
    auto pos = color_map.find(*itr);
    if (pos == color_map.end()) {
      frontier.push_back(*itr);
      color_map[*itr] = color;
    } else {
      color_map[*itr] |= color;
    }
    visited_matrix[__builtin_clzl(color)][*itr] = true;
    (O + __builtin_clzl(color))->push_back(*itr);
  }

  assert(frontier.size() != 0);

  while (frontier.size() != 0) {
    new_frontier.resize(0);
    new_color_map.clear();
    // The compacted frontier is now in `frontier`
    std::for_each(frontier.begin(), frontier.end(), [&](const auto & v) {
      // auto vertex = v.first;
      // auto colors = v.second;
      auto vertex = v;
      auto colors = color_map[vertex];

      if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
        while (colors != 0) {
          uint64_t color = __builtin_clzl(colors);

          for (auto u : G.neighbors(vertex)) {
            #ifdef NEIGHBOR_COLOR
            if (!visited_matrix[color][u.vertex] && value(generator[0][0]) <= u.weight) {
            #else
            if (!visited_matrix[color][u.vertex] && value(generator) <= u.weight) {
            #endif
              visited_matrix[color][u.vertex] = true;
              (O + color)->push_back(u.vertex);
              auto pos = new_color_map.find(u.vertex);
              if (pos == new_color_map.end()) {
                new_frontier.push_back(u.vertex);
                new_color_map[u.vertex] = (1ul << ((sizeof(colors) * 8 - 1) - color));
              } else {
                new_color_map[u.vertex] |= (1ul << ((sizeof(colors) * 8 - 1) - color));
              }
            }
          }

          colors -= (1ul << ((sizeof(colors) * 8 - 1) - color));
        }
      } else if (std::is_same<diff_model_tag, ripples::linear_threshold_tag>::value) {
        while (colors != 0) {
          uint64_t color = __builtin_clzl(colors);

          #ifdef NEIGHBOR_COLOR
          float threshold = value(generator[0][0]);
          #else
          float threshold = value(generator);
          #endif
          for (auto u : G.neighbors(vertex)) {
            threshold -= u.weight;
            if (threshold > 0) continue;

            if (!visited_matrix[color][u.vertex]) {
              visited_matrix[color][u.vertex] = true;
              (O + color)->push_back(u.vertex);
              auto pos = new_color_map.find(u.vertex);
              if (pos == new_color_map.end()) {
                new_frontier.push_back(u.vertex);
                new_color_map[u.vertex] = (1ul << ((sizeof(colors) * 8 - 1) - color));
              } else {
                new_color_map[u.vertex] |= (1ul << ((sizeof(colors) * 8 - 1) - color));
              }
            }
            break;
          }

          colors -= (1ul << ((sizeof(colors) * 8 - 1) - color));
        }
      } else {
        throw;
      }
    });

    std::swap(color_map, new_color_map);
    std::swap(frontier, new_frontier);
  }

  for (int i = 0; i < std::distance(B, E); ++i, ++O) {
    std::sort(O->begin(), O->end());
  }
}

template <typename GraphTy, typename SItrTy, typename OItrTy,
          typename PRNGeneratorTy, typename diff_model_tag>
void BatchedBFSNeighborColor(const GraphTy &G, SItrTy B, SItrTy E, OItrTy O,
                PRNGeneratorTy& generator,
                diff_model_tag &&tag) {
  assert(std::distance(B, E) <= 64 && "Only up to 64 BFS are supported");
  using vertex_type = typename GraphTy::vertex_type;
  std::vector<uint64_t> old_visited_matrix(G.num_nodes(), 0);
  std::vector<uint64_t> new_visited_matrix(G.num_nodes(), 0);

  trng::uniform01_dist<float> value;

  uint64_t color = 1ul << 63;
  for (auto itr = B; itr < E; ++itr, color >>= 1) {
    new_visited_matrix[*itr] |= color;
  }

  bool found_one = true;

  while (found_one) {
    found_one = false;
    // Iterate over both visited_vertex and new_visited_vertex
    for (vertex_type vertex = 0; vertex < G.num_nodes(); ++vertex) {
      const uint64_t visited_old = old_visited_matrix[vertex];
      const uint64_t visited_new = new_visited_matrix[vertex];
      if (visited_old == visited_new) continue;
      found_one = true;
      uint64_t colors = visited_new ^ visited_old;
      old_visited_matrix[vertex] |= visited_new;
      // Convert the colors to an array of color masks
      uint64_t color_masks[64];
      uint64_t num_colors = 0;
      while (colors != 0) {
        uint64_t color = __builtin_clzl(colors);
        color_masks[num_colors++] = (1ul << ((sizeof(colors) * 8 - 1) - color));
        colors -= (1ul << ((sizeof(colors) * 8 - 1) - color));
      }

      for (auto u : G.neighbors(vertex)) {
        const uint64_t old_mask = new_visited_matrix[u.vertex];
        uint64_t new_mask = 0;
        #pragma omp simd reduction(|:new_mask)
        for(size_t i = 0; i < num_colors; ++i) {
          const uint64_t color_mask = color_masks[i];
          if(!(old_mask & color_mask) && value(generator[i]) <= u.weight) {
            new_mask |= color_mask;
          }
        }
        if(new_mask != 0){
          new_visited_matrix[u.vertex] |= new_mask;
        }
      }
    }
  }

  // Traverse visited_matrix and push the vertices into the output
  for (int i = 0; i < G.num_nodes(); ++i) {
    uint64_t colors = new_visited_matrix[i];
    while (colors != 0) {
      uint64_t color = __builtin_clzl(colors);
      (O + color)->push_back(i);
      colors -= (1ul << ((sizeof(colors) * 8 - 1) - color));
    }
  }
}

template <typename GraphTy, typename SItrTy, typename OItrTy,
          typename PRNGeneratorTy, typename diff_model_tag>
void BatchedBFSNeighborColorOMP(const GraphTy &G, SItrTy B, SItrTy E, OItrTy O,
                PRNGeneratorTy& generator,
                diff_model_tag &&tag, BFSCPUContext &cpu_ctx, const size_t num_threads) {
  size_t rank = omp_get_thread_num();
  assert(std::distance(B, E) <= 32 && "Only up to 32 BFS are supported");
  // std::cout << "Num threads: " << num_threads << std::endl;
  // std::cout << "omp rank: " << omp_get_thread_num() << std::endl; 
  using vertex_type = typename GraphTy::vertex_type;
  // std::fill(cpu_ctx.old_visited_matrix.begin(), cpu_ctx.old_visited_matrix.end(), 0);
  // std::fill(cpu_ctx.new_visited_matrix.begin(), cpu_ctx.new_visited_matrix.end(), 0);
  // Perform chunk fill
  #pragma omp parallel num_threads(num_threads) proc_bind(close)
  {
    // if(rank == 0) 
    // {
      // std::cout << "hwthread = " << sched_getcpu() << std::endl;
      // printf("rank = %d | hwthread = %d\n", omp_get_thread_num(), sched_getcpu());
    // }
    const size_t chunk_size = (G.num_nodes() + num_threads - 1) / num_threads;
    const size_t chunk_start = chunk_size * omp_get_thread_num();
    const size_t chunk_end = std::min(chunk_start + chunk_size, G.num_nodes());
    std::fill(cpu_ctx.old_visited_matrix.begin() + chunk_start, cpu_ctx.old_visited_matrix.begin() + chunk_end, 0);
    std::fill(cpu_ctx.new_visited_matrix.begin() + chunk_start, cpu_ctx.new_visited_matrix.begin() + chunk_end, 0);
  }
  uint32_t itr_color_mask = (uint32_t)1 << 31;
  for (auto itr = B; itr < E; ++itr, itr_color_mask >>= 1) {
    cpu_ctx.new_visited_matrix[*itr] |= itr_color_mask;
  }
  bool found_one = true;
  // size_t rank = omp_get_thread_num();
  auto &old_visited_matrix = cpu_ctx.old_visited_matrix;
  auto &new_visited_matrix = cpu_ctx.new_visited_matrix;
  // std::cout << "Entering section: " << rank << std::endl;
  trng::uniform01_dist<float> value;
  // size_t iteration = 0;
  while (found_one) {
    found_one = false;
    // iteration++;
    // std::cout << "Iteration: " << iteration << std::endl;
    // Iterate over both visited_vertex and new_visited_vertex
    #pragma omp parallel for proc_bind(close) num_threads(num_threads) schedule(dynamic, 16)
    for (vertex_type vertex = 0; vertex < G.num_nodes(); ++vertex) {
      const uint32_t visited_old = old_visited_matrix[vertex];
      const uint32_t visited_new = new_visited_matrix[vertex];
      if (visited_old != visited_new){
        const size_t inner_rank = omp_get_thread_num();
        found_one = true;
        uint32_t colors = visited_new ^ visited_old;
        // #pragma omp single
        old_visited_matrix[vertex] |= visited_new;
        // Convert the colors to an array of color masks
        uint32_t color_masks[32];
        uint32_t num_colors = 0;
        while (colors != 0) {
          uint32_t color = __builtin_clz(colors);
          color_masks[num_colors++] = ((uint32_t)1 << ((sizeof(colors) * 8 - 1) - color));
          colors -= ((uint32_t)1 << ((sizeof(colors) * 8 - 1) - color));
        }
        for (auto u : G.neighbors(vertex)) {
          const uint32_t old_mask = new_visited_matrix[u.vertex];
          uint32_t new_mask = 0;
          #pragma omp simd reduction(|:new_mask)
          for(size_t i = 0; i < num_colors; ++i) {
            const uint32_t color_mask = color_masks[i];
            if(!(old_mask & color_mask) && value(generator[inner_rank][i]) <= u.weight) {
              new_mask |= color_mask;
            }
          }
          if(new_mask != 0){
            #pragma omp atomic
            new_visited_matrix[u.vertex] |= new_mask;
          }
        }
      }
    }
  }
  
  // std::cout << "Copy Rank: " << rank << std::endl;
  // Traverse visited_matrix and push the vertices into the output
  // #pragma omp parallel for proc_bind(close) num_threads(num_threads)
  // #pragma omp parallel for proc_bind(close) num_threads(num_threads) schedule(static)
  // for (uint32_t color = 0; color < std::distance(B, E); ++color){
  //   // std::cout << "Color: " << color << std::endl;
  //   const uint32_t color_mask = (uint32_t)1 << ((sizeof(color_mask) * 8 - 1) - color);
  //   for (size_t i = 0; i < G.num_nodes(); ++i){
  //     if(new_visited_matrix[i] & color_mask){
  //       (O + color)->push_back(i);
  //     }
  //   }
  // }
  for (int i = 0; i < G.num_nodes(); ++i) {
    uint32_t colors = new_visited_matrix[i];
    while (colors != 0) {
      uint32_t color = __builtin_clz(colors);
      (O + color)->push_back(i);
      colors -= ((uint32_t)1 << ((sizeof(colors) * 8 - 1) - color));
    }
  }
}

}



#endif
