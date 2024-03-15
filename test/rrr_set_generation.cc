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

#include "catch2/catch.hpp"

#include "ripples/imm_configuration.h"
#include "ripples/imm_interface.h"
#include "ripples/graph.h"
#include "ripples/imm.h"

#include "trng/lcg64.hpp"
#include <vector>

#if defined RIPPLES_ENABLE_UINT8_WEIGHTS
  using EdgeT = ripples::Edge<uint32_t, uint8_t>;
  constexpr uint8_t wt = std::numeric_limits<uint8_t>::max()/2;
#elif defined RIPPLES_ENABLE_UINT16_WEIGHTS
  using EdgeT = ripples::Edge<uint32_t, uint16_t>;
  constexpr uint16_t wt = std::numeric_limits<uint16_t>::max()/2;
#else
  using EdgeT = ripples::Edge<uint32_t, float>;
  constexpr float wt = 0.5;
#endif // RIPPLES_WEIGHT_QUANT
std::vector<EdgeT> karate{
    {1, 2, wt},   {1, 3, wt},   {1, 4, wt},   {1, 5, wt},   {1, 6, wt},
    {1, 7, wt},   {1, 8, wt},   {1, 9, wt},   {1, 11, wt},  {1, 12, wt},
    {1, 13, wt},  {1, 14, wt},  {1, 18, wt},  {1, 20, wt},  {1, 22, wt},
    {1, 32, wt},  {2, 3, wt},   {2, 4, wt},   {2, 8, wt},   {2, 14, wt},
    {2, 18, wt},  {2, 20, wt},  {2, 22, wt},  {2, 31, wt},  {3, 4, wt},
    {3, 8, wt},   {3, 9, wt},   {3, 10, wt},  {3, 14, wt},  {3, 28, wt},
    {3, 29, wt},  {3, 33, wt},  {4, 8, wt},   {4, 13, wt},  {4, 14, wt},
    {5, 7, wt},   {5, 11, wt},  {6, 7, wt},   {6, 11, wt},  {6, 17, wt},
    {7, 17, wt},  {9, 31, wt},  {9, 33, wt},  {9, 34, wt},  {10, 34, wt},
    {14, 34, wt}, {15, 33, wt}, {15, 34, wt}, {16, 33, wt}, {16, 34, wt},
    {19, 33, wt}, {19, 34, wt}, {20, 34, wt}, {21, 33, wt}, {21, 34, wt},
    {23, 33, wt}, {23, 34, wt}, {24, 26, wt}, {24, 28, wt}, {24, 30, wt},
    {24, 33, wt}, {24, 34, wt}, {25, 26, wt}, {25, 28, wt}, {25, 32, wt},
    {26, 32, wt}, {27, 30, wt}, {27, 34, wt}, {28, 34, wt}, {29, 32, wt},
    {29, 34, wt}, {30, 33, wt}, {30, 34, wt}, {31, 33, wt}, {31, 34, wt},
    {32, 33, wt}, {32, 34, wt}, {33, 34, wt}};

SCENARIO("Generate RRR sets", "[rrrsets]") {
  GIVEN("The Karate Graph") {
    #if defined RIPPLES_ENABLE_UINT8_WEIGHTS
      using dest_type = ripples::WeightedDestination<uint32_t, uint8_t>;
    #elif defined RIPPLES_ENABLE_UINT16_WEIGHTS
      using dest_type = ripples::WeightedDestination<uint32_t, uint16_t>;
    #else
      using dest_type = ripples::WeightedDestination<uint32_t, float>;
    #endif // RIPPLES_WEIGHT_QUANT
    using GraphFwd = ripples::Graph<uint32_t, dest_type,
                                    ripples::ForwardDirection<uint32_t>>;
    using GraphBwd = ripples::Graph<uint32_t, dest_type,
                                    ripples::BackwardDirection<uint32_t>>;
    using vertex_type = typename GraphFwd::vertex_type;

    GraphFwd Gfwd(karate.begin(), karate.end(), false);
    GraphBwd G = Gfwd.get_transpose();

    WHEN("I build the theta RRR sets sequentially") {
      size_t theta = 100;
      std::vector<ripples::RRRset<GraphBwd>> RR(theta);
      ripples::IMMExecutionRecord exRecord;

      std::vector<trng::lcg64> generator(1);
      for (size_t i = 0; i < 2; ++i) {
        ripples::GenerateRRRSets(G, generator, RR.end() - theta, RR.end(),
                                 exRecord, ripples::independent_cascade_tag{},
                                 ripples::sequential_tag{});

        THEN("They all contain a non empty list of vertices.") {
          for (auto& e : RR) {
            REQUIRE(!e.empty());
            for (auto v : e) {
              REQUIRE(v >= 0);
              REQUIRE(v < G.num_nodes());
            }
          }
        }

        RR.insert(RR.end(), theta, ripples::RRRset<GraphBwd>{});
      }
    }

    WHEN("I build the theta RRR sets in parallel") {
      size_t theta = 100;
      std::vector<ripples::RRRset<GraphBwd>> RR(theta);
      ripples::IMMExecutionRecord exRecord;

      size_t max_num_threads(1);
      max_num_threads = omp_get_max_threads();

      trng::lcg64 gen;
      ripples::IMMExecutionRecord R;
      ripples::ICStreamingGenerator generator(G, gen, max_num_threads, 0, 0, 0, 64,
                                                std::unordered_map<size_t, size_t>());
      for (size_t i = 0; i < 2; ++i) {
        ripples::GenerateRRRSets(G, generator, RR.end() - theta, RR.end(),
                                 exRecord, ripples::independent_cascade_tag{},
                                 ripples::omp_parallel_tag{});

        THEN("They all contain a non empty list of vertices.") {
          for (auto& e : RR) {
            REQUIRE(!e.empty());
            for (auto v : e) {
              REQUIRE(v >= 0);
              REQUIRE(v < G.num_nodes());
            }
          }
        }

        RR.insert(RR.end(), theta, ripples::RRRset<GraphBwd>{});
      }
    }
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
    WHEN("I build the theta RRR sets in parallel with all GPUs") {
      size_t theta = 100;
      std::vector<ripples::RRRset<GraphBwd>> RR(theta);
      ripples::IMMExecutionRecord exRecord;

      size_t max_num_threads(1);
      max_num_threads = omp_get_max_threads();
      size_t num_gpus = 1;

      std::unordered_map<size_t, size_t> gpu_mapping;
      for (int i = 0; i < num_gpus; ++i) {
        gpu_mapping[max_num_threads - num_gpus + i] = i;
      }

      trng::lcg64 gen;
      ripples::IMMExecutionRecord R;
      ripples::ICStreamingGenerator generator(G, gen, max_num_threads - num_gpus, 0, num_gpus, 64, 64,
                                              gpu_mapping);
      for (size_t i = 0; i < 2; ++i) {
        ripples::GenerateRRRSets(G, generator, RR.end() - theta, RR.end(),
                                 exRecord, ripples::independent_cascade_tag{},
                                 ripples::omp_parallel_tag{});

        THEN("They all contain a non empty list of vertices.") {
          for (auto& e : RR) {
            REQUIRE(!e.empty());
            for (auto v : e) {
              REQUIRE(v >= 0);
              REQUIRE(v < G.num_nodes());
            }
          }
        }

        RR.insert(RR.end(), theta, ripples::RRRset<GraphBwd>{});
      }
    }
#endif
  }
}
