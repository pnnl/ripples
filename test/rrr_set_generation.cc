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

#include "ripples/generate_rrr_sets.h"
#include "ripples/graph.h"
#include "ripples/imm.h"

#include "trng/lcg64.hpp"

using EdgeT = ripples::Edge<uint32_t, float>;
std::vector<EdgeT> karate{
    {1, 2, 0.5},   {1, 3, 0.5},   {1, 4, 0.5},   {1, 5, 0.5},   {1, 6, 0.5},
    {1, 7, 0.5},   {1, 8, 0.5},   {1, 9, 0.5},   {1, 11, 0.5},  {1, 12, 0.5},
    {1, 13, 0.5},  {1, 14, 0.5},  {1, 18, 0.5},  {1, 20, 0.5},  {1, 22, 0.5},
    {1, 32, 0.5},  {2, 3, 0.5},   {2, 4, 0.5},   {2, 8, 0.5},   {2, 14, 0.5},
    {2, 18, 0.5},  {2, 20, 0.5},  {2, 22, 0.5},  {2, 31, 0.5},  {3, 4, 0.5},
    {3, 8, 0.5},   {3, 9, 0.5},   {3, 10, 0.5},  {3, 14, 0.5},  {3, 28, 0.5},
    {3, 29, 0.5},  {3, 33, 0.5},  {4, 8, 0.5},   {4, 13, 0.5},  {4, 14, 0.5},
    {5, 7, 0.5},   {5, 11, 0.5},  {6, 7, 0.5},   {6, 11, 0.5},  {6, 17, 0.5},
    {7, 17, 0.5},  {9, 31, 0.5},  {9, 33, 0.5},  {9, 34, 0.5},  {10, 34, 0.5},
    {14, 34, 0.5}, {15, 33, 0.5}, {15, 34, 0.5}, {16, 33, 0.5}, {16, 34, 0.5},
    {19, 33, 0.5}, {19, 34, 0.5}, {20, 34, 0.5}, {21, 33, 0.5}, {21, 34, 0.5},
    {23, 33, 0.5}, {23, 34, 0.5}, {24, 26, 0.5}, {24, 28, 0.5}, {24, 30, 0.5},
    {24, 33, 0.5}, {24, 34, 0.5}, {25, 26, 0.5}, {25, 28, 0.5}, {25, 32, 0.5},
    {26, 32, 0.5}, {27, 30, 0.5}, {27, 34, 0.5}, {28, 34, 0.5}, {29, 32, 0.5},
    {29, 34, 0.5}, {30, 33, 0.5}, {30, 34, 0.5}, {31, 33, 0.5}, {31, 34, 0.5},
    {32, 33, 0.5}, {32, 34, 0.5}, {33, 34, 0.5}};

SCENARIO("Generate RRR sets", "[rrrsets]") {
  GIVEN("The Karate Graph") {
    using destination_type = ripples::WeightedDestination<uint32_t, float>;
    using GraphFwd = ripples::Graph<uint32_t, destination_type,
                                    ripples::ForwardDirection<uint32_t>>;
    using GraphBwd = ripples::Graph<uint32_t, destination_type,
                                    ripples::BackwardDirection<uint32_t>>;
    using vertex_type = typename GraphFwd::vertex_type;

    GraphFwd Gfwd(karate.begin(), karate.end());
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
#pragma omp single
      max_num_threads = omp_get_max_threads();

      trng::lcg64 gen;
      ripples::IMMExecutionRecord R;
      decltype(ripples::IMMConfiguration::worker_to_gpu) map;

      ripples::StreamingRRRGenerator<
          decltype(G), decltype(gen),
          typename ripples::RRRsets<decltype(G)>::iterator,
          ripples::independent_cascade_tag>
          generator(G, gen, R, max_num_threads, 0, map);

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
  }
}
