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

#include <algorithm>

#include "catch2/catch.hpp"
#include "ripples/graph.h"

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

std::vector<uint32_t> communityVector{0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0,
                                      0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 3,
                                      3, 3, 2, 3, 3, 2, 2, 3, 2, 2};

SCENARIO("Graph Build", "[graph build]") {
  GIVEN("The Karate Graph Edge List") {
    using destination_type = ripples::WeightedDestination<uint32_t, float>;
    using GraphFwd = ripples::Graph<uint32_t, destination_type,
                                    ripples::ForwardDirection<uint32_t>>;
    using GraphBwd = ripples::Graph<uint32_t, destination_type,
                                    ripples::BackwardDirection<uint32_t>>;
    using vertex_type = typename GraphFwd::vertex_type;

    auto b = karate.begin();
    auto e = karate.end();

    WHEN("I build the Karate Graph from its edge list") {
      GraphFwd G(b, e, true);

      THEN("G has 34 vertices and 78 edges") {
        REQUIRE(G.num_nodes() == 34);
        REQUIRE(G.num_edges() == 78);
      }

      THEN("All the edges exist in the edge list") {
        for (vertex_type v = 0; v < G.num_nodes(); ++v) {
          for (auto d : G.neighbors(v)) {
            auto itr = std::find(
                b, e, EdgeT{G.convertID(v), G.convertID(d.vertex), d.weight});
            REQUIRE(itr != e);
          }
        }
      }

      THEN("All the edges in the edge list exist in the Graph") {
        for (const auto e : karate) {
          auto n = G.neighbors(G.transformID(e.source));
          auto itr = std::find(n.begin(), n.end(),
                               typename GraphFwd::edge_type{
                                   G.transformID(e.destination), e.weight});
          REQUIRE(itr != n.end());
        }
      }
    }
  }
}

SCENARIO("Extract Communities", "[communities]") {
  GIVEN("The Karate Graph") {
    using destination_type = ripples::WeightedDestination<uint32_t, float>;
    using GraphFwd = ripples::Graph<uint32_t, destination_type,
                                    ripples::ForwardDirection<uint32_t>>;
    using GraphBwd = ripples::Graph<uint32_t, destination_type,
                                    ripples::BackwardDirection<uint32_t>>;
    using vertex_type = typename GraphFwd::vertex_type;

    GraphFwd Gf(karate.begin(), karate.end(), true);

    REQUIRE(Gf.num_nodes() == 34);

    WHEN("I extract the community sub-graphs") {
      const auto communities =
          ripples::getCommunitiesSubgraphs<GraphBwd>(Gf, communityVector);

      THEN("I get four sub-graphs") {
        REQUIRE(Gf.num_nodes() == 34);
        REQUIRE(communities.size() == 4);

        REQUIRE(communities[0].num_nodes() == 12);
        REQUIRE(communities[1].num_nodes() == 5);
        REQUIRE(communities[2].num_nodes() == 11);
        REQUIRE(communities[3].num_nodes() == 6);
      }
    }
  }
}
