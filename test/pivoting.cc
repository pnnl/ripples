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
#include <iostream>
#include <iterator>
#include <set>
#include <vector>

#include "catch2/catch.hpp"
#include "ripples/find_most_influential.h"
#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"

SCENARIO("Swap two sequences", "[pivoting]") {
  GIVEN("Two random sequence of elements") {
    std::vector<uint32_t> A(50);
    std::vector<uint32_t> B(50);

    trng::lcg64 generator;
    trng::uniform_int_dist rnd_vertex(0, 100);

    std::generate(
        A.begin(), A.end(), [&]() -> auto { return rnd_vertex(generator); });

    std::generate(
        B.begin(), B.end(), [&]() -> auto { return rnd_vertex(generator); });
    WHEN("I swap squentially two copies of A and B") {
      auto Acopy(A);
      auto Bcopy(B);

      ripples::swap_ranges(Acopy.begin(), Acopy.end(), Bcopy.begin(),
                           ripples::sequential_tag{});

      THEN("The copies are now swapped") {
        REQUIRE(Acopy == B);
        REQUIRE(Bcopy == A);
      }
    }

    WHEN("I swap in parallel two copies of A and B") {
      auto Acopy(A);
      auto Bcopy(B);

      ripples::swap_ranges(Acopy.begin(), Acopy.end(), Bcopy.begin(),
                           ripples::omp_parallel_tag{});

      THEN("The copies are now swapped") {
        REQUIRE(Acopy == B);
        REQUIRE(Bcopy == A);
      }
    }
  }
}

SCENARIO("RRR set can grouped in covered and uncovered", "[pivoting]") {
  GIVEN("A random sequence of RRR sets") {
    std::vector<std::set<uint32_t>> rrr_sets(50);

    trng::lcg64 generator;
    trng::uniform_int_dist rnd_vertex(0, 34);

    for (auto& s : rrr_sets) {
      for (size_t i = 0; i < 5; ++i) {
        s.insert(rnd_vertex(generator));
      }
    }

    uint32_t v = rnd_vertex(generator);
    auto cmp = [=](const std::set<uint32_t>& a) -> auto {
      return a.find(v) == a.end();
    };
    WHEN("the sequence is partitioned sequentially") {
      auto cmp = [=](const std::set<uint32_t>& a) -> auto {
        return a.find(v) == a.end();
      };
      auto pivot = ripples::partition(rrr_sets.begin(), rrr_sets.end(), cmp,
                                      ripples::sequential_tag{});
      THEN("The sequence becomes partitioned") {
        REQUIRE(rrr_sets.size() == 50);
        REQUIRE(pivot <= rrr_sets.end());
        REQUIRE(pivot >= rrr_sets.begin());

        REQUIRE(std::all_of(rrr_sets.begin(), pivot, cmp));
        REQUIRE(std::none_of(pivot, rrr_sets.end(), cmp));
      }
    }
    WHEN("the sequence is partitioned in parallel with OpenMP") {
      auto pivot = ripples::partition(rrr_sets.begin(), rrr_sets.end(), cmp,
                                      ripples::omp_parallel_tag{});

      THEN("The sequence becomes partitioned") {
        REQUIRE(rrr_sets.size() == 50);
        REQUIRE(pivot <= rrr_sets.end());
        REQUIRE(pivot >= rrr_sets.begin());

        REQUIRE(std::all_of(rrr_sets.begin(), pivot, cmp));
        REQUIRE(std::none_of(pivot, rrr_sets.end(), cmp));
      }
    }
  }
}
