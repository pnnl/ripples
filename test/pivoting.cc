//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <set>

#include "catch/catch.hpp"
#include "im/find_most_influential.h"
#include "trng/lcg64.hpp"
#include "trng/uniform_int_dist.hpp"


SCENARIO("RRR set can grouped in covered and uncovered", "[pivoting]") {
  GIVEN("A random sequence of RRR sets") {
    
    std::vector<std::set<uint32_t>> rrr_sets(50);

    trng::lcg64 generator;
    trng::uniform_int_dist rnd_vertex(0, 34);

    for (auto & s : rrr_sets) {
      for (size_t i = 0; i < 5; ++i) {
        s.insert(rnd_vertex(generator));
      }
    }

    uint32_t v = rnd_vertex(generator);
    auto cmp = [=](const std::set<uint32_t> & a) -> auto {
                 return a.find(v) == a.end();
               };
    WHEN("the sequence is partitioned sequentially") {
      auto cmp = [=](const std::set<uint32_t> & a) -> auto {
                   return a.find(v) == a.end();
                 };
      auto pivot = im::partition(rrr_sets.begin(), rrr_sets.end(),
                                 cmp, im::sequential_tag{});
      THEN("The sequence becomes partitioned") {
        REQUIRE(rrr_sets.size() == 50);
        REQUIRE(pivot <= rrr_sets.end());
        REQUIRE(pivot >= rrr_sets.begin());

        REQUIRE(std::all_of(rrr_sets.begin(), pivot, cmp));
        REQUIRE(std::none_of(pivot, rrr_sets.end(), cmp));
      }
    }
    WHEN("the sequence is partitioned in parallel with OpenMP") {
      auto pivot = im::partition(rrr_sets.begin(), rrr_sets.end(),
                                 cmp, im::omp_parallel_tag{});

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
