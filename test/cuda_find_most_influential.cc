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


#include <vector>

#include "catch2/catch.hpp"

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/cuda/find_most_influential.h"
#include "ripples/find_most_influential.h"

using namespace ripples;

SCENARIO("Count vertex occurrencies on GPU", "[count]") {
  GIVEN("RRR sets") {
    std::vector<std::vector<uint32_t>> rrr_sets(1000);

    for (size_t i = 0; i < rrr_sets.size(); ++i) {
      if (i % 2)
        rrr_sets[i] = { 2, 3 };
      else
        rrr_sets[i] = { 2, 5 };
    }

    for (size_t i = 0; i < 100; ++i) {
      switch (i % 3) {
        case 0:
          rrr_sets.push_back({3, 4, 5});
          break;
        case 1:
          rrr_sets.push_back({1, 3});
          break;
        case 2:
          rrr_sets.push_back({1, 5});
          break;
      }
    }

    WHEN("RRR sets are moved to device memory") {
        uint32_t * d_Counters;
        cuda_malloc(reinterpret_cast<void**>(&d_Counters), sizeof(uint32_t) * 6);
        cuda_memset(d_Counters, 0, sizeof(uint32_t) * 6);
        uint32_t * d_rrr_index;
        uint32_t * d_rrr_sets;
        cuda_malloc(reinterpret_cast<void**>(&d_rrr_index), sizeof(uint32_t) * (2000 + 66 *2 + 33* 3));
        cuda_malloc(reinterpret_cast<void**>(&d_rrr_sets), sizeof(uint32_t) * (2000 + 66 *2 + 33 * 3));

        uint32_t * d_rrr_mask;
        cuda_malloc(reinterpret_cast<void**>(&d_rrr_mask), sizeof(uint32_t) * 1100);


        THEN("") {
          MoveRRRSets(rrr_sets.begin(), rrr_sets.end(),
                      d_rrr_index, d_rrr_sets, (2000 + 66 *2 + 33 * 3), (2000 + 66 *2 + 33 * 3));

          std::vector<uint32_t> rrr_sets2((2000 + 66 *2 + 33 * 3));
          cuda_d2h(rrr_sets2.data(), d_rrr_sets, sizeof(uint32_t) * (2000 + 66 *2 + 33 * 3));
          for (auto v : rrr_sets2)
            REQUIRE((v == 1 || v == 2 || v == 3 || v == 4 || v == 5));

          CudaCountOccurrencies(d_Counters, d_rrr_sets, 2000 + 66 * 2 + 33 * 3, 6);

          std::vector<uint32_t> counters(6);

          cuda_d2h(counters.data(), d_Counters, sizeof(uint32_t) * 6);

          REQUIRE(counters[0] == 0);
          REQUIRE(counters[1] == 66);
          REQUIRE(counters[2] == 1000);
          REQUIRE(counters[3] == 566);
          REQUIRE(counters[4] == 33);
          REQUIRE(counters[5] == 566);

          CudaUpdateCounters(2000 + 66 * 2 + 33 * 3, d_rrr_index, d_rrr_sets, d_rrr_mask, d_Counters, 6, 2);

          std::vector<uint32_t> mask(1100);
          cuda_d2h(reinterpret_cast<void *>(mask.data()),
                   reinterpret_cast<void *>(d_rrr_mask),
                   sizeof(uint32_t) * mask.size());


          for (size_t i = 0; i < 1100; ++i) {
            if (i < 1000)
              REQUIRE(mask[i] == 1);
            else
              REQUIRE(mask[i] == 0);
          }

          cuda_d2h(counters.data(), d_Counters, sizeof(mask_word_t) * 6);
          REQUIRE(counters[0] == 0);
          REQUIRE(counters[1] == 66);
          REQUIRE(counters[2] == 0);
          REQUIRE(counters[3] == 66);
          REQUIRE(counters[4] == 33);
          REQUIRE(counters[5] == 66);

          CudaUpdateCounters(2000 + 66 * 2 + 33 * 3, d_rrr_index, d_rrr_sets, d_rrr_mask, d_Counters, 6, 1);

          cuda_d2h(reinterpret_cast<void *>(mask.data()),
                   reinterpret_cast<void *>(d_rrr_mask),
                   sizeof(uint32_t) * mask.size());

          for (size_t i = 0; i < 1000; ++i) {
              REQUIRE(mask[i] == 1);
          }

          for (size_t i = 0; i < 100; ++i) {
            switch (i % 3) {
              case 0:
                REQUIRE(mask[1000 + i] == 0);
                break;
              case 1:
                REQUIRE(mask[1000 + i] == 1);
                break;
              case 2:
                REQUIRE(mask[1000 + i] == 1);
                break;
            }
          }

          cuda_d2h(counters.data(), d_Counters, sizeof(mask_word_t) * 6);
          REQUIRE(counters[0] == 0);
          REQUIRE(counters[1] == 0);
          REQUIRE(counters[2] == 0);
          REQUIRE(counters[3] == 33);
          REQUIRE(counters[4] == 33);
          REQUIRE(counters[5] == 33);
        }

        cuda_free(reinterpret_cast<void*>(d_Counters));
        cuda_free(reinterpret_cast<void*>(d_rrr_index));
        cuda_free(reinterpret_cast<void*>(d_rrr_sets));
    }
  }
}
