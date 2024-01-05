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

#ifndef RIPPLES_RRR_SETS_H
#define RIPPLES_RRR_SETS_H

#include <vector>

#ifdef ENABLE_METALL_RRRSETS
#include "metall/metall.hpp"
#include "metall/container/vector.hpp"
#include "metall/container/unordered_map.hpp"
#endif

namespace ripples {
// #if defined ENABLE_METALL_RRRSETS
//   template<typename vertex_type>
//   using RRRsetAllocator = metall::manager::allocator_type<vertex_type>;

//   metall::manager &metall_manager_instance(std::string path) {
//     static metall::manager manager(metall::create_only, path.c_str());
//     return manager;
//   }

// #else
//   template <typename vertex_type>
//   using RRRsetAllocator = std::allocator<vertex_type>;
// #endif

//   //! \brief The Random Reverse Reachability Sets type
//   template <typename GraphTy>
//   using RRRset =
// #ifdef  ENABLE_METALL_RRRSETS
//     metall::container::vector<typename GraphTy::vertex_type,
//                               RRRsetAllocator<typename GraphTy::vertex_type>>;
// #else
//   std::vector<typename GraphTy::vertex_type,
//               RRRsetAllocator<typename GraphTy::vertex_type>>;
// #endif
//   // template <typename GraphTy>
//   // using RRRsets = std::vector<RRRset<GraphTy>>;
#if defined ENABLE_METALL_RRRSETS
template<typename vertex_type>
using RRRsetAllocator = metall::manager::allocator_type<vertex_type>;
#else
template <typename vertex_type>
using RRRsetAllocator = std::allocator<vertex_type>;
#endif

//! \brief The Random Reverse Reachability Sets type
template <typename GraphTy>
using RRRset =
#ifdef  ENABLE_METALL_RRRSETS
    metall::container::vector<typename GraphTy::vertex_type,
                              RRRsetAllocator<typename GraphTy::vertex_type>>;
    template<typename GraphTy>
using RRRsetsAllocator = metall::container::scoped_allocator_adaptor<
    metall::manager::allocator_type<RRRset<GraphTy>>>;
    template <typename GraphTy>
    using RRRsets = metall::container::vector<RRRset<GraphTy>, RRRsetsAllocator<GraphTy>>;
#else
    std::vector<typename GraphTy::vertex_type,
                              RRRsetAllocator<typename GraphTy::vertex_type>>;
    template <typename GraphTy>
    using RRRsets = std::vector<RRRset<GraphTy>>;
#endif
}

#endif
