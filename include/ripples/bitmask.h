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

#ifndef RIPPLES_BITMASK_H
#define RIPPLES_BITMASK_H

#include <cstddef>
#include <memory>

namespace ripples {
template <typename BaseTy>
class Bitmask {
 public:
  Bitmask() = default;
  Bitmask(const Bitmask &O)
      : size_(O.size_),
        data_size_(O.data_size_),
        data_(new BaseTy[data_size_]) {
    std::memcpy(data_.get(), O.data_.get(), data_size_);
  }

  Bitmask(Bitmask &&) = default;

  explicit Bitmask(size_t num_bits)
      : size_(num_bits),
        data_size_((size_ / (8 * sizeof(BaseTy)) + 1)),
        data_(new BaseTy[data_size_]) {
    std::memset(data_.get(), 0, data_size_);
  }

  Bitmask &operator=(const Bitmask &O) {
    size_ = O.size_;
    data_size_ = O.data_size_;
    data_ = new BaseTy[data_size_];
    std::memcpy(data_.get(), O.data_.get(), data_size_);
  }
  Bitmask &operator=(Bitmask &&) = default;

  void set(size_t i) {
    BaseTy m = 1 << (i % (8 * sizeof(BaseTy)));
    data_[i / (8 * sizeof(BaseTy))] |= m;
  }
  bool get(size_t i) const {
    BaseTy m = 1 << (i % (8 * sizeof(BaseTy)));
    return data_[i / (8 * sizeof(BaseTy))] && m;
  }

  size_t popcount() const {
    size_t count = 0;
    for (size_t i = 0; i < data_size_; ++i) {
      count += __builtin_popcount(data_[i]);
    }
    return count;
  }

  BaseTy *data() const { return data_.get(); }
  size_t bytes() const { return data_size_ * sizeof(BaseTy); }
  size_t size() const { return size_; }

 private:
  size_t size_;
  size_t data_size_;
  std::unique_ptr<BaseTy[]> data_;
};

}  // namespace ripples

#endif
