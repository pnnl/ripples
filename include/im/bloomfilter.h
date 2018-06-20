//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef IM_BLOOMFILTER_H
#define IM_BLOOMFILTER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <functional>
#include <iterator>

#include "boost/dynamic_bitset.hpp"


namespace im {

//! \brief Bloom filter.
//!
//! \targs T The type of the elements of the set.
//! \targs Hash A functor computing hashes.
template <typename T, typename Hash = std::hash<T>,
          bool = std::is_integral<T>::value>
class bloomfilter {
  class BloomfilterIterator;

 public:
  using value_type = T;
  using size_type = size_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = BloomfilterIterator;

  bloomfilter() = default;

  //! Default constructor
  //! \param m The number of bits for the bloomfilter
  //! \param k The number of hash function used by the bloomfilter
  bloomfilter(size_t m, size_t k)
      : bitset_(m, false),
        hash_(k) {
    std::random_device rd;
    for (auto & key : hash_) {
      key = rd();
    }
  }

  //! Copy constructor
  bloomfilter(const bloomfilter &other) = default;

  //! Move constructor
  bloomfilter(bloomfilter &&other) = default;

  //! Copy assignment operator
  bloomfilter &operator=(const bloomfilter &other) = default;

  //! Move assignment operator
  bloomfilter &operator=(bloomfilter &&other) = default;

  //! \brief Empty test.
  //! Test whether container is empty.
  bool empty() const noexcept { return bitset_.none(); }

  //! \brief The size of the filter.
  //! \return the size in bits of the bloom filter.
  size_type size() const { return bitset_.size(); }

  //! \bierf Estimate the number of elements in the filter.
  //! \return an estimate of the number of elements in the filter.
  size_type count() const {
    size_type numberOfBitSet = bitset_.count();

    ssize_t ratio = size() / hash_.size();
    float v = 1.0f - (float(numberOfBitSet) / size());
    float logV = -std::log(v);

    return numberOfBitSet ? std::max<size_type>(1, ratio * logV) : 0;
  }

  //! \brief Insert an element in the set.
  //! \param value The element to be inserted in the bloomfilter.
  void insert(const value_type value) {
    Hash h;
    size_t hash = h(value);
    for (auto key : hash_) {
      size_t pos = (hash ^ key) % bitset_.size();
      bitset_.set(pos);
    }
  }

  //! \brief Check if an element is in the set.
  //! \param value The element to search in the set.
  bool find(const value_type value) const noexcept {
    bool result = true;
    Hash h;
    size_t hash = h(value);
    for (auto itr = std::begin(hash_), end = std::end(hash_);
         result && itr != end; ++itr) {
      result = bitset_[(hash ^ *itr) % bitset_.size()];
    }
    return result;
  }

 private:
  boost::dynamic_bitset<uint64_t> bitset_;
  std::vector<size_t> hash_;
};

}  // namespace im

#endif /* IM_BLOOMFILTER_H */
