/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#include "check.h"
#include "utils.h"
#include <cassert>
#include <cstdint>

namespace dyn_emb {

// MurmurHash3_32 implementation from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
template <typename Key, uint32_t m_seed = 0> struct MurmurHash3_32 {
  using argument_type = Key;
  using result_type = uint32_t;

  /*__forceinline__
  __host__ __device__
  MurmurHash3_32() : m_seed( 0 ) {}*/

  __forceinline__ __host__ __device__ static uint32_t rotl32(uint32_t x,
                                                             int8_t r) {
    return (x << r) | (x >> (32 - r));
  }

  __forceinline__ __host__ __device__ static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Combines two hash values into a new single hash value. Called
   * repeatedly to create a hash value from several variables.
   * Taken from the Boost hash_combine function
   * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
   *
   * @Param lhs The first hash value to combine
   * @Param rhs The second hash value to combine
   *
   * @Returns A hash value that intelligently combines the lhs and rhs hash
   * values
   */
  /* ----------------------------------------------------------------------------*/
  __host__ __device__ static result_type hash_combine(result_type lhs,
                                                      result_type rhs) {
    result_type combined{lhs};

    combined ^= rhs + 0x9e3779b9 + (combined << 6) + (combined >> 2);

    return combined;
  }

  __forceinline__ __host__ __device__ static result_type hash(const Key &key) {
    constexpr int len = sizeof(argument_type);
    const uint8_t *const data = (const uint8_t *)&key;
    constexpr int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    const uint32_t *const blocks = (const uint32_t *)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i]; // getblock32(blocks,i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    const uint8_t *tail = (const uint8_t *)(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }

  __host__ __device__ __forceinline__ result_type
  operator()(const Key &key) const {
    return this->hash(key);
  }
};

template <typename key_type, typename index_type, index_type result>
struct Fix_Hash {
  using result_type = index_type;

  __forceinline__ __host__ __device__ static index_type
  hash(const key_type &key) {
    return result;
  }
};

template <typename key_type, typename result_type> struct Mod_Hash {
  __forceinline__ __host__ __device__ static result_type
  hash(const key_type &key) {
    return (result_type)key;
  }
};

// The unique op
template <typename KeyType, typename CounterType, KeyType empty_key,
          CounterType empty_val, typename hasher = MurmurHash3_32<KeyType>>
class unique_op {
public:
  // Ctor
  unique_op(KeyType *keys, CounterType *vals, CounterType *counter,
            const size_t capacity, const CounterType init_counter_val = 0);

  // Get the max capacity of unique op obj
  size_t get_capacity() const;

  // Unique operation
  void unique(const KeyType *d_key, const uint64_t len,
              CounterType *d_output_index, KeyType *d_unique_key,
              CounterType *d_output_counter, cudaStream_t stream,
              CounterType *offset_ptr = nullptr);

  void reset_capacity(KeyType *keys, CounterType *vals, const size_t capacity,
                      cudaStream_t stream);

  // Clear operation
  void clear(cudaStream_t stream);

private:
  static const size_t BLOCK_SIZE_ = 64;
  //   std::shared_ptr<CoreResourceManager> core_;
  // Capacity
  size_t capacity_;
  KeyType empty_key_;
  CounterType empty_val_;
  // Init counter value
  CounterType init_counter_val_;

  // Keys and vals buffer
  KeyType *keys_;
  CounterType *vals_;

  // Counter for value index
  CounterType *counter_;
};

} // namespace dyn_emb
