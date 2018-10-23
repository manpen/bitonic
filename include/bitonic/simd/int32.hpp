#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <iostream>
#include <string>

namespace Bitonic {
namespace SimdAdapter {

struct Int32Base {
    using simd_type = __m256i;
    constexpr static size_t kPacking = 8;

    template<int N1, int N2, int N3, int N4>
    static simd_type shuffle(simd_type x) {
        return _mm256_shuffle_epi32(x, _MM_SHUFFLE(N1, N2, N3, N4));
    }

    static simd_type swap_low_high(simd_type x) {
        return _mm256_permute2f128_si256(x, x, 0x01);
    }

    static simd_type mirror(simd_type x) {
        auto tmp = swap_low_high(x);
        return shuffle<0, 1, 2, 3>(tmp);
    }

    template<int N>
    static simd_type blend(simd_type a, simd_type b) {
        return _mm256_blend_epi32(a, b, N);
    }

    template<bool kAligned>
    static simd_type load(const simd_type *it) {
        if constexpr (kAligned) {
            return _mm256_load_si256(it);
        } else {
            return _mm256_loadu_si256(it);
        }
    }

    template<bool kAligned>
    static void store(simd_type *it, const simd_type x) {
        if constexpr (kAligned) {
            _mm256_store_si256(it, x);
        } else {
            _mm256_storeu_si256(it, x);
        }
    }

    static void print(const simd_type x, std::ostream& os = std::cout) {
        os << _mm256_extract_epi32(x, 0) << " "
           << _mm256_extract_epi32(x, 1) << " "
           << _mm256_extract_epi32(x, 2) << " "
           << _mm256_extract_epi32(x, 3) << " "
           << _mm256_extract_epi32(x, 4) << " "
           << _mm256_extract_epi32(x, 5) << " "
           << _mm256_extract_epi32(x, 6) << " "
           << _mm256_extract_epi32(x, 7) << " ";
    }
};

struct SignedInt32 : public Int32Base {
    using value_type = int32_t;

    static simd_type min(simd_type a, simd_type b) {
        return _mm256_min_epi32(a, b);
    }

    static simd_type max(simd_type a, simd_type b) {
        return _mm256_max_epi32(a, b);
    }

    static std::string name() {return "SignedInt32";}
};

struct UnsignedInt32 : public Int32Base {
    using value_type = uint32_t;

    static simd_type min(simd_type a, simd_type b) {
        return _mm256_min_epu32(a, b);
    }

    static simd_type max(simd_type a, simd_type b) {
        return _mm256_max_epu32(a, b);
    }

    static std::string name() {return "UnsignedInt32";}
};

}
}