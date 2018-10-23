#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <ostream>
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

    template<bool kAligned, bool kStream>
    static simd_type load(const simd_type *it) {
        static_assert(!(kStream && !kAligned), "Streaming stores require aligned access");

        if constexpr (kStream) {
            return _mm256_stream_load_si256(it);
        } else {
            if constexpr (kAligned) {
                return _mm256_load_si256(it);
            } else {
                return _mm256_loadu_si256(it);
            }
        }
    }

    template<typename value_type>
    static simd_type partial_load(const simd_type *begin, const size_t length, const value_type empty_value) {
        const simd_type index = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
        const simd_type len = _mm256_set1_epi32(length);
        auto mask = _mm256_cmpgt_epi32(len, index); // gives 1 where to load
        auto value = _mm256_maskload_epi32(reinterpret_cast<const int*>(begin), mask);

        if (empty_value) {
            auto tmp = _mm256_set1_epi32(-1);
            mask = _mm256_xor_si256(mask, tmp);
            auto empty = _mm256_and_si256(mask, _mm256_set1_epi32(empty_value));
            value = _mm256_or_si256(value, empty);
        }

        return value;
    };


    template<bool kAligned, bool kStream>
    static void store(simd_type *it, const simd_type x) {
        static_assert(!(kStream && !kAligned), "Streaming stores require aligned access");

        if constexpr (kStream) {
            _mm256_stream_si256(it, x);
        } else {
            if constexpr (kAligned) {
                _mm256_store_si256(it, x);
            } else {
                _mm256_storeu_si256(it, x);
            }
        }
    }

    static void partial_store(simd_type *begin, const size_t length, const simd_type x) {
        const simd_type index = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
        const simd_type len = _mm256_set1_epi32(length);
        auto mask = _mm256_cmpgt_epi32(len, index); // gives 1 where to load
        _mm256_maskstore_epi32(reinterpret_cast<int*>(begin), mask, x);
    };

    static void print(const simd_type x, std::ostream& os = std::cout) {
        os << _mm256_extract_epi32(x, 0) << " "
           << _mm256_extract_epi32(x, 1) << " "
           << _mm256_extract_epi32(x, 2) << " "
           << _mm256_extract_epi32(x, 3) << " "
           << _mm256_extract_epi32(x, 4) << " "
           << _mm256_extract_epi32(x, 5) << " "
           << _mm256_extract_epi32(x, 6) << " "
           << _mm256_extract_epi32(x, 7);
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