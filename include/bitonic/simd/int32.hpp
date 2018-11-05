#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <ostream>
#include <iomanip>
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

    static simd_type index_sequence() {
        return _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    }

    static simd_type shift_left(simd_type v, int i) {
        return _mm256_slli_epi32(v, i);
    }

    static simd_type bitwise_or(simd_type a, simd_type b) {
        return _mm256_or_si256(a, b);
    }

    static simd_type bitwise_and(simd_type a, simd_type b) {
        return _mm256_and_si256(a, b);
    }

    static simd_type add(simd_type a, simd_type b) {
        return _mm256_add_epi32(a, b);
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

    template <typename Callback>
    static simd_type load_callback(Callback&& cb) {
        return _mm256_setr_epi32(cb(0), cb(1), cb(2), cb(3), cb(4), cb(5), cb(6), cb(7));
    }

    template <typename Callback, typename value_type>
    static simd_type partial_load_callback(Callback&& cb, int len, const value_type empty_value) {
        return _mm256_setr_epi32(
                       cb(0)              ,
            len > 1 ? cb(1) : empty_value,
            len > 2 ? cb(2) : empty_value,
            len > 3 ? cb(3) : empty_value,
            len > 4 ? cb(4) : empty_value,
            len > 5 ? cb(5) : empty_value,
            len > 6 ? cb(6) : empty_value,
            len > 7 ? cb(7) : empty_value);
    }

    template<typename value_type>
    static simd_type partial_load(const simd_type *begin, const size_t length, const value_type empty_value) {
        const simd_type index = index_sequence();
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


    template <typename Callback>
    static void store_callback(Callback&& cb, simd_type x) {
        cb(0, _mm256_extract_epi32(x, 0));
        cb(1, _mm256_extract_epi32(x, 1));
        cb(2, _mm256_extract_epi32(x, 2));
        cb(3, _mm256_extract_epi32(x, 3));

        cb(4, _mm256_extract_epi32(x, 4));
        cb(5, _mm256_extract_epi32(x, 5));
        cb(6, _mm256_extract_epi32(x, 6));
        cb(7, _mm256_extract_epi32(x, 7));
    }

    template <typename Callback>
    static void partial_store_callback(Callback&& cb, int len, const simd_type x) {
                     cb(0, _mm256_extract_epi32(x, 0));
        if (len > 1) cb(1, _mm256_extract_epi32(x, 1)); else return;
        if (len > 2) cb(2, _mm256_extract_epi32(x, 2)); else return;
        if (len > 3) cb(3, _mm256_extract_epi32(x, 3)); else return;
        if (len > 4) cb(4, _mm256_extract_epi32(x, 4)); else return;
        if (len > 5) cb(5, _mm256_extract_epi32(x, 5)); else return;
        if (len > 6) cb(6, _mm256_extract_epi32(x, 6)); else return;
        if (len > 7) cb(7, _mm256_extract_epi32(x, 7)); else return;
    }



    static void print(const simd_type x, std::ostream& os = std::cout) {
        std::stringstream ss;
        ss << std::hex << _mm256_extract_epi32(x, 0) << " "
           << std::hex << _mm256_extract_epi32(x, 1) << " "
           << std::hex << _mm256_extract_epi32(x, 2) << " "
           << std::hex << _mm256_extract_epi32(x, 3) << " "
           << std::hex << _mm256_extract_epi32(x, 4) << " "
           << std::hex << _mm256_extract_epi32(x, 5) << " "
           << std::hex << _mm256_extract_epi32(x, 6) << " "
           << std::hex << _mm256_extract_epi32(x, 7);
        os << ss.str();
    }
};

struct SignedInt32 : public Int32Base {
    using value_type = int32_t;

    static simd_type broadcast(value_type a) {
        return _mm256_set1_epi32(a);
    }

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

    static simd_type broadcast(value_type a) {
        return _mm256_set1_epi32(a);
    }

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