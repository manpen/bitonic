#pragma once

#include <cstddef>
#include "../call_for_range.hpp"

#include <iostream>

namespace Bitonic {

template <typename SimdOps_>
class SimdSort {
    using SimdOps = SimdOps_;
    using value_type = typename SimdOps::value_type;
    using simd_type  = typename SimdOps::simd_type;

public:
    template <bool kAligned = false>
    static void sort(value_type* begin, value_type* end) {
        const size_t regs = ((end - begin) + SimdOps::kPacking  - 1) / SimdOps::kPacking;

        switch(regs) {
            case  1: load_sort_store< 1, kAligned>(begin); return;
            case  2: load_sort_store< 2, kAligned>(begin); return;
            case  3: load_sort_store< 3, kAligned>(begin); return;
            case  4: load_sort_store< 4, kAligned>(begin); return;
            case  5: load_sort_store< 5, kAligned>(begin); return;
            case  6: load_sort_store< 6, kAligned>(begin); return;
            case  7: load_sort_store< 7, kAligned>(begin); return;
            case  8: load_sort_store< 8, kAligned>(begin); return;
            case  9: load_sort_store< 9, kAligned>(begin); return;

            case 10: load_sort_store<10, kAligned>(begin); return;
            case 11: load_sort_store<11, kAligned>(begin); return;
            case 12: load_sort_store<12, kAligned>(begin); return;
            case 13: load_sort_store<13, kAligned>(begin); return;
            case 14: load_sort_store<14, kAligned>(begin); return;
            case 15: load_sort_store<15, kAligned>(begin); return;
            case 16: load_sort_store<16, kAligned>(begin); return;
            case 17: load_sort_store<17, kAligned>(begin); return;
            case 18: load_sort_store<18, kAligned>(begin); return;
            case 19: load_sort_store<19, kAligned>(begin); return;

            case 20: load_sort_store<20, kAligned>(begin); return;
            case 21: load_sort_store<21, kAligned>(begin); return;
            case 22: load_sort_store<22, kAligned>(begin); return;
            case 23: load_sort_store<23, kAligned>(begin); return;
            case 24: load_sort_store<24, kAligned>(begin); return;
            case 25: load_sort_store<25, kAligned>(begin); return;
            case 26: load_sort_store<26, kAligned>(begin); return;
            case 27: load_sort_store<27, kAligned>(begin); return;
            case 28: load_sort_store<28, kAligned>(begin); return;
            case 29: load_sort_store<29, kAligned>(begin); return;

            case 30: load_sort_store<30, kAligned>(begin); return;
            case 31: load_sort_store<31, kAligned>(begin); return;
            case 32: load_sort_store<32, kAligned>(begin); return;
        }
    }

    template <size_t k, bool kAligned>
    static void load_sort_store(value_type* it) {
        auto packed_it = reinterpret_cast<simd_type*>(it);

        simd_type registers[k];
        load<k, kAligned>(packed_it, registers);
        Sorter<k, true>::sort(registers);
        store<k, kAligned>(packed_it, registers);
    }

private:
    template <size_t kSize, bool kAligned>
    static void load(const simd_type* it, simd_type* x) {
        tlx::call_for_range<0, kSize>([&] (size_t idx) {
            x[idx] = SimdOps::template load<kAligned>(it + idx);
        });
    }

    template <size_t kSize, bool kAligned>
    static void store(simd_type* it, simd_type* x) {
        tlx::call_for_range<0, kSize>([&] (size_t idx) {
            SimdOps::template store<kAligned>(it + idx, x[idx]);
        });
    }

// sorter & merger
    template <size_t kN, bool kAscending>
    struct Sorter;

    // base case for one packed register
    template <bool kAscending>
    struct Sorter<1, kAscending> {
        static void merge(simd_type* vs) {
            auto& v = vs[0];
            constexpr int kSwitch = kAscending ? 0 : 0xff;

            {
                auto tmp = SimdOps::swap_low_high(v);
                auto mi = SimdOps::min(v, tmp);
                auto ma = SimdOps::max(v, tmp);
                v = SimdOps::template blend<0xf0 ^ kSwitch>(mi, ma);
            }

            {
                auto tmp = SimdOps::template shuffle<1, 0, 3, 2>(v);
                auto mi = SimdOps::min(v, tmp);
                auto ma = SimdOps::max(v, tmp);
                v = SimdOps::template blend<0xcc ^ kSwitch>(mi, ma);
            }

            {
                auto tmp = SimdOps::template shuffle<2, 3, 0, 1>(v);
                auto mi = SimdOps::min(v, tmp);
                auto ma = SimdOps::max(v, tmp);
                v = SimdOps::template blend<0xaa ^ kSwitch>(mi, ma);
            }

        }

        static void sort(simd_type* vs) {
            auto& v = vs[0];
            {
                auto tmp = SimdOps::template shuffle<2, 3, 0, 1>(v);
                auto mi = SimdOps::min(v, tmp);
                auto ma = SimdOps::max(v, tmp);
                v = SimdOps::template blend<0x66>(mi, ma);
            }

            {
                auto tmp = SimdOps::template shuffle<1, 0, 3, 2>(v);
                auto mi = SimdOps::min(v, tmp);
                auto ma = SimdOps::max(v, tmp);
                v = SimdOps::template blend<0x3c>(mi, ma);
            }

            {
                auto tmp = SimdOps::template shuffle<2, 3, 0, 1>(v);
                auto mi = SimdOps::min(v, tmp);
                auto ma = SimdOps::max(v, tmp);
                v = SimdOps::template blend<0x5a>(mi, ma);
            }

            merge(vs);
        }
    };

    // recursion for more packed registers
    template <size_t kN, bool kAscending>
    struct Sorter {
        constexpr size_t static greatestPowerOfTwoLessThan(size_t n, size_t pot = 1) {
            return 2*pot < n ? greatestPowerOfTwoLessThan(n, 2*pot) : pot;
        }

        static void merge(simd_type* vs) {
            static constexpr size_t kPOT0 = greatestPowerOfTwoLessThan(kN);
            static constexpr size_t kPOT1 = kN - kPOT0;

            auto flipped_min_max = [](simd_type& v0, simd_type& v1) {
                if constexpr (kAscending) {
                    auto tmp = v1;
                    v1 = SimdOps::max(v0, tmp);
                    v0 = SimdOps::min(v0, tmp);
                } else {
                    auto tmp = v0;
                    v0 = SimdOps::max(v1, tmp);
                    v1 = SimdOps::min(v1, tmp);
                }
            };

            tlx::call_for_range<0, kPOT1>([&] (size_t idx) {
                flipped_min_max(vs[idx], vs[idx + kPOT0]);
            });

            Sorter<kPOT0, kAscending>::merge(vs);
            Sorter<kPOT1, kAscending>::merge(vs + kPOT0);
        }

        static void sort(simd_type* vs) {
            static constexpr size_t kN0 = kN / 2;
            static constexpr size_t kN1 = kN - kN0;

            Sorter<kN0, !kAscending>::sort(vs);
            Sorter<kN1, kAscending>::sort(vs + kN0);

            merge(vs);
        }
    };
};

}