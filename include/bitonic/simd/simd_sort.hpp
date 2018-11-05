#pragma once

#include <cstddef>
#include "../call_for_range.hpp"

#include <limits>

#include <tlx/meta.hpp>

namespace Bitonic {

template <typename SimdOps_>
class SimdSort {
    using SimdOps = SimdOps_;
    using value_type = typename SimdOps::value_type;
    using simd_type  = typename SimdOps::simd_type;

public:
    template <bool kAligned = false, bool kStream = false>
    static void sort(value_type* begin, value_type* end) {
        const size_t regs = ((end - begin) + SimdOps::kPacking  - 1) / SimdOps::kPacking;

        switch(regs) {
            case  1: load_sort_store< 1, kAligned, kStream>(begin, end); return;
            case  2: load_sort_store< 2, kAligned, kStream>(begin, end); return;
            case  3: load_sort_store< 3, kAligned, kStream>(begin, end); return;
            case  4: load_sort_store< 4, kAligned, kStream>(begin, end); return;
            case  5: load_sort_store< 5, kAligned, kStream>(begin, end); return;
            case  6: load_sort_store< 6, kAligned, kStream>(begin, end); return;
            case  7: load_sort_store< 7, kAligned, kStream>(begin, end); return;
            case  8: load_sort_store< 8, kAligned, kStream>(begin, end); return;
            case  9: load_sort_store< 9, kAligned, kStream>(begin, end); return;

            case 10: load_sort_store<10, kAligned, kStream>(begin, end); return;
            case 11: load_sort_store<11, kAligned, kStream>(begin, end); return;
            case 12: load_sort_store<12, kAligned, kStream>(begin, end); return;
            case 13: load_sort_store<13, kAligned, kStream>(begin, end); return;
            case 14: load_sort_store<14, kAligned, kStream>(begin, end); return;
            case 15: load_sort_store<15, kAligned, kStream>(begin, end); return;
            case 16: load_sort_store<16, kAligned, kStream>(begin, end); return;
            case 17: load_sort_store<17, kAligned, kStream>(begin, end); return;
            case 18: load_sort_store<18, kAligned, kStream>(begin, end); return;
            case 19: load_sort_store<19, kAligned, kStream>(begin, end); return;

            case 20: load_sort_store<20, kAligned, kStream>(begin, end); return;
            case 21: load_sort_store<21, kAligned, kStream>(begin, end); return;
            case 22: load_sort_store<22, kAligned, kStream>(begin, end); return;
            case 23: load_sort_store<23, kAligned, kStream>(begin, end); return;
            case 24: load_sort_store<24, kAligned, kStream>(begin, end); return;
            case 25: load_sort_store<25, kAligned, kStream>(begin, end); return;
            case 26: load_sort_store<26, kAligned, kStream>(begin, end); return;
            case 27: load_sort_store<27, kAligned, kStream>(begin, end); return;
            case 28: load_sort_store<28, kAligned, kStream>(begin, end); return;
            case 29: load_sort_store<29, kAligned, kStream>(begin, end); return;

            case 30: load_sort_store<30, kAligned, kStream>(begin, end); return;
            case 31: load_sort_store<31, kAligned, kStream>(begin, end); return;
            case 32: load_sort_store<32, kAligned, kStream>(begin, end); return;
        }
    }


    template <typename CallbackLoad, typename CallbackStore>
    static void sort(CallbackLoad cbl, CallbackStore cbw, size_t n) {
        const size_t regs = (n + SimdOps::kPacking  - 1) / SimdOps::kPacking;

        switch(regs) {
            case  1: sort_callback<1>(cbl, cbw, n); return;
            case  2: sort_callback<2>(cbl, cbw, n); return;
            case  3: sort_callback<3>(cbl, cbw, n); return;
            case  4: sort_callback<4>(cbl, cbw, n); return;
            case  5: sort_callback<5>(cbl, cbw, n); return;
            case  6: sort_callback<6>(cbl, cbw, n); return;
            case  7: sort_callback<7>(cbl, cbw, n); return;
            case  8: sort_callback<8>(cbl, cbw, n); return;
            case  9: sort_callback<9>(cbl, cbw, n); return;

            case 10: sort_callback<10>(cbl, cbw, n); return;
            case 11: sort_callback<11>(cbl, cbw, n); return;
            case 12: sort_callback<12>(cbl, cbw, n); return;
            case 13: sort_callback<13>(cbl, cbw, n); return;
            case 14: sort_callback<14>(cbl, cbw, n); return;
            case 15: sort_callback<15>(cbl, cbw, n); return;
            case 16: sort_callback<16>(cbl, cbw, n); return;
            case 17: sort_callback<17>(cbl, cbw, n); return;
            case 18: sort_callback<18>(cbl, cbw, n); return;
            case 19: sort_callback<19>(cbl, cbw, n); return;

            case 20: sort_callback<20>(cbl, cbw, n); return;
            case 21: sort_callback<21>(cbl, cbw, n); return;
            case 22: sort_callback<22>(cbl, cbw, n); return;
            case 23: sort_callback<23>(cbl, cbw, n); return;
            case 24: sort_callback<24>(cbl, cbw, n); return;
            case 25: sort_callback<25>(cbl, cbw, n); return;
            case 26: sort_callback<26>(cbl, cbw, n); return;
            case 27: sort_callback<27>(cbl, cbw, n); return;
            case 28: sort_callback<28>(cbl, cbw, n); return;
            case 29: sort_callback<29>(cbl, cbw, n); return;

            case 30: sort_callback<30>(cbl, cbw, n); return;
            case 31: sort_callback<31>(cbl, cbw, n); return;
            case 32: sort_callback<32>(cbl, cbw, n); return;
        }
    }


    template <size_t k, bool kAligned, bool kStream>
    static void load_sort_store(value_type* begin, value_type* end) {
        auto packed_it = reinterpret_cast<simd_type*>(begin);
        const auto partial_size = (end - begin) % SimdOps::kPacking;

        simd_type registers[k];

        if ( !partial_size ) {
            load<k, kAligned, kStream>(packed_it, registers);
        } else {
            registers[k-1] = SimdOps::partial_load(packed_it + (k-1), partial_size,
                                                   std::numeric_limits<value_type>::max());
            load<k-1, kAligned, kStream>(packed_it, registers);
        }


        constexpr int shift = ::tlx::Log2<k * SimdOps::kPacking>::ceil;
        {
            auto index_seq = SimdOps::index_sequence();
            const auto pack_bc = SimdOps::broadcast(SimdOps::kPacking);

            tlx::call_for_range<0, k>([&](size_t idx) {
                auto tmp = SimdOps::shift_left(registers[idx], shift);
                tmp = SimdOps::bitwise_or(tmp, index_seq);
                registers[idx] = tmp;

                index_seq = SimdOps::add(index_seq, pack_bc);
            });
        }

        Sorter<k, true>::sort(registers);

        {
            auto mask = SimdOps::broadcast((1 << shift) - 1);

            tlx::call_for_range<0, k>([&](size_t idx) {
                registers[idx] = SimdOps::bitwise_and(registers[idx], mask);
            });
        }

        if ( !partial_size ) {
            store<k, kAligned, kStream>(packed_it, registers);
        } else {
            store<k-1, kAligned, kStream>(packed_it, registers);
            SimdOps::partial_store(packed_it + (k-1), partial_size, registers[k-1]);
        }
    }


    template <size_t k, typename CallbackLoad, typename CallbackStore>
    static void sort_callback(CallbackLoad cbl, CallbackStore cbw, size_t n) {
        const auto partial_size = n % SimdOps::kPacking;

        simd_type registers[k];

        constexpr int shift = ::tlx::Log2<k * SimdOps::kPacking>::ceil;
        {
            auto index_seq = SimdOps::index_sequence();
            const auto pack_bc = SimdOps::broadcast(SimdOps::kPacking);

            auto add_index = [&] (simd_type x) {
                auto tmp = SimdOps::shift_left(x, shift);
                x = SimdOps::bitwise_or(tmp, index_seq);

                index_seq = SimdOps::add(index_seq, pack_bc);
                return x;
            };

            if (!partial_size) {
                tlx::call_for_range<0, k>([&](size_t idx) {
                    registers[idx] = add_index(SimdOps::load_callback([idx, &cbl](auto i) { return cbl(idx * SimdOps::kPacking + i); }));
                });
            } else {
                tlx::call_for_range<0, k - 1>([&](size_t idx) {
                    registers[idx] = add_index(SimdOps::load_callback([idx, &cbl](auto i) { return cbl(idx * SimdOps::kPacking + i); }));
                });
                registers[k - 1] = add_index(SimdOps::partial_load_callback([&cbl](auto i) { return cbl((k - 1) * SimdOps::kPacking + i); },
                                                                  partial_size, std::numeric_limits<value_type>::max()));
            }
        }

        Sorter<k, true>::sort(registers);

        {
            auto mask = SimdOps::broadcast((1 << shift) - 1);

            if ( !partial_size ) {
                tlx::call_for_range<0, k>([&] (size_t idx) {
                    SimdOps::store_callback([&] (auto i, auto x) {cbw(idx * SimdOps::kPacking + i, x);},  SimdOps::bitwise_and(mask, registers[idx]));
                });
            } else {
                tlx::call_for_range<0, k - 1>([&] (size_t idx) {
                    SimdOps::store_callback([&] (auto i, auto x) {cbw(idx * SimdOps::kPacking + i, x);},  SimdOps::bitwise_and(mask, registers[idx]));
                });
                SimdOps::partial_store_callback([&] (auto i, auto x) {cbw((k-1) * SimdOps::kPacking + i, x);}, partial_size,  SimdOps::bitwise_and(mask, registers[k-1]));
            }
        }
    }

private:
    template <size_t kSize, bool kAligned, bool kStream>
    static void load(const simd_type* it, simd_type* x) {
        tlx::call_for_range<0, kSize>([&] (size_t idx) {
            x[idx] = SimdOps::template load<kAligned, kStream>(it + idx);
        });
    }

    template <size_t kSize, bool kAligned, bool kStream>
    static void store(simd_type* it, simd_type* x) {
        tlx::call_for_range<0, kSize>([&] (size_t idx) {
            SimdOps::template store<kAligned, kStream>(it + idx, x[idx]);
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
