#include <bitonic/simd.hpp>
#include "helper.hpp"

#include <memory>
#include <iostream>
#include <random>

std::mt19937_64 prng{0};

template <typename T>
static void random_fill(T* begin, size_t n) {
    std::uniform_int_distribution<T> distr;
    for(size_t i=0; i!=n; i++)
        begin[i] = distr(prng);
}

template<typename SimdOps>
static void test_all() {
    std::cout << "Test: " << SimdOps::name() << "\n";

    using simd_type = typename SimdOps::simd_type;
    using value_type = typename SimdOps::value_type;

    constexpr size_t num_simd_regs = 32;

    constexpr auto kPacking = SimdOps::kPacking;
    constexpr auto kAlign = kPacking * sizeof(value_type);

    auto buffer = std::make_unique<char*[]>(sizeof(simd_type) * (num_simd_regs+1));

    auto data = reinterpret_cast<value_type*>(align_pointer(buffer.get(), sizeof(simd_type)));
    auto regs = reinterpret_cast<simd_type*>(data);

// load and store
    std::cout << " Load&Store\n";
    {
        using simd_type = typename SimdOps::simd_type;

        random_fill(data, num_simd_regs * kPacking);

        std::uniform_int_distribution<size_t> distr{0, (num_simd_regs - 1) * kPacking};
        std::uniform_int_distribution<size_t> size_distr{1, kPacking};

        std::uniform_real_distribution<float_t> prob;

        for(int i=0; i != 100000; i++) {
            auto size = size_distr(prng);
            if (kPacking > 8 && size % 2) size++;

            // find random read pointer
            auto read_ptr = data + distr(prng);

            // find random non-overlapping write ptr
            auto write_ptr = read_ptr;
            while ((write_ptr <= read_ptr && write_ptr + kPacking > read_ptr) ||
                   (read_ptr <= write_ptr && read_ptr + kPacking > write_ptr)) {
                write_ptr = data + distr(prng);
            }

            simd_type reg;
            if (size == kPacking) {
                if (is_aligned(read_ptr, kAlign)) {
                    if (prob(prng) > 0.5)
                        reg = SimdOps::template load<true, false>(reinterpret_cast<simd_type *>(read_ptr));
                    else
                        reg = SimdOps::template load<true, true>(reinterpret_cast<simd_type *>(read_ptr));

                } else {
                    reg = SimdOps::template load<false, false>(reinterpret_cast<simd_type *>(read_ptr));
                }

                if (is_aligned(write_ptr, kAlign)) {
                    if (prob(prng) > 0.5)
                        SimdOps::template store<true, false>(reinterpret_cast<simd_type *>(write_ptr), reg);
                    else
                        SimdOps::template store<true, true>(reinterpret_cast<simd_type *>(write_ptr), reg);

                } else {
                    SimdOps::template store<false, false>(reinterpret_cast<simd_type *>(write_ptr), reg);
                }
            } else {
                reg = SimdOps::partial_load(reinterpret_cast<simd_type *>(read_ptr), size, 123);

                if (prob(prng) > 0.5) {
                    // partial write
                    for(int j=size; j != kPacking; j++)
                        write_ptr[j] = 0xde * j;

                    SimdOps::partial_store(reinterpret_cast<simd_type *>(write_ptr), size, reg);

                    for(int j=size; j != kPacking; j++)
                        die_unless_equal(write_ptr[j], 0xde * j, j);

                } else {
                    SimdOps::template store<false, false>(reinterpret_cast<simd_type *>(write_ptr), reg);

                    for(int j=size; j != kPacking; j++)
                        die_unless_equal(write_ptr[j], 123);
                }
            }

            for(int j=0; j != size; j++)
                die_unless_equal(read_ptr[j], write_ptr[j], j);
        }
    }

    std::cout << " swap_low_high\n";
    {
        for(int i=0; i != 100; i++) {
            random_fill(data, 2*kPacking);

            {
                auto reg = SimdOps::template load<true, false>(regs);
                reg = SimdOps::swap_low_high(reg);
                SimdOps::template store<true, false>(regs + 1, reg);
            }

            for(int j = 0; j != kPacking / 2; j++) {
                die_unless_equal(data[j], data[3 * kPacking / 2 + j], j);
                die_unless_equal(data[kPacking/2 + j], data[kPacking + j], j);
            }
        }
    }

    std::cout << " Mirror\n";
    {
        for(int i=0; i != 100; i++) {
            random_fill(data, 2*kPacking);

            {
                auto reg = SimdOps::template load<true, false>(regs);
                reg = SimdOps::mirror(reg);
                SimdOps::template store<true, false>(regs + 1, reg);
            }

            for(int j = 0; j != kPacking; j++)
                die_unless_equal(data[j], data[2*kPacking - 1 - j], j);
        }
    }

    std::cout << " min/max\n";
    {
        for(int i=0; i != 100; i++) {
            random_fill(data, 4*kPacking);

            {
                auto a = SimdOps::template load<true, false>(regs);
                auto b = SimdOps::template load<true, false>(regs + 1);

                auto mi = SimdOps::min(a,b);
                auto ma = SimdOps::max(a,b);

                SimdOps::template store<true, false>(regs + 2, mi);
                SimdOps::template store<true, false>(regs + 3, ma);
            }

            for(int j = 0; j != kPacking; j++) {
                auto mi = std::min(data[j], data[j+kPacking]);
                auto ma = std::max(data[j], data[j+kPacking]);

                die_unless_equal(data[2*kPacking + j], mi, j);
                die_unless_equal(data[3*kPacking + j], ma, j);
            }
        }
    }


}


int main() {
    test_all<Bitonic::SimdAdapter::SignedInt16>();
    test_all<Bitonic::SimdAdapter::UnsignedInt16>();

    test_all<Bitonic::SimdAdapter::SignedInt32>();
    test_all<Bitonic::SimdAdapter::UnsignedInt32>();

    return 0;
}
