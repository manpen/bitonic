#include <bitonic/simd.hpp>
#include "helper.hpp"

#include <algorithm>
#include <random>

std::mt19937_64 prng;

template <size_t N, typename SimdOps>
void test_sorted_1toN() {
    std::cout << "test_sorted_1toN      <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    value_type tmp[N];
    for(size_t i=0; i != N; i++)
        tmp[i] = i+1;

    Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

    for(size_t i=0; i != N; i++)
        die_unless_equal(tmp[i], i+1);
}

template <size_t N, typename SimdOps>
void test_sorted_1toNHalf() {
    std::cout << "test_sorted_1toNHalf  <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    value_type tmp[N];
    for(size_t i=0; i != N; i++)
        tmp[i] = i/2+1;

    Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

    for(size_t i=0; i != N; i++)
        die_unless_equal(tmp[i], i/2+1);
}

template <size_t N, typename SimdOps>
void test_inverted_1toN() {
    std::cout << "test_inverted_1toN    <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    value_type tmp[N];
    for(size_t i=0; i != N; i++)
        tmp[i] = N - i;

    Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

    for(size_t i=0; i != N; i++)
        die_unless_equal(tmp[i], i+1);
}

template <size_t N, typename SimdOps>
void test_inverted_1toNHalf() {
    std::cout << "test_inverted_1toNHalf<" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    value_type tmp[N];
    for(size_t i=0; i != N; i++)
        tmp[i] = N/2 - i/2;

    Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

    for(size_t i=0; i != N; i++)
        die_unless_equal(tmp[i], i/2+1);
}

template <typename T>
int pop_count(T num) {
    int res = 0;
    for(; num; num >>= 1) {
        res += num & 1;
    }
    return res;
}

template <size_t N, typename SimdOps>
void test_zero_one() {
    std::cout << "test_zero_one         <" << N << ", " << SimdOps::name() << ">";
    if constexpr (N >= 32) {
        std::cout << " ... skipped due to size\n";
        return;
    } else {
        std::cout << "\n";
    }

    using value_type = typename SimdOps::value_type;

    const size_t max = N < sizeof(size_t)*8 ? 1ull << N : 0;
    value_type tmp[N];

    for(size_t num = 0; num != max; num++) {
        for (size_t i = 0; i != N; i++)
            tmp[i] = (num >> i) & 1;

        Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

        const size_t num_ones = pop_count(num);
        const size_t num_zeros = N - num_ones;

        for (size_t i = 0; i != num_zeros; i++) {
            die_unless_equal(tmp[i], 0);
        }

        for (size_t i = num_zeros; i != N; i++) {
            die_unless_equal(tmp[i], 1);
        }
    }
}

template <size_t N, typename SimdOps>
void test_random_elements() {
    std::cout << "test_random_elements  <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    std::uniform_int_distribution<value_type> distr;

    value_type tmp[N], ref[N];

    for(size_t iter = 0; iter != 100000; iter++) {
        for (size_t i = 0; i != N; i++)
            tmp[i] = ref[i] = distr(prng);

        Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

        std::sort(ref, ref + N);

        for (size_t i = 0; i != N; i++)
            die_unless_equal(tmp[i], ref[i]);
    }
}

template <size_t N, typename SimdOps>
void test_all_by_type_and_size() {
    if (N < SimdOps::kPacking) return;

    test_sorted_1toN<N, SimdOps>();
    test_sorted_1toNHalf<N, SimdOps>();

    test_inverted_1toN<N, SimdOps>();
    test_inverted_1toNHalf<N, SimdOps>();

    test_zero_one<N, SimdOps>();

    test_random_elements<N, SimdOps>();

    std::cout << "\n";
}

template <typename SimdOps>
void test_all_by_type() {
    test_all_by_type_and_size<  2, SimdOps>();
    test_all_by_type_and_size<  4, SimdOps>();
    test_all_by_type_and_size<  8, SimdOps>();
    test_all_by_type_and_size< 16, SimdOps>();
    test_all_by_type_and_size< 24, SimdOps>();
    test_all_by_type_and_size< 32, SimdOps>();
    test_all_by_type_and_size< 40, SimdOps>();
    test_all_by_type_and_size< 48, SimdOps>();
    test_all_by_type_and_size< 56, SimdOps>();
    test_all_by_type_and_size< 64, SimdOps>();
    test_all_by_type_and_size<120, SimdOps>();
    test_all_by_type_and_size<128, SimdOps>();

    std::cout << "\n\n";
}

int main() {
    test_all_by_type<Bitonic::SimdAdapter::SignedInt32>();
    test_all_by_type<Bitonic::SimdAdapter::UnsignedInt32>();

    return 0;
}
