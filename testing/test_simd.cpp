#include <bitonic/simd.hpp>
#include "helper.hpp"

#include <algorithm>
#include <random>

constexpr size_t kMaxSize = 256 + 1;
std::mt19937_64 prng;

template <typename SimdOps>
void test_sorted_1toN(size_t N) {
    std::cout << "test_sorted_1toN      <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    value_type tmp[kMaxSize];
    for(size_t i=0; i != N; i++)
        tmp[i] = i+1;

    Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

    for(size_t i=0; i != N; i++)
        die_unless_equal(tmp[i], i+1);
}

template <typename SimdOps>
void test_sorted_1toNHalf(size_t N) {
    std::cout << "test_sorted_1toNHalf  <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    value_type tmp[kMaxSize];
    for(size_t i=0; i != N; i++)
        tmp[i] = i/2+1;

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

template <typename SimdOps>
void test_zero_one(size_t N) {
    std::cout << "test_zero_one         <" << N << ", " << SimdOps::name() << ">";
    if (N >= 32) {
        std::cout << " ... skipped due to size\n";
        return;
    } else {
        std::cout << "\n";
    }

    using value_type = typename SimdOps::value_type;

    const size_t max = N < sizeof(size_t)*8 ? 1ull << N : 0;
    value_type tmp[kMaxSize];

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

template <typename SimdOps>
void test_random_elements(size_t N) {
    std::cout << "test_random_elements  <" << N << ", " << SimdOps::name() << ">\n";
    using value_type = typename SimdOps::value_type;

    std::uniform_int_distribution<value_type> distr;

    value_type tmp[kMaxSize], ref[kMaxSize];

    for(size_t iter = 0; iter != 100000; iter++) {
        for (size_t i = 0; i <= N; i++)
            tmp[i] = ref[i] = distr(prng);

        const auto last_value = ref[N]; // check that this value is not override and SN stays within bounds

        Bitonic::SimdSort<SimdOps>::template sort<false>(tmp, tmp + N);

        std::sort(ref, ref + N);

        for (size_t i = 0; i != N; i++)
            die_unless_equal(tmp[i], ref[i]);

        die_unless_equal(tmp[N], last_value, "Last value");
    }
}

template <typename T>
void test_all_by_type() {
    using SimdOps = typename Bitonic::SimdAdapter::Select<T>::type;

    for(size_t N : {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                    24, 32, 40, 48, 56, 64, 120, 128, 192, 256}) {
        if (N < SimdOps::kPacking) continue;
        if (N / SimdOps::kPacking > 32) continue;

        test_sorted_1toN<SimdOps>(N);
        test_sorted_1toNHalf<SimdOps>(N);

        test_zero_one<SimdOps>(N);

        test_random_elements<SimdOps>(N);

        std::cout << "\n";
    }
    std::cout << "\n\n";
}

int main() {
    test_all_by_type<int32_t>();
    test_all_by_type<uint32_t>();

    return 0;
}
