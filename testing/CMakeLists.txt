add_executable(test_simd test_simd.cpp)
target_compile_options(test_simd PRIVATE "${BITONIC_CXX_FLAGS}")
add_test(test_simd test_simd)

add_executable(test_adapter test_adapter.cpp)
target_compile_options(test_adapter PRIVATE "${BITONIC_CXX_FLAGS}")
add_test(test_adapter test_adapter)