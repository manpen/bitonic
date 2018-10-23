cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)

option(BITONIC_ENABLE_TESTING "Enable building of tests" OFF)

include_directories(include)
set(BITONIC_CXX_FLAGS "-mavx2")

if (BITONIC_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(testing)
endif()