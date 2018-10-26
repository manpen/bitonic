#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

inline void die_unless(bool predicate) {
    if (!predicate) abort();
}

template <typename T, typename ... Ts>
inline void die_unless(bool predicate, const T& x, const Ts&... xs) {
    if (!predicate) {
        std::cerr << x;
        die_unless(false, xs...);
    }
}


template <typename T1, typename T2, typename ... Ts>
inline void die_unless_equal(T1&& a, T2&& b, const Ts&... xs) {
    if (a != b) {
        std::cerr << "Mismatch: " << a << " != " << b << ". ";
        die_unless(false, xs...);
    }
}

template <class T>
static T* align_pointer(T* ptr, std::size_t alignment) {
    uintptr_t v = reinterpret_cast<std::uintptr_t>(ptr);
    v = (v - 1 + alignment) & ~(alignment - 1);
    return reinterpret_cast<T*>(v);
}

template <class T>
static bool is_aligned(T* ptr, std::size_t alignment) {
    uintptr_t v = reinterpret_cast<std::uintptr_t>(ptr);
    return 0 == (v % alignment);
}

