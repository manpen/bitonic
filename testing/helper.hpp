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