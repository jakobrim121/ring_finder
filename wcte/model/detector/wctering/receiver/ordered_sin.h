/* SIMD adaptation of the ordinary-range glibc 2.37 sine branches.
 * IBM Accurate Mathematical Library.
 * Copyright (C) 2001-2023 Free Software Foundation, Inc.
 * LGPL-2.1-or-later; see COPYING.LIBM in the fitter package.
 * Raw bitwise checks passed on the pinned libm; unknown builds use scalar calls.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
typedef union {
    uint64_t bits;
    double value;
} sin_Coefficient;
static inline Vec sin_fill(double x) {
    return _mm512_set1_pd(x);
}
static inline int sin_mask(Vec x) {
    return _mm512_movepi64_mask(_mm512_castpd_si512(x));
}
static inline Vec sin_choose(Vec test, Vec yes, Vec no) {
    return _mm512_mask_blend_pd(sin_mask(test), no, yes);
}
static inline Vec sin_absolute(Vec x) {
    return _mm512_max_pd(x, -x);
}
#include "sin_coefficients.h"
static inline void sin_lookup(Vec rounded, Vec *sn, Vec *ssn, Vec *cs, Vec *ccs) {
    for (int lane = 0; lane < 8; lane++) {
        sin_Coefficient index = {.value = rounded[lane]};
        size_t offset = (uint32_t)index.bits * 4;
        (*sn)[lane] = sincos_table[offset].value;
        (*ssn)[lane] = sincos_table[offset + 1].value;
        (*cs)[lane] = sincos_table[offset + 2].value;
        (*ccs)[lane] = sincos_table[offset + 3].value;
    }
}

#if defined(__clang__)
#define sin_MATH_FUNCTION __attribute__((noinline))
#else
#define sin_MATH_FUNCTION __attribute__((noinline, optimize("fp-contract=fast")))
#endif
static sin_MATH_FUNCTION Vec ordered_sin(Vec input) {
#if defined(__clang__)
#pragma clang fp contract(fast)
#endif
    const double sn3 = -1.66666666666664880952546298448555E-01;
    const double sn5 = 8.33333214285722277379541354343671E-03;
    const double cs2 = 4.99999999999999999999950396842453E-01;
    const double cs4 = -4.16666666666664434524222570944589E-02;
    const double cs6 = 1.38888874007937613028114285595617E-03;
    Vec a = sin_absolute(input), result = sin_fill(0.);
    Vec low = (Vec)((a >= sin_fill(.126)) & (a < sin_fill(0x1.b6p-1)));
    Vec middle = (Vec)((a >= sin_fill(0x1.b6p-1)) & (a < sin_fill(0x1.368fdp1)));
    if (sin_mask(low)) {
        Vec x = sin_choose(low, a, sin_fill(.5));
        Vec u = sin_fill(0x1.8p45) + x;
        x = x - (u - sin_fill(0x1.8p45));
        Vec xx = x * x;
        Vec s = x + x * xx * (sin_fill(sn3) + xx * sin_fill(sn5));
        Vec c = xx * (sin_fill(cs2) + xx * (sin_fill(cs4) + xx * sin_fill(cs6)));
        Vec sn, ssn, cs, ccs;
        sin_lookup(u, &sn, &ssn, &cs, &ccs);
        Vec cor = (ssn + s * ccs - sn * c) + cs * s;
        result = sin_choose(low, sn + cor, result);
    }
    if (sin_mask(middle)) {
        Vec x = sin_fill(0x1.921fb54442d18p0) - sin_choose(middle, a, sin_fill(1.));
        Vec dx = sin_choose((Vec)(x < sin_fill(0.)), sin_fill(-0x1.1a62633145c07p-54),
                            sin_fill(0x1.1a62633145c07p-54));
        Vec u = sin_fill(0x1.8p45) + sin_absolute(x);
        x = sin_absolute(x) - (u - sin_fill(0x1.8p45)) + dx;
        Vec xx = x * x;
        Vec s = x + x * xx * (sin_fill(sn3) + xx * sin_fill(sn5));
        Vec c = xx * (sin_fill(cs2) + xx * (sin_fill(cs4) + xx * sin_fill(cs6)));
        Vec sn, ssn, cs, ccs;
        sin_lookup(u, &sn, &ssn, &cs, &ccs);
        Vec cor = (ccs - s * ssn - cs * c) - sn * s;
        result = sin_choose(middle, cs + cor, result);
    }
    result = sin_choose((Vec)(input < sin_fill(0.)), -result, result);
    int ordinary = sin_mask(low) | sin_mask(middle);
    for (int lane = 0; lane < 8; lane++)
        if (!(ordinary & (1 << lane)))
            result[lane] = sin(input[lane]);
    return result;
}
