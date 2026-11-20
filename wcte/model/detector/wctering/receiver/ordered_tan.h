/* SIMD adaptation of ordinary-range glibc 2.37 tangent branches.
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
} tan_Coefficient;
static inline Vec tan_fill(double x) {
    return _mm512_set1_pd(x);
}
static inline int tan_mask(Vec x) {
    return _mm512_movepi64_mask(_mm512_castpd_si512(x));
}
static inline Vec tan_choose(Vec test, Vec yes, Vec no) {
    return _mm512_mask_blend_pd(tan_mask(test), no, yes);
}
static inline Vec tan_absolute(Vec x) {
    return _mm512_max_pd(x, -x);
}
#include "tan_coefficients.h"
#if defined(__clang__)
#define tan_MATH_FUNCTION __attribute__((noinline))
#else
#define tan_MATH_FUNCTION __attribute__((noinline, optimize("fp-contract=fast")))
#endif
static tan_MATH_FUNCTION Vec ordered_tan(Vec input) {
#if defined(__clang__)
#pragma clang fp contract(fast)
#endif
    Vec w = tan_absolute(input), zero = tan_fill(0.);
    Vec direct = (Vec)((w > tan_fill(g2.value)) & (w <= tan_fill(g3.value)));
    Vec reduced = (Vec)((w > tan_fill(g3.value)) & (w <= tan_fill(g4.value)));
    /* glibc's first range reduction, preserving every operation and FMA. */
    Vec x = tan_choose(reduced, input, zero);
    Vec rounded = x * tan_fill(hpinv.value) + tan_fill(toint.value);
    Vec xn = rounded - tan_fill(toint.value);
    Vec t1 = (x - xn * tan_fill(mp1.value)) - xn * tan_fill(mp2.value);
    Vec da = xn * tan_fill(mp3.value);
    Vec a = t1 - da;
    da = (t1 - a) - da;
    Vec negative = (Vec)(a < zero), ya = tan_absolute(a);
    Vec yya = tan_choose(negative, -da, da);
    reduced = _mm512_and_pd(reduced, (Vec)(ya > tan_fill(g2.value)));
    Vec ordinary = _mm512_or_pd(direct, reduced);
    Vec magnitude = tan_choose(direct, w, ya);
    magnitude = tan_choose(ordinary, magnitude, tan_fill(.5));
    Vec correction = tan_choose(reduced, yya, zero);
    Vec rounded_index = tan_fill(mfftnhf.value) + tan_fill(256.) * magnitude;
    Vec origin, fi, gi, cotangent;
    for (int lane = 0; lane < 8; lane++) {
        int index = (int)rounded_index[lane];
        origin[lane] = tangent_table[index][0].value;
        fi[lane] = tangent_table[index][1].value;
        gi[lane] = tangent_table[index][2].value;
        tan_Coefficient quadrant = {.value = rounded[lane]};
        cotangent[lane] = (quadrant.bits & 1) ? 1. : 0.;
    }
    Vec z = magnitude - origin;
    z = tan_choose(reduced, z + correction, z);
    Vec z2 = z * z;
    Vec pz = z + z * z2 * (tan_fill(e0.value) + z2 * tan_fill(e1.value));
    Vec cot = _mm512_and_pd(reduced, (Vec)(cotangent > zero));
    Vec denominator = tan_choose(cot, fi + pz, gi - pz);
    Vec t2 = pz * (gi + fi) / denominator;
    Vec result = tan_choose(cot, gi - t2, fi + t2);
    Vec sign =
        tan_choose(direct, tan_choose((Vec)(input < zero), tan_fill(-1.), tan_fill(1.)),
                   tan_choose(negative, tan_fill(-1.), tan_fill(1.)));
    result *= tan_choose(cot, -sign, sign);
    int usable = tan_mask(ordinary);
    for (int lane = 0; lane < 8; lane++)
        if (!(usable & (1 << lane)))
            result[lane] = tan(input[lane]);
    return result;
}
