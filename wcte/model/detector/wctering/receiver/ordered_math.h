/* Coefficients and arithmetic adapted from glibc 2.37.
 * Copyright (C) 2001-2023 Free Software Foundation, Inc.
 * atan2: IBM Accurate Mathematical Library; exp: Arm Limited.
 * LGPL-2.1-or-later; see COPYING.LIBM and LIBM_ORIGIN.txt.
 */
/* Include after the caller's Vec/fill/choose/any_lane adapters.
 * These routines preserve the verified scalar library's FMA operation order.
 * Each lane is independent; panel, PMT, and likelihood reductions are unchanged.
 * The loader enables them only for the independently tested libm binary.
 */
#include "math_tables.h"
/* Vectorize the normal-range, nonnegative-y branch of glibc 2.37's IBM atan2.
 * Explicit FMA operations follow the installed scalar FMA implementation.
 * Small ratios and exceptional lanes retain the system scalar function.
 * This adaptation passed independent raw bitwise checks on the pinned libm.
 * Unknown libm builds use scalar calls; see build.py and LIBM_ORIGIN.txt.
 */
static inline Vec table_fma(Vec a, Vec b, Vec c) {
#if MATH_LANES == 8
    return _mm512_fmadd_pd(a, b, c);
#else
    return _mm256_fmadd_pd(a, b, c);
#endif
}
static inline Vec atan_coefficient(const int *indices, int column) {
    Vec result;
    for (int lane = 0; lane < MATH_LANES; lane++)
        result[lane] = atan_table[indices[lane]][column].value;
    return result;
}
#if MATH_LANES == 8
#include "vector_transpose.h"
#endif
static inline void atan_coefficients(const int *indices, Vec *c) {
#if MATH_LANES == 8
    /* Each row has seven coefficients; its final masked lane is zero. */
    Vec rows[8];
    for (int i = 0; i < 8; i++)
        rows[i] = _mm512_maskz_loadu_pd(0x7f, &atan_table[indices[i]][0]);
    transpose8(rows, c);
#else
    for (int column = 0; column < 7; column++)
        c[column] = atan_coefficient(indices, column);
#endif
}

/* Small positive-y ratios use the original glibc polynomial and compensated
 * quadrant reconstruction. Explicit FMA follows the installed scalar routine.
 * The exceptionally tiny/large arguments still use that system routine. */
static inline Vec small_atan(Vec u, Vec du, Vec x, Vec ax, Vec ay) {
    const AtanCoefficient d3 = {.bits = UINT64_C(0xbfd5555555555555)};
    const AtanCoefficient d5 = {.bits = UINT64_C(0x3fc99999999997fd)};
    const AtanCoefficient d7 = {.bits = UINT64_C(0xbfc24924923f7603)};
    const AtanCoefficient d9 = {.bits = UINT64_C(0x3fbc71c6e5129a3b)};
    const AtanCoefficient d11 = {.bits = UINT64_C(0xbfb7458022b13c25)};
    const AtanCoefficient d13 = {.bits = UINT64_C(0x3fb375f08b31cbce)};
    Vec v = u * u;
    Vec polynomial = table_fma(v, fill(d13.value), fill(d11.value));
    polynomial = table_fma(v, polynomial, fill(d9.value));
    polynomial = table_fma(v, polynomial, fill(d7.value));
    polynomial = table_fma(v, polynomial, fill(d5.value));
    polynomial = table_fma(v, polynomial, fill(d3.value));
    Vec direct = u + table_fma(u * v, polynomial, du);
    Vec add = (Vec)((x < fill(0.)) & (ax < ay));
    Vec half = (Vec)((x > fill(0.)) | ((x < fill(0.)) & (ax < ay)));
    Vec pi = choose(half, fill(0x1.921fb54442d18p+0), fill(0x1.921fb54442d18p+1));
    Vec remainder =
        choose(half, fill(0x1.1a62633145c07p-54), fill(0x1.1a62633145c07p-53));
    Vec signed_u = choose(add, u, -u), signed_du = choose(add, du, -du);
    Vec base = pi + signed_u;
    Vec correction = (pi - base) + signed_u;
    Vec polynomial_tail = (u * v) * polynomial;
    Vec tail = ((remainder + correction) + signed_du) +
               choose(add, polynomial_tail, -polynomial_tail);
    return choose((Vec)((x > fill(0.)) & (ay < ax)), direct, base + tail);
}

static inline Vec ordered_atan2(Vec y, Vec x) {
    Vec zero = fill(0.), ax = maximum(x, -x), ay = maximum(y, -y);
    Vec denominator = maximum(ax, ay), numerator = minimum(ax, ay);
    Vec u = numerator / denominator;
    Vec usable = (Vec)((y > zero) & (denominator > fill(0x1p-500)) &
                       (denominator < fill(0x1p500)) & (u >= fill(0.0625)));
    Vec small = (Vec)((y > zero) & (numerator > fill(0x1p-500)) &
                      (denominator < fill(0x1p500)) & (u >= fill(0x1p-32)) &
                      (u < fill(0.0625)));
    Vec safe_u = choose(usable, u, fill(0.0625));
    Vec product = denominator * u;
    Vec product_error = table_fma(denominator, u, -product);
    Vec du = ((numerator - product) - product_error) / denominator;
    Vec result = zero;
    if (any_lane(usable)) {
        Vec rounded = (fill(0x1p52) + fill(256.) * safe_u) - fill(0x1p52);
        int indices[MATH_LANES];
        for (int lane = 0; lane < MATH_LANES; lane++)
            indices[lane] = (int)rounded[lane] - 16;
        Vec coefficients[8];
        atan_coefficients(indices, coefficients);
        Vec c0 = coefficients[0], c1 = coefficients[1], c2 = coefficients[2];
        Vec c3 = coefficients[3], c4 = coefficients[4], c5 = coefficients[5],
            c6 = coefficients[6];
        Vec t = u - c0, v = t + du;
        Vec dv = choose((Vec)(maximum(t, -t) > maximum(du, -du)), (t - v) + du,
                        (du - v) + t);
        Vec polynomial = table_fma(v, c6, c5);
        polynomial = table_fma(v, polynomial, c4);
        polynomial = table_fma(v, polynomial, c3);

        /* The first quadrant below pi/4 uses the compensated-addition branch. */
        Vec direct = table_fma(dv, c2, (v * v) * polynomial);
        direct = c1 + table_fma(v, c2, direct);
        Vec direct_mask = (Vec)((x > zero) & (ay < ax));

        /* The other quadrants share the same polynomial but retain their original
         * pi/2 or pi subtraction/addition and final rounding order. */
        polynomial = table_fma(v, polynomial, c2);
        Vec third = (Vec)((x < zero) & (ax < ay));
        Vec half = (Vec)((x > zero) | ((x < zero) & (ax < ay)));
        Vec pi = choose(half, fill(0x1.921fb54442d18p+0), fill(0x1.921fb54442d18p+1));
        Vec remainder =
            choose(half, fill(0x1.1a62633145c07p-54), fill(0x1.1a62633145c07p-53));
        Vec base = choose(third, pi + c1, pi - c1);
        Vec correction = table_fma(choose(third, v, -v), polynomial, remainder);
        result = choose(direct_mask, direct, base + correction);
    }
    if (any_lane(small))
        result = choose(small, small_atan(u, du, x, ax, ay), result);
    int mask = any_lane(usable) | any_lane(small);
    for (int lane = 0; lane < MATH_LANES; lane++)
        if (!(mask & (1 << lane)))
            result[lane] = atan2(y[lane], x[lane]);
    return result;
}

/* Pack glibc 2.37's ordinary-range FMA exp operations without reordering.
 * Tiny, exceptional and |x|>=512 lanes keep the system scalar routine. */
static inline Vec ordered_exp(Vec x) {
    Vec magnitude = maximum(x, -x);
    Vec ordinary = (Vec)((magnitude >= fill(0x1p-54)) & (magnitude < fill(512.)));
    Vec safe = choose(ordinary, x, fill(0.));
    Vec rounded = table_fma(safe, fill(0x1.71547652b82fep7), fill(0x1.8p52));
    Vec k = rounded - fill(0x1.8p52);
    Vec remainder = table_fma(k, fill(-0x1.62e42fefa0000p-8), safe);
    remainder = table_fma(k, fill(-0x1.cf79abc9e3b3ap-47), remainder);
    Vec tail, scale;
    for (int lane = 0; lane < MATH_LANES; lane++) {
        AtanCoefficient integer = {.value = rounded[lane]}, t, s;
        uint64_t index = 2 * (integer.bits % 128);
        t.bits = exp_table[index];
        s.bits = exp_table[index + 1] + (integer.bits << 45);
        tail[lane] = t.value;
        scale[lane] = s.value;
    }
    Vec r2 = remainder * remainder;
    Vec low =
        table_fma(remainder, fill(0x1.555555555543cp-3), fill(0x1.ffffffffffdbdp-2));
    Vec high =
        table_fma(remainder, fill(0x1.1111167a4d017p-7), fill(0x1.55555cf172b91p-5));
    Vec correction = table_fma(r2, low, tail + remainder);
    correction = table_fma(r2 * r2, high, correction);
    Vec result = table_fma(scale, correction, scale);
    int mask = any_lane(ordinary);
    for (int lane = 0; lane < MATH_LANES; lane++)
        if (!(mask & (1 << lane)))
            result[lane] = exp(x[lane]);
    return result;
}
