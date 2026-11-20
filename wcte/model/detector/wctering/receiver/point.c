/* Experimental four-ray CPU kernel for the unchanged V101 point source.
 *
 * Each SIMD lane is an independent PMT. Scalar libm and the installed SciPy
 * BLAS retain the reference operation order and rounding. Only independent
 * per-PMT arithmetic is packed into vector instructions; no reductions move.
 * Compile without fast-math and with floating-point contraction disabled.
 *
 * Input geometry is mm; parameters are [centre_m(3), axis(3), 0, opening_rad,
 * log(width_rad)]. Output is unnormalized light and its N-by-9 Jacobian.
 */
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifndef POINT_LANES
#define POINT_LANES 4
#endif
typedef double Vec __attribute__((vector_size(8 * POINT_LANES)));
/* Only independent PMTs occupy different lanes; scalar sums never move. */
static inline Vec scalar_acos(Vec a) {
    Vec result;
    for (int lane = 0; lane < POINT_LANES; lane++)
        result[lane] = acos(a[lane]);
    return result;
}
static inline Vec fill(double value) {
    Vec result;
    for (int lane = 0; lane < POINT_LANES; lane++)
        result[lane] = value;
    return result;
}
#if POINT_LANES == 8
static inline Vec maximum(Vec a, Vec b) {
    return _mm512_max_pd(a, b);
}
static inline Vec minimum(Vec a, Vec b) {
    return _mm512_min_pd(a, b);
}
static inline Vec square_root(Vec a) {
    return _mm512_sqrt_pd(a);
}
static inline int any_lane(Vec a) {
    return _mm512_movepi64_mask(_mm512_castpd_si512(a));
}
static inline Vec choose(Vec mask, Vec yes, Vec no) {
    return _mm512_mask_blend_pd(any_lane(mask), no, yes);
}
#else
static inline Vec maximum(Vec a, Vec b) {
    return _mm256_max_pd(a, b);
}
static inline Vec minimum(Vec a, Vec b) {
    return _mm256_min_pd(a, b);
}
static inline Vec square_root(Vec a) {
    return _mm256_sqrt_pd(a);
}
static inline int any_lane(Vec a) {
    return _mm256_movemask_pd(a);
}
static inline Vec choose(Vec mask, Vec yes, Vec no) {
    return _mm256_blendv_pd(no, yes, mask);
}
#endif
#ifdef ORDERED_MATH
#define MATH_LANES POINT_LANES
#include "ordered_math.h"
#define scalar_exp ordered_exp
#define scalar_atan2 ordered_atan2
#else
static inline Vec scalar_exp(Vec a) {
    Vec result;
    for (int lane = 0; lane < POINT_LANES; lane++)
        result[lane] = exp(a[lane]);
    return result;
}
static inline Vec scalar_atan2(Vec a, Vec b) {
    Vec result;
    for (int lane = 0; lane < POINT_LANES; lane++)
        result[lane] = atan2(a[lane], b[lane]);
    return result;
}

#endif

static inline Vec gather(const double *a, size_t i, int c, size_t n) {
    /* Repeat the last PMT in unused tail lanes; those lanes are never stored. */
    Vec result;
    for (size_t lane = 0; lane < POINT_LANES; lane++)
        result[lane] = a[3 * (i + lane < n ? i + lane : n - 1) + c];
    return result;
}

/* Numba's length-three np.dot uses this exact SciPy BLAS entry point. */
typedef double (*Dot)(int *, double *, int *, double *, int *);
static Dot dot_function = NULL;
static int packed_dot = 0;
/* AOCL 5.1 uses (term0+term2)+term1 for three contiguous doubles.
 * The loader enables this only for that backend, after checking its rounding.
 * All other libraries retain the exact BLAS function call. */
int enable_packed_dot(void) {
    uint64_t state = 0x517cc1b727220a95ULL;
    int n = 3, stride = 1;
    for (int sample = 0; sample < 2048; sample++) {
        double a[3], b[3];
        for (int j = 0; j < 3; j++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            a[j] = ((double)(state >> 11) / 4503599627370496. - 1.) * 10000.;
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            b[j] = ((double)(state >> 11) / 4503599627370496. - 1.);
        }
        double expected = dot_function(&n, a, &stride, b, &stride);
        if (expected != (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1])
            return 0;
    }
    packed_dot = 1;
    return 1;
}
void set_dot_function(void *address) {
    dot_function = (Dot)address;
}
static inline double dot3(double *a, double *b) {
    if (packed_dot)
        return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1];
    int n = 3, stride = 1;
    return dot_function(&n, a, &stride, b, &stride);
}
static inline Vec vector_dot(Vec a0, Vec a1, Vec a2, Vec b0, Vec b1, Vec b2) {
    if (packed_dot)
        return (a0 * b0 + a2 * b2) + a1 * b1;
    Vec result;
    for (int lane = 0; lane < POINT_LANES; lane++) {
        double a[3] = {a0[lane], a1[lane], a2[lane]};
        double b[3] = {b0[lane], b1[lane], b2[lane]};
        result[lane] = dot3(a, b);
    }
    return result;
}

void point_rays(size_t n, const double *positions, const double *normals,
                const double *x, double *values, double *jacobian) {
    const double z0 = 34.597 / 53., t2 = 1. - z0 * z0, threshold = sqrt(t2);
    const double norm = sqrt(dot3((double *)x + 3, (double *)x + 3));
    const double variance = exp(2. * x[8]);
    Vec axis[3] = {fill(x[3] / norm), fill(x[4] / norm), fill(x[5] / norm)};
    const Vec zero = fill(0.), one = fill(1.);
    for (size_t i = 0; i < n; i += POINT_LANES) {
        Vec d[3], normal[3], direction[3];
        for (int c = 0; c < 3; c++) {
            d[c] = gather(positions, i, c, n) - fill(1000. * x[c]);
            normal[c] = gather(normals, i, c, n);
        }
        Vec distance = maximum(
            square_root(vector_dot(d[0], d[1], d[2], d[0], d[1], d[2])), fill(1e-9));
        for (int c = 0; c < 3; c++)
            direction[c] = d[c] / distance;
        Vec incidence = vector_dot(normal[0], normal[1], normal[2], direction[0],
                                   direction[1], direction[2]);
        Vec mu = -incidence;
        Vec acceptance = maximum(mu, zero),
            derivative = choose((Vec)(mu > zero), one, zero);
        Vec partial = (Vec)((mu < fill(threshold)) & (mu > fill(-threshold)));
        if (any_lane(partial)) {
            Vec h = square_root(maximum(fill(t2) - mu * mu, zero));
            Vec phi = scalar_atan2(h, -mu * fill(z0));
            Vec projected =
                (scalar_atan2(h, fill(z0)) - fill(z0) * h + mu * fill(t2) * phi) /
                fill(M_PI * t2);
            Vec slope = (fill(t2) * phi + mu * fill(z0) * h / (one - mu * mu)) /
                        fill(M_PI * t2);
            acceptance = choose(partial, maximum(projected, zero), acceptance);
            derivative =
                choose(partial, minimum(maximum(slope, zero), one), derivative);
        }
        Vec denominator = distance * distance + fill(10000.);
        Vec raw = acceptance / denominator;
        Vec exposure = maximum(raw, fill(1e-30));
        Vec cosine = vector_dot(direction[0], direction[1], direction[2], axis[0],
                                axis[1], axis[2]);
        Vec bounded = minimum(maximum(cosine, fill(-1. + 1e-10)), fill(1. - 1e-10));
        Vec residual = scalar_acos(bounded) - fill(x[7]);
        Vec angular = scalar_exp(fill(-.5) * residual * residual / fill(variance));
        Vec value = exposure * angular;
        Vec slope = residual / (fill(variance) * square_root(one - bounded * bounded));
        slope =
            choose((Vec)((cosine > fill(-1. + 1e-10)) & (cosine < fill(1. - 1e-10))),
                   slope, zero);
        Vec spatial[3], rotation[3], derivatives[9];
        for (int c = 0; c < 3; c++) {
            Vec de = derivative * (normal[c] - incidence * direction[c]) /
                         (distance * denominator) +
                     exposure * fill(2.) * d[c] / denominator;
            de = choose((Vec)(raw > fill(1e-30)), de, zero);
            spatial[c] = angular * de -
                         value * slope * (axis[c] - bounded * direction[c]) / distance;
            rotation[c] = value * slope * direction[c];
            derivatives[c] = spatial[c] * fill(1000.);
        }
        Vec axial_rotation = vector_dot(rotation[0], rotation[1], rotation[2], axis[0],
                                        axis[1], axis[2]);
        for (int c = 0; c < 3; c++)
            derivatives[c + 3] = (rotation[c] - axial_rotation * axis[c]) / fill(norm);
        derivatives[6] = zero;
        derivatives[7] = value * residual / fill(variance);
        derivatives[8] = value * residual * residual / fill(variance);
        Vec v = value;
        Vec j[9];
        for (int c = 0; c < 9; c++)
            j[c] = derivatives[c];
        for (size_t lane = 0; lane < POINT_LANES && i + lane < n; lane++) {
            values[i + lane] = v[lane];
            for (int c = 0; c < 9; c++)
                jacobian[9 * (i + lane) + c] = j[c][lane];
        }
    }
}
