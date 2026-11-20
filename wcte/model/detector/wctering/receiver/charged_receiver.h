#pragma once
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include "vector_ops.h"

static inline Vec v_acos(Vec x) {
    Vec v;
    for (int lane = 0; lane < LANES; ++lane) v[lane] = acos(x[lane]);
    return v;
}

static inline void cap_at(double mu, const double *coefficients, size_t count,
                          double low, double step, double *value) {
    const double z = 34.597/53., t2 = 1.-z*z, limit = sqrt(t2);
    if (mu >= limit || mu <= -limit) {
        *value = mu >= limit ? mu : 0.;
        return;
    }
    if (mu <= low || mu >= low+step*count) {
        const double h = sqrt(fmax(t2-mu*mu, 0.));
        const double phi = atan2(h, -mu*z);
        *value = fmax((atan2(h, z)-z*h+mu*t2*phi)/(M_PI*t2), 0.);
        return;
    }
    size_t j = (size_t)((mu-low)/step);
    if (j >= count) j = count-1;
    const double t = (mu-(low+j*step))/step;
    const double *a = coefficients + 6*j;
    *value = a[0]+t*(a[1]+t*(a[2]+t*(a[3]+t*(a[4]+t*a[5]))));
}

static inline Vec read_column(const double *data, size_t first, size_t count, int column) {
    Vec result = fill(0.);
    for (size_t lane = 0; lane < count; ++lane)
        result[lane] = data[3*(first+lane)+column];
    return result;
}

static inline void vector_cap(Vec mu, const double *coefficients, size_t count,
                              double low, double step, Vec *value) {
    const double z = 34.597/53., limit = sqrt(1.-z*z);
    const Vec interpolated = (Vec)((mu > fill(low)) & (mu < fill(low+step*count)));
    const Vec front = (Vec)(mu >= fill(limit));
    *value = choose(front, mu, fill(0.));
    if (any_lane(interpolated)) {
        const Vec grid = minimum(maximum((mu-low)/step, fill(0.)), fill(count-1.));
#if LANES == 8
        const __m512i j = _mm512_cvttpd_epi64(grid);
        const __m512i row = _mm512_mullo_epi64(j, _mm512_set1_epi64(6));
        const Vec t = (mu-(low+_mm512_cvtepi64_pd(j)*step))/step;
#define COEFF(k) _mm512_i64gather_pd(row, coefficients+(k), 8)
#else
        const __m128i j = _mm256_cvttpd_epi32(grid);
        const __m128i row = _mm_mullo_epi32(j, _mm_set1_epi32(6));
        const Vec t = (mu-(low+_mm256_cvtepi32_pd(j)*step))/step;
#define COEFF(k) _mm256_i32gather_pd(coefficients+(k), row, 8)
#endif
        const Vec a = COEFF(0), b = COEFF(1), c = COEFF(2);
        const Vec d = COEFF(3), e = COEFF(4), f = COEFF(5);
#undef COEFF
        *value = choose(interpolated, a+t*(b+t*(c+t*(d+t*(e+t*f)))), *value);
    }
    const Vec grazing = (Vec)((mu > fill(-limit)) & (mu < fill(limit)));
    const int boundary = any_lane(choose(interpolated, fill(0.), grazing));
    if (boundary) {
        for (size_t lane = 0; lane < LANES; ++lane) {
            if (!(boundary & (1 << lane))) continue;
            double y;
            cap_at(mu[lane], coefficients, count, low, step, &y);
            (*value)[lane] = y;
        }
    }
}

