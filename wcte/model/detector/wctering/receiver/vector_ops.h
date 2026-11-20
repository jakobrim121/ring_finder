#if defined(APPLE_NEON) && APPLE_NEON
/* ARM adapter for the existing single-thread double-precision CSDA kernel. */
#define LANES 2
typedef double Vec __attribute__((vector_size(16)));
static inline Vec fill(double x) { return vdupq_n_f64(x); }
static inline Vec maximum(Vec a, Vec b) { return vmaxq_f64(a,b); }
static inline Vec minimum(Vec a, Vec b) { return vminq_f64(a,b); }
static inline Vec square_root(Vec a) { return vsqrtq_f64(a); }
static inline int any_lane(Vec a) {
    uint64x2_t bits=vreinterpretq_u64_f64(a);
    return (vgetq_lane_u64(bits,0) != 0 ? 1 : 0) | (vgetq_lane_u64(bits,1) != 0 ? 2 : 0);
}
static inline Vec choose(Vec mask, Vec yes, Vec no) {
    return vbslq_f64(vreinterpretq_u64_f64(mask),yes,no);
}
#else
/* Single-CPU vector operations shared by the two finite integrators. */
#ifndef LANES
#define LANES 4
#endif
typedef double Vec __attribute__((vector_size(8 * LANES)));

/* Platform adapters: broadcast, elementwise operations, and lane selection. */
static inline Vec fill(double x) {
    Vec result;
    for (int lane = 0; lane < LANES; lane++)
        result[lane] = x;
    return result;
}
#if LANES == 8
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
#define VEC_UNARY(name) _ZGVeN8v_##name
#define VEC_ATAN2 _ZGVeN8vv_atan2
#elif LANES == 4
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
#define VEC_UNARY(name) _ZGVdN4v_##name
#define VEC_ATAN2 _ZGVdN4vv_atan2
#else
#error Unsupported vector width
#endif
#endif
