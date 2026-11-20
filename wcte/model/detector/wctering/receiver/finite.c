/* Finite-track quadrature on one CPU.
 *
 * Each vector lane is an independent quadrature node. The physics and scalar
 * accumulation order are shared by both widths. Only this short adapter knows
 * the CPU instruction names; the integral below is ordinary vector algebra.
 *
 * GNU vector libm changes last-bit rounding relative to V101's scalar library.
 * Full event regression, not an exactness claim, qualifies this backend.
 */
#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#define APPLE_NEON 1
#else
#include <immintrin.h>
#define APPLE_NEON 0
#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#include "vector_ops.h"
/* The scalar reference and vector libm paths share every panel, derivative,
 * and accumulation below. Only transcendental calls differ. Reference builds
 * preserve scalar rounding, with optional verified SIMD math. */
#ifdef REFERENCE_MATH
#define UNARY(name)                                                                    \
    static inline Vec v_##name(Vec x) {                                                \
        Vec result;                                                                    \
        for (int lane = 0; lane < LANES; lane++)                                       \
            result[lane] = name(x[lane]);                                              \
        return result;                                                                 \
    }
#ifndef ORDERED_MATH
UNARY(acos)
#endif
#if defined(ORDERED_MATH) && LANES == 8
#include "ordered_tan.h"
#define v_tan ordered_tan
#else
UNARY(tan)
#endif
#if defined(ORDERED_MATH) && LANES == 8
#include "ordered_sin.h"
#define v_sin ordered_sin
#else
UNARY(sin)
#endif
#ifdef ORDERED_MATH
#define MATH_LANES LANES
#include "ordered_math.h"
#include "acos_inline.h"
#define v_acos ordered_acos
#define v_exp ordered_exp
#define v_atan2 ordered_atan2
#else
UNARY(exp)
static inline Vec v_atan2(Vec x, Vec y) {
    Vec result;
    for (int lane = 0; lane < LANES; lane++)
        result[lane] = atan2(x[lane], y[lane]);
    return result;
}
#endif
#else
#define UNARY(name)                                                                    \
    extern Vec VEC_UNARY(name)(Vec);                                                   \
    static inline Vec v_##name(Vec x) {                                                \
        return VEC_UNARY(name)(x);                                                     \
    }
/* Older vector libm omits some inverse-trigonometric functions. */
#define SCALAR_UNARY(name) \
    static inline Vec v_##name(Vec x) { \
        Vec r; \
        for (int j = 0; j < LANES; ++j) r[j] = name(x[j]); \
        return r; \
    }
#ifdef WCTE_SCALAR_ACOS
SCALAR_UNARY(acos)
#else
UNARY(acos)
#endif
UNARY(exp)
#ifdef WCTE_SCALAR_TAN
SCALAR_UNARY(tan)
#else
UNARY(tan)
#endif
UNARY(sin)
#ifdef WCTE_SCALAR_ATAN2
static inline Vec v_atan2(Vec x, Vec y) {
    Vec r;
    for (int j = 0; j < LANES; ++j) r[j] = atan2(x[j], y[j]);
    return r;
}
#else
extern Vec VEC_ATAN2(Vec, Vec);
static inline Vec v_atan2(Vec x, Vec y) {
    return VEC_ATAN2(x, y);
}
#endif
#endif
static inline Vec take(const double *a, size_t j, size_t n) {
    Vec result;
    for (size_t lane = 0; lane < LANES; lane++)
        result[lane] = a[j + lane < n ? j + lane : n - 1];
    return result;
}

#include "cap_response.h"
static void add_fraction(double *out, int *n, double t, double t0, double length) {
    double f = (t0 - t) / length;
    if (-.5 < f && f < .5 && isfinite(f))
        out[(*n)++] = f;
}
/* Receiver-specific boundaries are exactly the original scalar landmarks. */
static int panel_edges(double *out, double dx, double dy, double dz, double ax,
                       double ay, double az, double nx, double ny, double nz,
                       double length, const double *cone_tangents, double clip_tangent,
                       int near_chart) {
    double t0 = dx * ax + dy * ay + dz * az;
    double rx = dx - t0 * ax, ry = dy - t0 * ay, rz = dz - t0 * az;
    double rho2 = rx * rx + ry * ry + rz * rz, rho = sqrt(rho2);
    double parallel = nx * ax + ny * ay + nz * az, B = nx * rx + ny * ry + nz * rz;
    double c[160];
    int n = 0;
    if (near_chart) {
        c[n++] = t0 - .5 * length;
        c[n++] = t0 + .5 * length;
        c[n++] = 0.;
    } else {
        out[0] = -.5;
        out[1] = .5;
        n = 2;
        add_fraction(out, &n, 0., t0, length);
    }
    const double around[8] = {-8., -4., -2., -1., 1., 2., 4., 8.};
    if (rho > 0.) {
        for (int j = 0; j < 9; j++) {
            if (isfinite(cone_tangents[j])) {
                double t = rho / cone_tangents[j];
                if (near_chart)
                    c[n++] = t;
                else
                    add_fraction(out, &n, t, t0, length);
            }
        }
        double t = rho / clip_tangent;
        if (near_chart) {
            c[n++] = t;
            c[n++] = -t;
        } else {
            add_fraction(out, &n, t, t0, length);
            add_fraction(out, &n, -t, t0, length);
        }
        for (int j = 0; j < 8; j++) {
            double v = around[j] * rho;
            if (near_chart)
                c[n++] = v;
            else
                add_fraction(out, &n, v, t0, length);
        }
        if (near_chart) {
            double scale = 8. * rho, stop = sqrt(rho * rho + 10000.);
            for (int j = 0; j < 50; j++) {
                if (scale >= stop)
                    break;
                scale *= 4.;
                c[n++] = scale;
                c[n++] = -scale;
            }
        }
    }
    double scale = sqrt((near_chart ? rho * rho : rho2) + 10000.);
    const double around_distance[4] = {-4., -1., 1., 4.};
    for (int j = 0; j < 4; j++) {
        double t = around_distance[j] * scale;
        if (near_chart)
            c[n++] = t;
        else
            add_fraction(out, &n, t, t0, length);
    }
    double t2 = threshold2(), qa = parallel * parallel - t2, qb = 2. * parallel * B;
    double qc = near_chart ? B * B - t2 * rho * rho : B * B - t2 * rho2;
    double roots[2];
    int nr = 0;
    if (fabs(qa) < 1e-14) {
        if (fabs(qb) > 1e-30)
            roots[nr++] = -qc / qb;
    } else {
        double discr = qb * qb - 4. * qa * qc;
        if (discr >= 0.) {
            double q = -.5 * (qb + copysign(sqrt(discr), qb));
            if (q != 0.) {
                roots[nr++] = q / qa;
                roots[nr++] = qc / q;
            } else
                roots[nr++] = -qb / (2. * qa);
        }
    }
    for (int j = 0; j < nr; j++) {
        if (near_chart)
            c[n++] = roots[j];
        else
            add_fraction(out, &n, roots[j], t0, length);
    }
    if (fabs(B) > 1e-30) {
        double t = near_chart ? parallel * rho * rho / B : parallel * rho2 / B;
        if (near_chart)
            c[n++] = t;
        else
            add_fraction(out, &n, t, t0, length);
    }
    if (near_chart) {
        double lo = t0 - .5 * length, hi = t0 + .5 * length;
        int m = 0;
        for (int j = 0; j < n; j++)
            if (lo <= c[j] && c[j] <= hi && isfinite(c[j]))
                out[m++] = c[j];
        n = m;
    }
    for (int j = 1; j < n; j++) {
        double value = out[j];
        int k = j;
        while (k > 0 && out[k - 1] > value) {
            out[k] = out[k - 1];
            k--;
        }
        out[k] = value;
    }
    return n;
}
static double ray(double xx, double yy, double zz, double nx, double ny, double nz,
                  double ax, double ay, double az, double opening, double variance) {
    double r = fmax(sqrt(xx * xx + yy * yy + zz * zz), 1e-9);
    double ux = xx / r, uy = yy / r, uz = zz / r;
    double acceptance = cap_value(-(nx * ux + ny * uy + nz * uz));
    double cosine = fmin(fmax(ux * ax + uy * ay + uz * az, -1. + 1e-10), 1. - 1e-10);
    double residual = acos(cosine) - opening;
    return fmax(acceptance / (r * r + 10000.), 1e-30) *
           exp(-.5 * residual * residual / variance);
}
#if LANES == 8
#include "vector_transpose.h"
static inline void accumulate_columns(double *sum, const Vec *d, size_t count) {
    Vec node[8];
    transpose8(d, node);
    Vec accumulated = _mm512_loadu_pd(sum);
    for (size_t i = 0; i < count; i++)
        accumulated += node[i];
    _mm512_storeu_pd(sum, accumulated);
}
#endif

static __attribute__((always_inline)) inline int
integrate_order(size_t n, const double *positions, const double *normals,
                const double *x, size_t order, const double *nodes,
                const double *weights, double *values, double *jacobian,
                int64_t *counts, double tail_sigma) {
    double cx = x[0] * 1000., cy = x[1] * 1000., cz = x[2] * 1000.;
    double norm = sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]);
    double ax = x[3] / norm, ay = x[4] / norm, az = x[5] / norm;
    double length = x[6] * 1000., opening = x[7], variance = exp(2. * x[8]),
           width = sqrt(variance);
    /* Add a generous angular margin before omitting a tail. Bounds alone
     * decide omission; they never replace the original node coordinates. */
    double upper_angle = opening + tail_sigma * width + 1e-8;
    double lower_angle = opening - tail_sigma * width - 1e-8;
    int upper_tail = tail_sigma > 0. && upper_angle < M_PI;
    int lower_tail = tail_sigma > 0. && lower_angle > 0.;
    double upper_cot = upper_tail ? 1. / tan(upper_angle) : 0.;
    double lower_cot = lower_tail ? 1. / tan(lower_angle) : 0.;
    int omitted = 0;
    if (tail_sigma > 0. && length < .1)
        return -1;
    /* These landmarks depend on the source, never the receiving PMT.
     * Keep division by tan(theta), rather than multiplying by a reciprocal,
     * so every original panel boundary keeps the same rounding. */
    const double cone[9] = {-8., -4., -2., -1., 0., 1., 2., 4., 8.};
    double cone_tangents[9];
    for (int j = 0; j < 9; j++) {
        double theta = opening + cone[j] * width;
        cone_tangents[j] = (0. < theta && theta < M_PI) ? tan(theta) : NAN;
    }
    double clip_tangent = tan(acos(1. - 1e-10));
    Vec one = fill(1.), zero = fill(0.), axis[3] = {fill(ax), fill(ay), fill(az)};
    size_t batches = (order + LANES - 1) / LANES;
    Vec node_shift[batches], node_curve[batches], node_rest[batches],
        node_weight[batches];
    for (size_t b = 0; b < batches; b++) {
        Vec v = take(nodes, b * LANES, order) + fill(.5);
        node_shift[b] = v;
        node_curve[b] = fill(3.) - fill(2.) * v;
        node_rest[b] = one - v;
        node_weight[b] = take(weights, b * LANES, order);
    }
    /* Four-node screens otherwise leave half of an eight-lane packet idle.
     * Repeat its nodes for a second compatible panel, preserving node order. */
    Vec paired_shift = node_shift[0], paired_curve = node_curve[0];
    Vec paired_rest = node_rest[0], paired_weight = node_weight[0];
#if LANES == 8 && !defined(REFERENCE_MATH)
    if (order == 4)
        for (int lane = 4; lane < 8; lane++) {
            paired_shift[lane] = node_shift[0][lane - 4];
            paired_curve[lane] = node_curve[0][lane - 4];
            paired_rest[lane] = node_rest[0][lane - 4];
            paired_weight[lane] = node_weight[0][lane - 4];
        }
#endif
    for (size_t i = 0; i < n; i++) {
        double dx = positions[3 * i] - cx, dy = positions[3 * i + 1] - cy,
               dz = positions[3 * i + 2] - cz;
        double nx = normals[3 * i], ny = normals[3 * i + 1], nz = normals[3 * i + 2];
        double t0 = dx * ax + dy * ay + dz * az, rx = dx - t0 * ax, ry = dy - t0 * ay,
               rz = dz - t0 * az;
        double rho = sqrt(rx * rx + ry * ry + rz * rz);
        if (tail_sigma > 0. && rho < 1.)
            return -1;
        double upper_t = rho * upper_cot, lower_t = rho * lower_cot;
        if ((upper_tail && t0 + .5 * length < upper_t) ||
            (lower_tail && t0 - .5 * length > lower_t)) {
            values[i] = 0.;
            counts[i] = 0;
            for (int c = 0; c < 9; c++)
                jacobian[9 * i + c] = 0.;
            omitted++;
            continue;
        }
        int near_chart = 0. < rho && rho < .5 * length;
        double edges[160];
        int ne = panel_edges(edges, dx, dy, dz, ax, ay, az, nx, ny, nz, length,
                             cone_tangents, clip_tangent, near_chart);
        values[i] = 0.;
        counts[i] = 0;
        for (int c = 0; c < 9; c++)
            jacobian[9 * i + c] = 0.;
        for (int panel = 0; panel < ne - 1; panel++) {
            double left = edges[panel], right = edges[panel + 1];
            double highest_t = near_chart ? right : t0 - left * length;
            double lowest_t = near_chart ? left : t0 - right * length;
            if ((upper_tail && highest_t < upper_t) ||
                (lower_tail && lowest_t > lower_t)) {
                omitted++;
                continue;
            }
            int angular_chart =
                near_chart && fmax(fabs(left), fabs(right)) <= 8. * rho * (1. + 1e-14);
            if (angular_chart) {
                left = atan2(rho, edges[panel + 1]);
                right = atan2(rho, edges[panel]);
            }
            double span = right - left;
            if (span <= 0.)
                continue;
            int paired = 0;
            size_t packet_nodes = order;
            Vec panel_left = fill(left), panel_span = fill(span);
#if LANES == 8 && !defined(REFERENCE_MATH)
            if (order == 4 && panel + 2 < ne) {
                double next_left = edges[panel + 1], next_right = edges[panel + 2];
                int next_chart =
                    near_chart &&
                    fmax(fabs(next_left), fabs(next_right)) <= 8. * rho * (1. + 1e-14);
                if (next_chart == angular_chart) {
                    if (next_chart) {
                        next_left = atan2(rho, edges[panel + 2]);
                        next_right = atan2(rho, edges[panel + 1]);
                    }
                    double next_span = next_right - next_left;
                    if (next_span > 0.) {
                        paired = 1;
                        packet_nodes = 8;
                        panel++;
                        for (int lane = 4; lane < 8; lane++) {
                            panel_left[lane] = next_left;
                            panel_span[lane] = next_span;
                        }
                    }
                }
            }
#endif
            counts[i] += packet_nodes;
            for (size_t j = 0; j < packet_nodes; j += LANES) {
                size_t b = j / LANES;
                Vec v = paired ? paired_shift : node_shift[b];
                Vec curve = paired ? paired_curve : node_curve[b];
                Vec rest = paired ? paired_rest : node_rest[b];
                Vec quadrature_weight = paired ? paired_weight : node_weight[b];
                Vec abscissa = panel_left + panel_span * v * v * curve;
                Vec weight = panel_span * quadrature_weight * fill(6.) * v * rest;
                Vec f, xx, yy, zz;
                if (angular_chart) {
                    Vec t = fill(rho) / v_tan(abscissa);
                    f = (fill(t0) - t) / fill(length);
                    Vec sn = v_sin(abscissa);
                    weight *= fill(rho) / (fill(length) * (sn * sn));
                    xx = fill(rx) + t * axis[0];
                    yy = fill(ry) + t * axis[1];
                    zz = fill(rz) + t * axis[2];
                } else if (near_chart) {
                    Vec t = abscissa;
                    f = (fill(t0) - t) / fill(length);
                    weight /= fill(length);
                    xx = fill(rx) + t * axis[0];
                    yy = fill(ry) + t * axis[1];
                    zz = fill(rz) + t * axis[2];
                } else {
                    f = abscissa;
                    xx = fill(dx) - f * fill(length) * axis[0];
                    yy = fill(dy) - f * fill(length) * axis[1];
                    zz = fill(dz) - f * fill(length) * axis[2];
                }
                Vec r = maximum(square_root(xx * xx + yy * yy + zz * zz), fill(1e-9));
                Vec ux = xx / r, uy = yy / r, uz = zz / r;
                Vec rawcos = ux * axis[0] + uy * axis[1] + uz * axis[2];
                Vec cosine =
                    minimum(maximum(rawcos, fill(-1. + 1e-10)), fill(1. - 1e-10));
                Vec sine = square_root(one - cosine * cosine);
                Vec residual = v_acos(cosine) - fill(opening);
                Vec angular = v_exp(fill(-.5) * residual * residual / fill(variance));
#ifndef REFERENCE_MATH
                /* No cutoff approximation: the installed exponential has
                 * already returned zero in every lane. Remaining light and
                 * derivative contributions are zero for the bounded geometry.
                 * Panel/node counts were recorded above and stay unchanged. */
                if (!any_lane((Vec)(angular != zero)))
                    continue;
#endif
                Vec dot = fill(nx) * ux + fill(ny) * uy + fill(nz) * uz, acceptance,
                    dacceptance;
                vector_cap(-dot, &acceptance, &dacceptance);
                Vec denominator = r * r + fill(10000.), raw = acceptance / denominator;
                Vec exposure = maximum(raw, fill(1e-30));
                Vec value = exposure * angular,
                    slope = residual / (fill(variance) * sine);
                slope = choose(
                    (Vec)((rawcos > fill(-1. + 1e-10)) & (rawcos < fill(1. - 1e-10))),
                    slope, zero);
                Vec first = dacceptance / (r * denominator),
                    second = fill(2.) * exposure / denominator;
                Vec ex = first * (fill(nx) - dot * ux) + second * xx;
                Vec ey = first * (fill(ny) - dot * uy) + second * yy;
                Vec ez = first * (fill(nz) - dot * uz) + second * zz;
                Vec mask = (Vec)(raw > fill(1e-30));
                ex = choose(mask, ex, zero);
                ey = choose(mask, ey, zero);
                ez = choose(mask, ez, zero);
                Vec sx = angular * ex - value * slope * (axis[0] - cosine * ux) / r;
                Vec sy = angular * ey - value * slope * (axis[1] - cosine * uy) / r;
                Vec sz = angular * ez - value * slope * (axis[2] - cosine * uz) / r;
                Vec vx = f * fill(length) * sx + value * slope * ux;
                Vec vy = f * fill(length) * sy + value * slope * uy;
                Vec vz = f * fill(length) * sz + value * slope * uz;
                Vec va = vx * axis[0] + vy * axis[1] + vz * axis[2];
                Vec d[9] = {weight * sx * fill(1000.),
                            weight * sy * fill(1000.),
                            weight * sz * fill(1000.),
                            weight * (vx - va * axis[0]) / fill(norm),
                            weight * (vy - va * axis[1]) / fill(norm),
                            weight * (vz - va * axis[2]) / fill(norm),
                            weight * f * (sx * axis[0] + sy * axis[1] + sz * axis[2]) *
                                fill(1000.),
                            weight * value * residual / fill(variance),
                            weight * value * residual * residual / fill(variance)};
                Vec weighted_value = weight * value;
                /* Scalar node order is deliberate: a horizontal SIMD sum
                 * would change rounding and can change a fitted ring count. */
#if LANES == 8
                accumulate_columns(jacobian + 9 * i, d,
                                   packet_nodes - j < 8 ? packet_nodes - j : 8);
                const int first_scalar_column = 8;
#else
                const int first_scalar_column = 0;
#endif
                for (int c = first_scalar_column; c < 9; c++) {
                    double sum = jacobian[9 * i + c];
                    for (size_t lane = 0; lane < LANES && j + lane < packet_nodes;
                         lane++)
                        sum += d[c][lane];
                    jacobian[9 * i + c] = sum;
                }
                double sum = values[i];
                for (size_t lane = 0; lane < LANES && j + lane < packet_nodes; lane++)
                    sum += weighted_value[lane];
                values[i] = sum;
            }
        }
        if (near_chart && length > 1e-5 * fmax(fmax(fabs(t0), rho), 100.)) {
            double start =
                ray(dx + .5 * length * ax, dy + .5 * length * ay, dz + .5 * length * az,
                    nx, ny, nz, ax, ay, az, opening, variance);
            double end =
                ray(dx - .5 * length * ax, dy - .5 * length * ay, dz - .5 * length * az,
                    nx, ny, nz, ax, ay, az, opening, variance);
            double parallel = jacobian[9 * i] * ax + jacobian[9 * i + 1] * ay +
                              jacobian[9 * i + 2] * az;
            double correction = 1000. * (end - start) / length - parallel;
            jacobian[9 * i] += correction * ax;
            jacobian[9 * i + 1] += correction * ay;
            jacobian[9 * i + 2] += correction * az;
            jacobian[9 * i + 6] = 1000. * (.5 * (start + end) - values[i]) / length;
        }
    }
    return omitted;
}

static int integrate(size_t n, const double *positions, const double *normals,
                     const double *x, size_t order, const double *nodes,
                     const double *weights, double *values, double *jacobian,
                     int64_t *counts, double sigma) {
    if (order == 16)
        return integrate_order(n, positions, normals, x, 16, nodes, weights, values,
                               jacobian, counts, sigma);
    if (order == 32)
        return integrate_order(n, positions, normals, x, 32, nodes, weights, values,
                               jacobian, counts, sigma);
    return integrate_order(n, positions, normals, x, order, nodes, weights, values,
                           jacobian, counts, sigma);
}

static void evaluate_tail(size_t n, const double *positions, const double *normals,
                          const double *x, size_t order, const double *nodes,
                          const double *weights, double *values, double *jacobian,
                          int64_t *counts, double sigma) {
#ifndef REFERENCE_MATH
    sigma = 0.;
#endif
    int omitted = integrate(n, positions, normals, x, order, nodes, weights, values,
                            jacobian, counts, sigma);
    if (omitted == 0)
        return;
    double total = 0.;
    if (omitted > 0)
        for (size_t i = 0; i < n; i++)
            total += values[i];
    /* In an almost-dark pattern, even a tiny absolute tail could dominate
     * normalization. Retain every original node in that case. */
    if (omitted < 0 || !isfinite(total) || total < 1e-16)
        integrate(n, positions, normals, x, order, nodes, weights, values, jacobian,
                  counts, 0.);
}
void finite_rays(size_t n, const double *positions, const double *normals,
                 const double *x, size_t order, const double *nodes,
                 const double *weights, double *values, double *jacobian,
                 int64_t *counts) {
    evaluate_tail(n, positions, normals, x, order, nodes, weights, values, jacobian,
                  counts, APPLE_NEON ? 0. : 10.);
}
