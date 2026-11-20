/* Finite-track quadrature on one CPU.
 *
 * Each vector lane is an independent quadrature node. The physics and scalar
 * accumulation order are shared by both widths. Only this short adapter knows
 * the CPU instruction names; the integral below is ordinary vector algebra.
 *
 * Angular panels change quadrature and floating-point rounding; this is the
 * accepted, event-validated occupancy path, not an exact-arithmetic identity.
 * Full event regression, not an exactness claim, qualifies this backend.
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#include "vector_ops.h"
#define UNARY(name)                                                                    \
    extern Vec VEC_UNARY(name)(Vec);                                                   \
    static inline Vec v_##name(Vec x) {                                                \
        return VEC_UNARY(name)(x);                                                     \
    }
UNARY(exp)
UNARY(sin)
UNARY(cos)
static inline Vec v_atan2(Vec x, Vec y) {
    Vec r;
    for (int j = 0; j < LANES; ++j) r[j] = atan2(x[j], y[j]);
    return r;
}
static inline Vec take(const double *a, size_t j, size_t n) {
    Vec result;
    for (size_t lane = 0; lane < LANES; lane++)
        result[lane] = a[j + lane < n ? j + lane : n - 1];
    return result;
}

#if LANES == 8
#include "vector_transpose.h"
#endif

#include "cap_response.h"
/* Panel boundaries use scalar libm arithmetic in the original integral.
 * The optional helpers pack that same verified operation sequence. The node
 * integral below retains its existing GNU math and summation order. */
#ifdef ORDERED_MATH
#define MATH_LANES LANES
#include "ordered_math.h"
#include "acos_inline.h"
#endif

typedef struct {
    double d[3], radial[3], along, radius, parallel, radial_dot;
    double lower, upper, exposure_angle, phase, amplitude;
    double negative_boundary, positive_boundary;
} ReceiverGeometry;

static int receiver_geometry(ReceiverGeometry *g, size_t count, size_t first,
                             const double *positions, const double *normals,
                             const double *x, const double *axis, double length) {
    /* Preserve the scalar coordinate arithmetic and the robust system hypot.
     * Only independent transcendental calls are packed across receivers. */
    for (size_t lane = 0; lane < count; lane++) {
        ReceiverGeometry *r = &g[lane];
        size_t i = first + lane;
        r->along = 0.;
        double rho2 = 0.;
        r->parallel = 0.;
        r->radial_dot = 0.;
        for (int c = 0; c < 3; c++) {
            r->d[c] = positions[3 * i + c] - 1000. * x[c];
            r->along += r->d[c] * axis[c];
        }
        for (int c = 0; c < 3; c++) {
            r->radial[c] = r->d[c] - r->along * axis[c];
            rho2 += r->radial[c] * r->radial[c];
        }
        r->radius = sqrt(rho2);
        if (r->radius < 1.)
            return 1;
        for (int c = 0; c < 3; c++) {
            r->parallel += normals[3 * i + c] * axis[c];
            r->radial_dot += normals[3 * i + c] * r->radial[c] / r->radius;
        }
        r->amplitude = sqrt(r->parallel * r->parallel + r->radial_dot * r->radial_dot);
    }
#ifdef ORDERED_MATH
    Vec radius, lo, hi, parallel, radial, positive, negative;
    for (size_t lane = 0; lane < LANES; lane++) {
        const ReceiverGeometry *r = &g[lane < count ? lane : count - 1];
        radius[lane] = r->radius;
        lo[lane] = r->along + .5 * length;
        hi[lane] = r->along - .5 * length;
        parallel[lane] = r->parallel;
        radial[lane] = r->radial_dot;
        /* Inactive boundaries get benign inputs; they are never consumed. */
        int active = r->amplitude > sqrt(threshold2());
        positive[lane] = active ? sqrt(threshold2()) / r->amplitude : 0.;
        negative[lane] = active ? -sqrt(threshold2()) / r->amplitude : 0.;
    }
    Vec lower = ordered_atan2(radius, lo), upper = ordered_atan2(radius, hi);
    Vec exposure = ordered_atan2(radius, fill(100.));
    Vec phase = ordered_atan2(radial, parallel);
    Vec pos = ordered_acos(positive), neg = ordered_acos(negative);
    for (size_t lane = 0; lane < count; lane++) {
        g[lane].lower = lower[lane];
        g[lane].upper = upper[lane];
        g[lane].exposure_angle = exposure[lane];
        g[lane].phase = phase[lane];
        g[lane].negative_boundary = neg[lane];
        g[lane].positive_boundary = pos[lane];
    }
#else
    for (size_t lane = 0; lane < count; lane++) {
        ReceiverGeometry *r = &g[lane];
        r->lower = atan2(r->radius, r->along + .5 * length);
        r->upper = atan2(r->radius, r->along - .5 * length);
        r->exposure_angle = atan2(r->radius, 100.);
        r->phase = atan2(r->radial_dot, r->parallel);
        if (r->amplitude > sqrt(threshold2())) {
            r->negative_boundary = acos(-sqrt(threshold2()) / r->amplitude);
            r->positive_boundary = acos(sqrt(threshold2()) / r->amplitude);
        }
    }
#endif
    return 0;
}

static void add(double *edges, int *n, double value, double lo, double hi) {
    if (lo < value && value < hi)
        edges[(*n)++] = value;
}
/* Split at cone scales, angular clipping, and receiver-cap boundaries. The
 * cubic endpoint map resolves the cap's nonanalytic derivatives efficiently. */
static int angular_edges(double *edges, const ReceiverGeometry *g, double opening,
                         double width) {
    double lo = fmax(g->lower, fmax(0., opening - 12. * width));
    double hi = fmin(g->upper, fmin(M_PI, opening + 12. * width));
    if (hi <= lo)
        return 0;
    int n = 2;
    edges[0] = lo;
    edges[1] = hi;
    for (int z = -8; z <= 8; z += 4)
        add(edges, &n, opening + z * width, lo, hi);
    double clip = acos(1. - 1e-10);
    add(edges, &n, clip, lo, hi);
    add(edges, &n, M_PI - clip, lo, hi);
    /* The softened inverse-square exposure varies on the 100-mm scale. */
    double scale = g->exposure_angle;
    add(edges, &n, scale, lo, hi);
    add(edges, &n, M_PI - scale, lo, hi);
    double amplitude = g->amplitude, phase = g->phase;
    if (amplitude > sqrt(threshold2())) {
        for (int sign = -1; sign <= 1; sign += 2) {
            double angle = sign < 0 ? g->negative_boundary : g->positive_boundary;
            for (int side = -1; side <= 1; side += 2) {
                double boundary = phase + side * angle;
                if (boundary < 0.)
                    boundary += 2. * M_PI;
                if (boundary >= 2. * M_PI)
                    boundary -= 2. * M_PI;
                add(edges, &n, boundary, lo, hi);
            }
        }
    }
    /* There are only a few landmarks. Sort their values directly, using
     * the same comparisons, without a generic comparator/function dispatch.
     * Equal boundaries remain equal; zero-width panels are still skipped. */
    for (int j = 1; j < n; j++) {
        double value = edges[j];
        int k = j;
        while (k > 0 && edges[k - 1] > value) {
            edges[k] = edges[k - 1];
            k--;
        }
        edges[k] = value;
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

static __attribute__((always_inline)) inline int
integrate_order(size_t n, const double *positions, const double *normals,
                const double *x, size_t order, const double *nodes,
                const double *weights, double *values, double *jacobian,
                int64_t *counts) {
    double length = x[6] * 1000., norm = sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]);
    if (length < .1)
        return 1;
    double variance = exp(2. * x[8]), width = sqrt(variance), opening = x[7];
    double clip = acos(1. - 1e-10), a[3] = {x[3] / norm, x[4] / norm, x[5] / norm};
    Vec axis[3] = {fill(a[0]), fill(a[1]), fill(a[2])}, zero = fill(0.), one = fill(1.);
    size_t batches = (order + LANES - 1) / LANES;
    Vec node[batches], curve[batches], rest[batches], node_weight[batches];
    for (size_t b = 0; b < batches; b++) {
        node[b] = take(nodes, b * LANES, order) + fill(.5);
        curve[b] = fill(3.) - fill(2.) * node[b];
        rest[b] = one - node[b];
        node_weight[b] = take(weights, b * LANES, order);
    }
    ReceiverGeometry group[LANES];
    for (size_t i = 0; i < n; i++) {
        if (i % LANES == 0) {
            size_t count = n - i < LANES ? n - i : LANES;
            if (receiver_geometry(group, count, i, positions, normals, x, a, length))
                return 1;
        }
        const ReceiverGeometry *g = &group[i % LANES];
        const double *d = g->d, *r = g->radial;
        double t = g->along, radius = g->radius;
        Vec normal[3], radial[3], rho = fill(radius), along = fill(t);
        for (int c = 0; c < 3; c++) {
            normal[c] = fill(normals[3 * i + c]);
            radial[c] = fill(r[c] / radius);
        }
        double edges[32];
        int ne = angular_edges(edges, g, opening, width);
        values[i] = 0.;
        counts[i] = 0;
        for (int c = 0; c < 9; c++)
            jacobian[9 * i + c] = 0.;
        for (int panel = 0; panel < ne - 1; panel++) {
            double span = edges[panel + 1] - edges[panel];
            if (span <= 0.)
                continue;
            counts[i] += order;
            for (size_t j = 0; j < order; j += LANES) {
                size_t batch = j / LANES;
                Vec v = node[batch];
                Vec theta = fill(edges[panel]) + fill(span) * v * v * curve[batch];
                Vec sn = v_sin(theta), cs = v_cos(theta);
                Vec distance = rho / sn, u[3], d[3];
                for (int c = 0; c < 3; c++) {
                    u[c] = radial[c] * sn + axis[c] * cs;
                    d[c] = distance * u[c];
                }
                Vec dot = normal[0] * u[0] + normal[1] * u[1] + normal[2] * u[2],
                    acceptance, dacceptance;
                vector_cap(-dot, &acceptance, &dacceptance);
                Vec denominator = distance * distance + fill(10000.);
                Vec raw = acceptance / denominator,
                    exposure = maximum(raw, fill(1e-30));
                Vec bounded = minimum(maximum(cs, fill(-1. + 1e-10)), fill(1. - 1e-10));
                Vec angle = minimum(maximum(theta, fill(clip)), fill(M_PI - clip));
                Vec residual = angle - fill(opening);
                Vec angular = v_exp(fill(-.5) * residual * residual / fill(variance));
                Vec value = exposure * angular;
                Vec slope =
                    residual / (fill(variance) * square_root(one - bounded * bounded));
                slope =
                    choose((Vec)((cs > fill(-1. + 1e-10)) & (cs < fill(1. - 1e-10))),
                           slope, zero);
                Vec fraction = (along - distance * cs) / fill(length);
                Vec weight = fill(span) * node_weight[batch] * fill(6.) * v *
                             rest[batch] * rho / (fill(length) * sn * sn);
                Vec spatial[3], rotation[3], derivative[9];
                for (int c = 0; c < 3; c++) {
                    Vec de = dacceptance * (normal[c] - dot * u[c]) /
                                 (distance * denominator) +
                             fill(2.) * exposure * d[c] / denominator;
                    de = choose((Vec)(raw > fill(1e-30)), de, zero);
                    spatial[c] = angular * de -
                                 value * slope * (axis[c] - bounded * u[c]) / distance;
                    rotation[c] =
                        fraction * fill(length) * spatial[c] + value * slope * u[c];
                    derivative[c] = spatial[c] * fill(1000.);
                }
                Vec along_rotation = rotation[0] * axis[0] + rotation[1] * axis[1] +
                                     rotation[2] * axis[2];
                for (int c = 0; c < 3; c++)
                    derivative[c + 3] =
                        (rotation[c] - along_rotation * axis[c]) / fill(norm);
                derivative[6] = fraction *
                                (spatial[0] * axis[0] + spatial[1] * axis[1] +
                                 spatial[2] * axis[2]) *
                                fill(1000.);
                derivative[7] = value * residual / fill(variance);
                derivative[8] = value * residual * residual / fill(variance);
                Vec weighted[9];
                for (int c = 0; c < 9; c++)
                    weighted[c] = weight * derivative[c];
#if LANES == 8
                // Shuffling derivative columns lets each sum retain its original
                // node order while eight independent columns accumulate together.
                Vec columns[8];
                transpose8(weighted, columns);
                Vec accumulated = _mm512_loadu_pd(jacobian + 9 * i);
                for (int lane = 0; lane < 8 && j + lane < order; lane++)
                    accumulated += columns[lane];
                _mm512_storeu_pd(jacobian + 9 * i, accumulated);
                const int first_scalar_column = 8;
#else
                const int first_scalar_column = 0;
#endif
                for (int c = first_scalar_column; c < 9; c++)
                    for (int lane = 0; lane < LANES && j + lane < order; lane++)
                        jacobian[9 * i + c] += weighted[c][lane];
                Vec weighted_value = weight * value;
                for (int lane = 0; lane < LANES && j + lane < order; lane++)
                    values[i] += weighted_value[lane];
            }
        }
        if (radius < .5 * length && length > 1e-5 * fmax(fmax(fabs(t), radius), 100.)) {
            double start =
                ray(d[0] + .5 * length * a[0], d[1] + .5 * length * a[1],
                    d[2] + .5 * length * a[2], normals[3 * i], normals[3 * i + 1],
                    normals[3 * i + 2], a[0], a[1], a[2], opening, variance);
            double end =
                ray(d[0] - .5 * length * a[0], d[1] - .5 * length * a[1],
                    d[2] - .5 * length * a[2], normals[3 * i], normals[3 * i + 1],
                    normals[3 * i + 2], a[0], a[1], a[2], opening, variance);
            double *g = jacobian + 9 * i;
            double correction = 1000. * (end - start) / length -
                                (g[0] * a[0] + g[1] * a[1] + g[2] * a[2]);
            for (int c = 0; c < 3; c++)
                g[c] += correction * a[c];
            g[6] = 1000. * (.5 * (start + end) - values[i]) / length;
        }
    }
    return 0;
}

int finite_rays(size_t n, const double *positions, const double *normals,
                const double *x, size_t order, const double *nodes,
                const double *weights, double *values, double *jacobian,
                int64_t *counts) {
    if (order == 16)
        return integrate_order(n, positions, normals, x, 16, nodes, weights, values,
                               jacobian, counts);
    if (order == 32)
        return integrate_order(n, positions, normals, x, 32, nodes, weights, values,
                               jacobian, counts);
    return integrate_order(n, positions, normals, x, order, nodes, weights, values,
                           jacobian, counts);
}
