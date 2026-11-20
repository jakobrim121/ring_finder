/* Evaluate charged-track light in adjacent PMT lanes. */
#include "charged_receiver.h"

extern Vec VEC_UNARY(exp)(Vec);
static inline Vec v_exp(Vec v) { return VEC_UNARY(exp)(v); }

void csda_batch(size_t count, size_t nodes,
                const double *positions, const double *normals, const double *weights,
                const double *points, const double *axes, const double *emission,
                const double *cosines, const double *cap, size_t cap_count,
                double cap_low, double cap_step, double *values) {
    for (size_t i = 0; i < count; i += LANES) {
        const size_t lanes = count-i < LANES ? count-i : LANES;
        const Vec px = read_column(positions, i, lanes, 0);
        const Vec py = read_column(positions, i, lanes, 1);
        const Vec pz = read_column(positions, i, lanes, 2);
        const Vec nx = read_column(normals, i, lanes, 0);
        const Vec ny = read_column(normals, i, lanes, 1);
        const Vec nz = read_column(normals, i, lanes, 2);
        Vec value = fill(0.);
        for (size_t j = 0; j < nodes; ++j) {
            const double *e = emission + 9*j;
            const double *point = points + 3*j, *a = axes + 3*j;
            const Vec dx = px-point[0], dy = py-point[1], dz = pz-point[2];
            const Vec distance = maximum(square_root(dx*dx+dy*dy+dz*dz), fill(1e-9));
            /* Reject a wholly dark block before the three ray divisions.
             * The guard exceeds the roundoff in moving the normalization
             * across the dot product. The original comparisons below
             * still decide every lane near either edge of the cone. */
            const Vec projection = dx*a[0]+dy*a[1]+dz*a[2];
            const Vec lower = distance*(cosines[2*j]-4e-14);
            const Vec upper = distance*(cosines[2*j+1]+4e-14);
            if (!any_lane((Vec)((projection >= lower) & (projection <= upper)))) continue;
            const Vec rx = dx/distance, ry = dy/distance, rz = dz/distance;
            const Vec dot_axis = rx*a[0]+ry*a[1]+rz*a[2];
            const Vec active = (Vec)((dot_axis >= fill(cosines[2*j])) &
                                     (dot_axis <= fill(cosines[2*j+1])));
            if (!any_lane(active)) continue;
            const Vec c = minimum(maximum(dot_axis, fill(-1+1e-10)), fill(1-1e-10));
            const Vec residual = choose(active,
                v_acos(c)-e[0], fill(0.));
            /* A zero-weight lane must not send the whole vector exponential
             * through a scalar underflow path or evaluate a grazing cap. */
            const Vec dot = choose(active, nx*rx+ny*ry+nz*rz, fill(-1.));
            Vec accept;
            for (size_t lane = 0; lane < LANES; ++lane) {
                double cap_value;
                cap_at(-dot[lane], cap, cap_count, cap_low, cap_step, &cap_value);
                accept[lane] = cap_value;
            }
            const Vec denominator = distance*distance+10000.;
            const Vec raw = accept/denominator;
            const Vec exposure = maximum(raw, fill(1e-30));
            const Vec angular = v_exp(-.5*residual*residual/e[6]);
            const Vec flux = exposure*angular*e[1];
            const Vec weight = choose(active, fill(weights[j]), fill(0.));
            value += weight*flux;
        }
        for (size_t lane = 0; lane < lanes; ++lane)
            values[i+lane] = value[lane];
    }
}
