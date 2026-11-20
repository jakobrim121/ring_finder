/* Evaluate the calibrated knock-on angular spline in PMT lanes. */
#include <stdint.h>
#include "charged_receiver.h"

void knockon_batch(size_t count, size_t nodes, const double *positions, const double *normals,
                   const double *points, const double *axes, const double *opening,
                   const int64_t *opening_cells, const double *weights, double length,
                   const double *opening_grid, size_t n_opening, const double *polar_grid,
                   size_t n_polar, const double *coefficients, const double *cap,
                   size_t cap_count, double cap_low, double cap_step, double *values) {
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
            const double *p = points+3*j, *a = axes+3*j;
            const Vec dx = px-p[0], dy = py-p[1], dz = pz-p[2];
            const Vec distance = maximum(square_root(dx*dx+dy*dy+dz*dz),fill(1e-9));
            const Vec rx = dx/distance, ry = dy/distance, rz = dz/distance;
            const Vec c = minimum(maximum(rx*a[0]+ry*a[1]+rz*a[2],fill(-1.)),fill(1.));
            const Vec polar = v_acos(c);
            const double theta = opening[j];
            const size_t open_cell = opening_cells[j];
            const double off = fmin(fmax(theta,opening_grid[0]),opening_grid[n_opening-1])
                               -opening_grid[open_cell];
            const double *spline = coefficients+16*(n_polar-1)*open_cell;
            Vec offset, coefficient[16];
            for (size_t lane = 0; lane < LANES; ++lane) {
                const double angle = polar[lane];
                size_t cell = (size_t)(angle*(n_polar-1)/polar_grid[n_polar-1]);
                if (cell > n_polar-2) cell = n_polar-2;
                while (cell > 0 && angle <= polar_grid[cell]) --cell;
                while (cell+2 < n_polar && angle > polar_grid[cell+1]) ++cell;
                offset[lane] = angle-polar_grid[cell];
                const double *g = spline+16*cell;
                for (int k = 0; k < 16; ++k) coefficient[k][lane] = g[k];
            }
            Vec v[4];
            for (int k = 0; k < 4; ++k) {
                const Vec *g = coefficient+4*k;
                v[k] = ((g[0]*offset+g[1])*offset+g[2])*offset+g[3];
            }
            const Vec density = ((v[0]*off+v[1])*off+v[2])*off+v[3];
            const Vec dot = nx*rx+ny*ry+nz*rz;
            Vec acceptance;
            vector_cap(-dot,cap,cap_count,cap_low,cap_step,&acceptance);
            const Vec denominator = distance*distance+10000.;
            const Vec exposure = maximum(acceptance/denominator,fill(1e-30));
            const Vec flux = exposure*density;
            value += weights[j]*length*flux;
        }
        for (size_t lane = 0; lane < lanes; ++lane)
            values[i+lane] = value[lane];
    }
}
