/* Four-parameter charge-weighted point-source timing fit.
 * Coordinates are millimetres; times are nanoseconds. The last parameter is
 * emission time. Keep arithmetic order, damping and stopping rules identical
 * to the frozen native kernel: its result supplies trained counter features.
 */

#include <math.h>
static int solve4(double A[4][4], double b[4], double out[4]) {
    int i, j, k, piv;
    double maxv, tmp, f;
    for (i = 0; i < 4; i++)
        out[i] = b[i];
    for (k = 0; k < 4; k++) {
        piv = k;
        maxv = fabs(A[k][k]);
        for (i = k + 1; i < 4; i++) {
            if (fabs(A[i][k]) > maxv) {
                maxv = fabs(A[i][k]);
                piv = i;
            }
        }
        if (maxv < 1e-300)
            return -1;
        if (piv != k) {
            for (j = 0; j < 4; j++) {
                tmp = A[k][j];
                A[k][j] = A[piv][j];
                A[piv][j] = tmp;
            }
            tmp = out[k];
            out[k] = out[piv];
            out[piv] = tmp;
        }
        for (i = k + 1; i < 4; i++) {
            f = A[i][k] / A[k][k];
            for (j = k; j < 4; j++)
                A[i][j] -= f * A[k][j];
            out[i] -= f * out[k];
        }
    }
    for (i = 3; i >= 0; i--) {
        tmp = out[i];
        for (j = i + 1; j < 4; j++)
            tmp -= A[i][j] * out[j];
        out[i] = tmp / A[i][i];
    }
    return 0;
}
static double cost_h_g(const double *p, const double *t, const double *sqrtw, double vc,
                       int n, const double *x, double H[4][4], double g[4]) {
    int i, a, b;
    double cost = 0.0;
    for (a = 0; a < 4; a++) {
        g[a] = 0.0;
        for (b = 0; b < 4; b++)
            H[a][b] = 0.0;
    }
    for (i = 0; i < n; i++) {
        double dx = p[3 * i] - x[0], dy = p[3 * i + 1] - x[1], dz = p[3 * i + 2] - x[2];
        double d = sqrt(dx * dx + dy * dy + dz * dz);
        if (d < 1e-12)
            d = 1e-12;
        double sw = sqrtw[i];
        double r = sw * (t[i] - x[3] - d / vc);
        cost += r * r;
        double f = sw / (d * vc);
        double j0 = f * dx, j1 = f * dy, j2 = f * dz, j3 = -sw;
        H[0][0] += j0 * j0;
        H[0][1] += j0 * j1;
        H[0][2] += j0 * j2;
        H[0][3] += j0 * j3;
        H[1][1] += j1 * j1;
        H[1][2] += j1 * j2;
        H[1][3] += j1 * j3;
        H[2][2] += j2 * j2;
        H[2][3] += j2 * j3;
        H[3][3] += j3 * j3;
        g[0] += j0 * r;
        g[1] += j1 * r;
        g[2] += j2 * r;
        g[3] += j3 * r;
    }
    H[1][0] = H[0][1];
    H[2][0] = H[0][2];
    H[3][0] = H[0][3];
    H[2][1] = H[1][2];
    H[3][1] = H[1][3];
    H[3][2] = H[2][3];
    return cost;
}
void bounded_lm(const double *p, const double *t, const double *sqrtw, double vc,
                const double *x0, int n, int max_nfev, double *xout, double *cost_out,
                double *Hout) {
    double lower[4] = {-5000, -5000, -5000, -1000}, upper[4] = {5000, 5000, 5000, 1000};
    double x[4], H[4][4], g[4];
    int i, k, it, trial;
    for (k = 0; k < 4; k++) {
        double v = x0[k], lo = lower[k] + 1e-6, hi = upper[k] - 1e-6;
        if (v < lo)
            v = lo;
        if (v > hi)
            v = hi;
        x[k] = v;
    }
    double cost = cost_h_g(p, t, sqrtw, vc, n, x, H, g);
    double lam = 1e-3;
    for (it = 0; it < max_nfev; it++) {
        double gmax = 0.0;
        for (k = 0; k < 4; k++) {
            double a = fabs(g[k]);
            if (a > gmax)
                gmax = a;
        }
        if (gmax < 2e-6)
            break;
        double old_cost = cost;
        int accepted = 0;
        double step[4] = {0, 0, 0, 0};
        for (trial = 0; trial < 12; trial++) {
            double Hlm[4][4], b[4], diag[4];
            for (k = 0; k < 4; k++) {
                diag[k] = H[k][k];
                if (diag[k] < 1e-12)
                    diag[k] = 1e-12;
            }
            for (i = 0; i < 4; i++) {
                for (k = 0; k < 4; k++)
                    Hlm[i][k] = H[i][k];
                b[i] = -g[i];
            }
            for (k = 0; k < 4; k++)
                Hlm[k][k] += lam * diag[k];
            if (solve4(Hlm, b, step) != 0) {
                for (k = 0; k < 4; k++)
                    Hlm[k][k] += (lam + 1.0) * 10.0;
                for (i = 0; i < 4; i++)
                    b[i] = -g[i];
                solve4(Hlm, b, step);
            }
            double xt[4];
            for (k = 0; k < 4; k++) {
                double v = x[k] + step[k], lo = lower[k] + 1e-6, hi = upper[k] - 1e-6;
                if (v < lo)
                    v = lo;
                if (v > hi)
                    v = hi;
                xt[k] = v;
            }
            double Ht[4][4], gt[4];
            double trial_cost = cost_h_g(p, t, sqrtw, vc, n, xt, Ht, gt);
            if (trial_cost <= cost) {
                for (k = 0; k < 4; k++) {
                    x[k] = xt[k];
                    g[k] = gt[k];
                    for (i = 0; i < 4; i++)
                        H[k][i] = Ht[k][i];
                }
                cost = trial_cost;
                lam = lam * 0.33;
                if (lam < 1e-12)
                    lam = 1e-12;
                accepted = 1;
                break;
            }
            lam *= 10.0;
            if (lam > 1e12)
                break;
        }
        double sn = 0.0;
        for (k = 0; k < 4; k++)
            sn += step[k] * step[k];
        sn = sqrt(sn);
        double xn = 0.0;
        for (k = 0; k < 4; k++)
            xn += x[k] * x[k];
        xn = sqrt(xn);
        if (sn < 2e-6 * (2e-6 + xn))
            break;
        if (fabs(old_cost - cost) < 2e-6 * (old_cost > 1.0 ? old_cost : 1.0))
            break;
        if (!accepted)
            break;
    }
    for (k = 0; k < 4; k++)
        xout[k] = x[k];
    *cost_out = cost;
    for (i = 0; i < 4; i++)
        for (k = 0; k < 4; k++)
            Hout[4 * i + k] = H[i][k];
}
