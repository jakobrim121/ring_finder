/* Unobstructed spherical-cap acceptance and its incidence derivative. */
static const double z0 = 34.597 / 53.;
static inline double threshold2(void) {
    return 1. - z0 * z0;
}
static inline double cap_value(double mu) {
    double t2 = threshold2(), threshold = sqrt(t2);
    if (mu >= threshold)
        return mu;
    if (mu <= -threshold)
        return 0.;
    double h = sqrt(fmax(t2 - mu * mu, 0.));
    double phi = atan2(h, -mu * z0);
    return fmax((atan2(h, z0) - z0 * h + mu * t2 * phi) / (M_PI * t2), 0.);
}
static inline void vector_cap(Vec mu, Vec *value, Vec *derivative) {
    const double t2 = threshold2(), threshold = sqrt(t2);
    Vec zero = fill(0.), one = fill(1.);
    *value = maximum(mu, zero);
    *derivative = choose((Vec)(mu > zero), one, zero);
    Vec partial = (Vec)((mu < fill(threshold)) & (mu > fill(-threshold)));
    if (any_lane(partial)) {
        Vec h = square_root(maximum(fill(t2) - mu * mu, zero));
        Vec phi = v_atan2(h, -mu * fill(z0));
        Vec v = (v_atan2(h, fill(z0)) - fill(z0) * h + mu * fill(t2) * phi) /
                fill(M_PI * t2);
        Vec d =
            (fill(t2) * phi + mu * fill(z0) * h / (one - mu * mu)) / fill(M_PI * t2);
        *value = choose(partial, maximum(v, zero), *value);
        *derivative = choose(partial, minimum(maximum(d, zero), one), *derivative);
    }
}
