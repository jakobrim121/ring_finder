"""Stable evaluation of the existing spherical scattering probability.

No physics or fitted parameter is added. Short arcs can have concentration
above 1e8. Subtracting nearly equal cosines or cumulative scattering values
then loses useful digits. The formulas below use algebraic differences and
the small complement of the mean cosine instead.
"""
import numpy as np

SERIES = np.array([1/3, -1/45, 2/945, -1/4725, 2/93555,
                   -1382/638512875, 4/18243225, -3617/162820783125])


def moments(k):
    """Mean cosine, its complement, and derivative for the spherical vMF."""
    if k < .25:
        mean = k*np.polynomial.polynomial.polyval(k*k, SERIES)
        slope = np.polynomial.polynomial.polyval(k*k, SERIES*np.arange(1,16,2))
        return mean, 1-mean, slope
    e = np.exp(-2*k); den = -np.expm1(-2*k)
    complement = 1/k-2*e/den
    return 1-complement, complement, 1/(k*k)-4*e/(den*den)


def concentration(q):
    """Solve mean cosine = exp(-q), without subtracting numbers near one."""
    if q <= 0: raise ValueError('A resolved scattering arc needs positive variance')
    a = np.exp(-q); b = -np.expm1(-q)
    if b < .04:
        # Here k>25: the omitted correction is below 4e-22 in the mean.
        k = 1/b
        return k, -a*k*k
    if a < 1e-10: return 3*a, -3*a
    k = a*(3-a*a)/(b*(2-b))
    for _ in range(30):
        mean, complement, slope = moments(k)
        residual = mean-a if k < 1 else b-complement
        new = max(k-residual/slope, 1e-300)
        if abs(new-k) <= 4*np.finfo(float).eps*max(1., k):
            k = new; break
        k = new
    return k, -a/moments(k)[2]


def angular(a, b, q):
    """Log density, unit-vector derivatives and variance derivative.

For unit vectors, a.b-1 = -|a-b|^2/2. Differentiating this expression before
projecting through the normalization Jacobian also avoids cancellation in
nearly parallel tangent derivatives.
    """
    k, dk = concentration(q); delta = a-b
    cosine_loss = -.5*float(delta@delta)
    _, complement, _ = moments(k)
    if k < 1e-3:
        value = -np.log(4*np.pi)-k*k/6+k**4/180+k*(1+cosine_loss)
    else:
        value = np.log(k)-np.log(2*np.pi)+k*cosine_loss-np.log(-np.expm1(-2*k))
    return float(value), -k*delta, k*delta, float((cosine_loss+complement)*dk)


def spline_difference(spline, a, b):
    """Evaluate a scalar piecewise polynomial's f(b)-f(a) accurately.

Within each interval, factor b-a out of the polynomial difference. Across
knots, sum the individual interval changes. In particular, the large
constant term of a cumulative scattering table is never subtracted.
    """
    if a == b: return 0.
    if b < a: return -spline_difference(spline, b, a)
    knots, coefficients = spline.x, spline.c
    first = int(np.clip(np.searchsorted(knots, a, side='right')-1, 0, len(knots)-2))
    last = int(np.clip(np.searchsorted(knots, b, side='left')-1, 0, len(knots)-2))
    indices = np.arange(first, last+1)
    left = np.r_[a, knots[first+1:last+1]]-knots[indices]
    right = np.r_[knots[first+1:last+1], b]-knots[indices]
    degree = len(coefficients)-1; divided = np.zeros(len(indices))
    for row in range(degree):
        power = degree-row
        divided += coefficients[row, indices]*sum(left**j*right**(power-1-j) for j in range(power))
    return float(np.sum((right-left)*divided))
