"""First water exit of a circular mean trajectory, without geometric stepping.

Use u = tan(phi/2)/curvature to turn each plane or sphere intersection into
a quadratic whose coefficients have a finite straight-track limit. This
avoids subtracting enormous circle-center coordinates at small curvature.
All lengths are in metres; the bend vector is transverse to the initial axis.
"""

import numpy as np

from wctering.likelihood.geometry import NORMALS as FACES, UPPER
from wctering.likelihood.water import CENTRES, NORMALS, RADIUS, CUT


def quadratic_roots(a, b, c):
    """Real roots, using the stable quadratic formula and a linear limit."""
    a, b, c = np.broadcast_arrays(a, b, c)
    result = np.full((*a.shape, 2), np.nan)
    linear = abs(a) <= 1e-14 * np.maximum.reduce([np.ones_like(a), abs(b), abs(c)])
    np.divide(-c, b, out=result[..., 0], where=linear & (abs(b) > 1e-15))
    discriminant = b * b - 4 * a * c
    valid = ~linear & (discriminant >= 0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    q = -0.5 * (b + np.where(b >= 0, root, -root))
    np.divide(q, a, out=result[..., 0], where=valid)
    np.divide(c, q, out=result[..., 1], where=valid & (q != 0))
    return result


def points_and_tangents(origin, axis, curvature, u):
    norm2 = float(curvature @ curvature)
    denominator = 1 + norm2 * u * u
    displacement = (2 * u / denominator)[..., None] * (axis + u[..., None] * curvature)
    tangent = ((1 - norm2 * u * u)[..., None] * axis + 2 * u[..., None] * curvature) / denominator[..., None]
    return origin + displacement, tangent


def first_exit(origin, axis, bend, full_length):
    """Return the first outward crossing before the full CSDA arc ends.

    The detector domain matches the existing prism minus truncated spherical
    mPMT caps. A boundary normal is returned for implicit endpoint derivatives.
    ``None`` means the complete optical range stays within that domain.
    """
    origin, axis, bend = map(lambda x: np.asarray(x, float), (origin, axis, bend))
    angle = np.linalg.norm(bend)
    if not 0 < full_length or angle >= np.pi:
        raise ValueError("The circular chart requires positive length and turn below pi")
    curvature = bend / full_length
    k2 = float(curvature @ curvature)
    values, normals, labels = [], [], []

    def retain(roots, normal, labels_for_surface, valid):
        length = (
            2 * roots if angle == 0 else 2 * np.arctan((angle / full_length) * roots) / (angle / full_length)
        )
        keep = valid & np.isfinite(length) & (roots > 1e-10) & (length <= full_length)
        if np.any(keep):
            values.extend(length[keep].tolist())
            normals.extend(normal[keep])
            labels.extend(np.broadcast_to(labels_for_surface[:, None], roots.shape)[keep])

    margin = UPPER - FACES @ origin
    roots = quadratic_roots(2 * (FACES @ curvature) - margin * k2, 2 * (FACES @ axis), -margin)
    with np.errstate(invalid="ignore"):
        points, tangents = points_and_tangents(origin, axis, curvature, roots)
    normal = np.broadcast_to(FACES[:, None, :], points.shape)
    outward = np.einsum("ijk,ijk->ij", normal, tangents) > 1e-12
    retain(roots, normal, np.array([f"prism{i}" for i in range(len(FACES))]), outward)

    displacement = origin - CENTRES
    sphere_offset = np.sum(displacement * displacement, axis=1) - RADIUS**2
    roots = quadratic_roots(
        4 + 4 * (displacement @ curvature) + sphere_offset * k2, 4 * (displacement @ axis), sphere_offset
    )
    with np.errstate(invalid="ignore"):
        points, tangents = points_and_tangents(origin, axis, curvature, roots)
    radial = points - CENTRES[:, None, :]
    in_cap = np.einsum("ijk,ik->ij", radial, NORMALS) >= CUT - 1e-10
    entering_sphere = np.einsum("ijk,ijk->ij", radial, tangents) < -1e-12
    retain(
        roots, radial, np.array([f"dome{i}:sphere" for i in range(len(CENTRES))]), in_cap & entering_sphere
    )

    margin = CUT - np.sum(NORMALS * displacement, axis=1)
    roots = quadratic_roots(2 * (NORMALS @ curvature) - margin * k2, 2 * (NORMALS @ axis), -margin)
    with np.errstate(invalid="ignore"):
        points, tangents = points_and_tangents(origin, axis, curvature, roots)
    radial = points - CENTRES[:, None, :]
    in_sphere = np.sum(radial * radial, axis=2) <= RADIUS**2 + 1e-10
    entering_cap = np.einsum("ijk,ik->ij", tangents, NORMALS) > 1e-12
    normal = np.broadcast_to(NORMALS[:, None, :], points.shape)
    retain(roots, normal, np.array([f"dome{i}:plane" for i in range(len(CENTRES))]), in_sphere & entering_cap)
    if not values:
        return None
    first = int(np.argmin(values))
    return dict(length=float(values[first]), normal=np.asarray(normals[first]), label=str(labels[first]))
