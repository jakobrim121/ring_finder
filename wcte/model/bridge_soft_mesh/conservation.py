"""Conservative four-momentum feasibility for a connected water interaction.

The incoming momentum is evaluated AFTER its visible energy loss. A water
atom starts at rest. The unobserved recoil must carry the missing momentum
and at least the lightest nuclear rest mass with the required baryon number.
We minimize over charge states and omit unobserved leptons/excitation energy;
this deliberately relaxes the bound, never claims a particular reaction.

Natural H/O isotopes are alternatives, not input labels. Atomic target masses
include their electrons, while the residual lower bounds omit all electrons
and their binding correction. This is conservative at the few-MeV level.
The additional 0.02 MeV allowance covers negligible thermal/molecular scales;
it cannot permit the hundreds-of-MeV energy creation motivating this test.

Atomic masses: NIST isotope tables (H, He, N, O, F), retrieved 2026-11-11.
Conversion and electron mass: CODATA 2018, matching the material tables.
"""

import numpy as np
from material import material

U = 931.49410242
ELECTRON = 0.51099895
# A: (smallest atomic mass needed here, corresponding Z).
ISOBARS = {
    0: (0.0, 0),
    1: (1.00782503223, 1),
    2: (2.01410177812, 1),
    3: (3.0160293201, 2),
    15: (15.00010889888, 7),
    16: (15.99491461957, 8),
    17: (16.99913175650, 8),
    18: (17.99915961286, 8),
    19: (18.99840316273, 9),
}
TARGETS = (1, 2, 16, 17, 18)


def margins(x, particles):
    """Return each target's energy surplus in GeV and exact coordinate gradient."""
    incoming, outgoing = map(material, particles)
    ma, mb = incoming["mass"], outgoing["mass"]
    r, f = x[9:11]
    remaining = 1000 * r * (1 - f)
    ea = ma + float(incoming["energy_at"](remaining))
    eb = mb + float(outgoing["energy_at"](1000 * x[11]))
    pa = np.sqrt(max(ea * ea - ma * ma, 1e-30))
    pb = np.sqrt(max(eb * eb - mb * mb, 1e-30))
    a = x[3:6]
    b = x[6:9]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    a = a / na
    b = b / nb
    cosine = float(a @ b)
    momentum2 = max(pa * pa + pb * pb - 2 * pa * pb * cosine, 0.0)
    da = float(incoming["energy_at"].derivative()(remaining))
    db = float(outgoing["energy_at"].derivative()(1000 * x[11]))
    baryon_change = int(particles[0] == "proton") - int(particles[1] == "proton")
    values = []
    gradients = []
    for target in TARGETS:
        atomic, z = ISOBARS[target]
        target_mass = U * atomic
        atom_res, z_res = ISOBARS[target + baryon_change]
        residual_mass = max(U * atom_res - z_res * ELECTRON, 0.0)
        recoil = np.sqrt(residual_mass * residual_mass + momentum2)
        # A zero-mass, zero-momentum residual is a nonsmooth boundary. Use a
        # finite limiting derivative there; another target gives a smooth,
        # less restrictive alternative for the fits considered here.
        denominator = max(recoil, 1e-12)
        d_ea = 1 - (pa - pb * cosine) * ea / (denominator * pa)
        d_eb = -1 - (pb - pa * cosine) * eb / (denominator * pb)
        d_cos = pa * pb / denominator
        gradient = np.zeros_like(x)
        gradient[9] = d_ea * 1000 * (1 - f) * da
        gradient[10] = -d_ea * 1000 * r * da
        gradient[11] = d_eb * 1000 * db
        gradient[3:6] = d_cos * (b - cosine * a) / na
        gradient[6:9] = d_cos * (a - cosine * b) / nb
        values.append((ea + target_mass - eb - recoil + 0.02) / 1000)
        gradients.append(gradient / 1000)
    return np.asarray(values), np.asarray(gradients)
