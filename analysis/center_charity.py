"""Is any projection given an *uncharitable* centre?

The paper's five projections are all built with `delta_center = -56 deg`
(astra.yaml, grid/overnight.sh) and `alpha_center = 0` (the FlatSkyMap
default; see _project.ALPHA_CENTER_DEG_DEFAULT). The analysis masks are
concentric disks about the *same* point (0, -56).

The question this script answers: for each projection, does moving the
projection centre change the mask-weighted distortion at all -- and if so,
is -56 the charitable choice, or is some other declination better?

Two numbers per (projection, centre), both mask^2-weighted over the disk
(the pseudo-C_l weighting the paper uses in analysis/shear_leakage.py):

    <g^2>   with g = (s1 - s2) / (s1 + s2)      -- shear (E->B leakage)
    <(mu-1)^2>  with mu = s1 s2 = |det J|       -- magnification

Run:  bash env.sh python analysis/center_charity.py
"""

import numpy as np
import healpy as hp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import jacobian, singular_values

PROJS = ["ProjSFL", "ProjSIN", "ProjSTG", "ProjZEA", "ProjBICEP"]
RADII_RAD = [0.175, 0.2, 0.225, 0.25, 0.35, 0.45, 0.65]  # astra.yaml:172
APOSIZE_DEG = 2.5
FIELD = (0.0, -56.0)
NSIDE = 512


def disk_weights(radius_rad, nside=NSIDE, center=FIELD):
    """(alpha_deg, delta_deg, mask^2) for one C2-apodized disk about `center`."""
    cvec = hp.ang2vec(center[0], center[1], lonlat=True)
    idx = hp.query_disc(nside, cvec, radius_rad)
    vecs = np.array(hp.pix2vec(nside, idx))
    cosr = np.clip(cvec @ vecs, -1, 1)
    r = np.arccos(cosr)
    theta = radius_rad - r                      # distance to the boundary
    apo = np.radians(APOSIZE_DEG)
    x = np.clip(theta / apo, 0, 1)
    w = np.where(x >= 1, 1.0, x - np.sin(2 * np.pi * x) / (2 * np.pi))  # C2
    alpha, delta = hp.pix2ang(nside, idx, lonlat=True)
    return alpha, delta, w ** 2


def moments(proj, alpha, delta, w2, delta_center_deg, alpha_center_deg=0.0):
    J = jacobian(proj, alpha, delta,
                 alpha_center_deg=alpha_center_deg,
                 delta_center_deg=delta_center_deg)
    s1, s2 = singular_values(J)
    g = (s1 - s2) / (s1 + s2)
    mu = s1 * s2
    W = w2.sum()
    return (w2 * g ** 2).sum() / W, (w2 * (mu - 1) ** 2).sum() / W


def main():
    # Largest disk (0.65 rad ~ 37 deg) plus the smallest, for contrast.
    for radius in [0.65, 0.175]:
        alpha, delta, w2 = disk_weights(radius)
        print(f"\n=== disk radius {radius} rad ({np.degrees(radius):.1f} deg), "
              f"{alpha.size} pixels, centre {FIELD} ===")
        centers = [-90, -80, -75, -70, -66, -61, -56, -51, -46, -36, -20, 0]
        print(f"{'dec_center':>11} " + "".join(f"{p:>13}" for p in PROJS))
        for label, key in [("<g^2>", 0), ("<(mu-1)^2>", 1)]:
            print(f"  -- {label} --")
            for dc in centers:
                row = [moments(p, alpha, delta, w2, dc)[key] for p in PROJS]
                star = " *" if dc == -56 else ""
                print(f"{dc:>11}" + "".join(f"{v:13.4e}" for v in row) + star)

        # alpha_center scan at fixed dec, to confirm RA-independence
        print("  -- <g^2> vs alpha_center (dec_center=-56) --")
        for ac in [0.0, 30.0, 90.0]:
            row = [moments(p, alpha, delta, w2, -56.0, ac)[0] for p in PROJS]
            print(f"{ac:>11}" + "".join(f"{v:13.4e}" for v in row))


def pole_and_joint():
    """(a) how much of <g^2> comes from near the south pole, and (b) what a
    *joint* recentering (mask disk and projection moved together) does.

    (b) is the test that matters for the charity question: it is the only
    recentering an analyst could actually perform, since --ra-deg/--dec-deg
    sets the mask centre and the projection centre with one number.
    """
    print("\n(a) sub-pole contribution, R=0.65 disk and projections at (0,-56)")
    alpha, delta, w2 = disk_weights(0.65)
    print(f"    disk spans dec {delta.min():.2f} to {delta.max():.2f}")
    for cut in (-85.0, -80.0):
        sel = delta < cut
        print(f"  dec < {cut:g}: {100 * w2[sel].sum() / w2.sum():.3f}% of mask^2 weight")
        for p in PROJS:
            s1, s2 = singular_values(
                jacobian(p, alpha, delta, alpha_center_deg=0.0, delta_center_deg=-56.0))
            gg = ((s1 - s2) / (s1 + s2)) ** 2
            tot = (w2 * gg).sum()
            print(f"    {p:<10} <g2>={tot / w2.sum():.4e}  "
                  f"frac from dec<{cut:g}: {100 * (w2 * gg)[sel].sum() / tot:6.2f}%")

    for radius in (0.65, 0.45):
        print(f"\n(b) joint recentering, R={radius} rad")
        print(f"{'dec_center':>11}{'edge':>9} " + "".join(f"{p:>13}" for p in PROJS))
        for dc in (-56.0, -53.0, -50.0, -47.0, -43.0):
            a, d, w = disk_weights(radius, center=(0.0, dc))
            row = [moments(p, a, d, w, dc)[0] for p in PROJS]
            print(f"{dc:>11.0f}{dc - np.degrees(radius):>9.1f} "
                  + "".join(f"{v:13.4e}" for v in row))


if __name__ == "__main__":
    main()
    pole_and_joint()
