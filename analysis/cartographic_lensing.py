"""Effective lensing fields of a projection, referred to the TAN chart.

What this establishes
----------------------
Section `subsec:qe` of the paper treats the composite map from the analysis
grid to the gnomonic (TAN) chart, C = P_TAN o P^-1, as a lens: its gradient
decomposes into an effective convergence, shear and curl exactly as gravitational
lensing does (Appendix `app:eff`, Eq. eq:eff_lensing),

    D_ij = dC_i/dx_j - delta_ij,
    kappa = -(D11+D22)/2,  gamma1 = -(D11-D22)/2,
    gamma2 = -(D12+D21)/2,  omega = (D12-D21)/2,

and the effective displacement d(x) = C(x) - x itself. This module puts
numbers on Table `table:curl`, using the same Jacobian machinery as the rest
of the paper (`geometry.relative_jacobian`) rather than the hand-written
finite differences of the retired `notes/curl.py`.

C_ij is exactly `geometry.relative_jacobian("ProjTAN", proj, alpha, delta)`:
that function returns J_to . J_from^-1, the operator carrying the *from*
plane to the *to* plane, which is precisely P_TAN o P^-1 differentiated. See
its docstring for the derivation.

|d| needs actual coordinates, not just their derivative, so it is built
straight from `geometry._xy`: project the same sky point through P and
through TAN and take the plane-coordinate difference, each referred to its
own field-centre pixel so an arbitrary grid-reference offset cancels rather
than dominating a real cartographic displacement.

Footprint and weighting are the SAME as `paper_numbers._field_1500d` and
`shear_leakage.report_1500d`: `footprints.rect_mask(50.0, -70.0, -42.0,
nside=1024)`, an apodized (C2, 2.5 deg) mask, weighted by mask^2 -- so this
table and everything else the paper says about the 1500d field are one
quadrature, not two independent ones that happen to agree.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import footprints as F
import geometry as G
import spectra as S

TAN = "ProjTAN"


def _centered_xy(proj, alpha_deg, delta_deg):
    """Plane coordinates in degrees, referred to the projection's own centre.

    `geometry._xy` returns raw fractional-pixel-turned-degree coordinates,
    which carry an arbitrary additive offset from the pixel grid's reference
    pixel. That offset is irrelevant to the Jacobian (it cancels in every
    finite difference `geometry.jacobian` takes) but not to a coordinate
    *difference* between two projections, so subtract each projection's value
    at the shared field centre before comparing them.
    """
    fsm = G._flat_map(proj)
    c = G._xy(fsm, G.ALPHA_CENTER_DEG, G.DELTA_CENTER_DEG)
    return G._xy(fsm, alpha_deg, delta_deg) - c


def effective_lensing(proj, alpha_deg, delta_deg):
    """kappa, gamma1, gamma2, omega, |d| (deg) for one projection, per pixel.

    `alpha_deg`, `delta_deg` are sky positions; everything is referred to the
    TAN chart via C = P_TAN o P^-1, Eq. eq:eff_lensing.
    """
    C = G.relative_jacobian(TAN, proj, alpha_deg, delta_deg)
    D = C - np.eye(2)
    D11, D12 = D[..., 0, 0], D[..., 0, 1]
    D21, D22 = D[..., 1, 0], D[..., 1, 1]
    kappa = -0.5 * (D11 + D22)
    gamma1 = -0.5 * (D11 - D22)
    gamma2 = -0.5 * (D12 + D21)
    omega = 0.5 * (D12 - D21)

    x_p = _centered_xy(proj, alpha_deg, delta_deg)
    x_t = _centered_xy(TAN, alpha_deg, delta_deg)
    d = np.hypot(x_t[0] - x_p[0], x_t[1] - x_p[1])

    return kappa, gamma1, gamma2, omega, d


def _rms(a, w):
    return float(np.sqrt(np.sum(w * a ** 2) / np.sum(w)))


def field_1500d_moments(nside=1024):
    """RMS of kappa, |gamma|, omega, |d| over SPT's 1500d field, per projection.

    Returns {label: dict(kappa_rms, gamma_rms, omega_rms, d_rms_deg)}, in the
    same mask-squared weighting as `paper_numbers._field_1500d`.
    """
    import healpy as hp

    mask = F.rect_mask(50.0, -70.0, -42.0, nside=nside)
    idx = np.flatnonzero(mask)
    w = mask[idx] ** 2
    alpha, delta = hp.pix2ang(nside, idx, lonlat=True)
    alpha = np.where(alpha > 180.0, alpha - 360.0, alpha)

    out = {}
    for proj in S.PROJS:
        kappa, g1, g2, omega, d = effective_lensing(proj, alpha, delta)
        gamma = np.hypot(g1, g2)
        out[S.PROJ_LABEL[proj]] = dict(
            kappa_rms=_rms(kappa, w),
            gamma_rms=_rms(gamma, w),
            omega_rms=_rms(omega, w),
            d_rms_deg=_rms(d, w),
        )
    return out


if __name__ == "__main__":
    mom = field_1500d_moments()
    print(f"{'proj':<8}{'kappa_rms':>12}{'gamma_rms':>12}{'omega_rms':>14}{'d_rms_deg':>12}")
    for label in ("SFL", "BICEP", "ZEA", "STG"):
        m = mom[label]
        print(f"{label:<8}{m['kappa_rms']:>12.4f}{m['gamma_rms']:>12.4f}"
              f"{m['omega_rms']:>14.3e}{m['d_rms_deg']:>12.3f}")
