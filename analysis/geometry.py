"""Local projection geometry: Jacobians, shear, and a zero-parameter forward model.

Why this exists
---------------
The measured high-l excess in SFL-projected EE is caused by the *local shear*
of the projection, not by sky area and not by mask corners. Area tracks shear
across the footprint sweep only because its rectangles share one shape and
one centre, which makes size and shear degenerate; a footprint of another
shape breaks that degeneracy and cannot be read off the sweep's area trend.
This module computes the thing that actually matters, for any mask, in
minutes.

The physics, in one line
------------------------
A projection is locally a linear map J from the sky tangent plane to the map
plane. A field f on the sky becomes g(y) = f(J^-1 y) on the plane, so

    P_plane(k) = |det J| * P_sky(J^T k).

Everything below is that identity, evaluated pixel by pixel and averaged over
the mask. There are no free parameters and no reference to NaMaster, coupling
matrices, bandpower binning or purification -- which is what makes agreement
with the measured bandpowers meaningful rather than circular.

Only the singular values matter
-------------------------------
Write J = U S V^T with S = diag(s1, s2), s1 >= s2. Then |J^T k| =
|S U^T k|, and since NaMaster bins in |l| annuli we average over the direction
of k uniformly -- which averages uniformly over the direction of U^T k too.
So both U and V drop out and

    P_plane(l) = s1*s2 * < P_sky( l * sqrt(s1^2 cos^2 psi + s2^2 sin^2 psi) ) >_psi

depends on the projection only through (s1, s2). Two derived numbers carry the
whole story:

    magnification  mu = s1*s2   (= |det J|; exactly 1 for an equal-area projection)
    shear ratio     q = s1/s2   (= 1 for a conformal projection)

SFL is equal-area, so mu = 1 and it is pure shear. STG is conformal, so q = 1
and it is pure magnification -- and it moves the ratio the *other* way, because
magnifying a falling spectrum removes power. That sign flip is the sharpest
test the model passes.

Note on polarization
--------------------
The pipeline flattens Q/U by the polar-decomposition angle psi_R (the
`flattening` decision), which undoes the projection-induced *rotation* of
the polarization basis. It does not and cannot undo the shear:
a sheared spin-2 field is not a rotated spin-2 field. So E stays mixed into B
at order (q - 1), which is why the BB panel scales as (q - 1)^2.
"""

import sys
from pathlib import Path

import numpy as np

from spt3g import core, maps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "stages"))
from _footprints import DEC_DEG, RA_DEG  # noqa: E402

DEG = core.G3Units.deg
ARCMIN = core.G3Units.arcmin


def _flat_map(proj, alpha_center_deg=RA_DEG,
              delta_center_deg=DEC_DEG, res_arcmin=2.0, n=64):
    """A FlatSkyMap about the patch centre, used only as a coordinate transform.

    The pixel grid is irrelevant here -- angle_to_xy extrapolates the analytic
    projection formulae well outside the array bounds -- so `n` is kept small.
    """
    if isinstance(proj, str):
        proj = getattr(maps.MapProjection, proj)
    return maps.FlatSkyMap(
        n, n, res_arcmin * ARCMIN, proj=proj,
        alpha_center=alpha_center_deg * DEG, delta_center=delta_center_deg * DEG,
    )


def _xy(fsm, alpha_deg, delta_deg):
    """Projected coordinates in *degrees* (not pixels), shape (2, N).

    angle_to_xy returns fractional pixel indices; multiplying by the pixel size
    turns them into an angular map coordinate, which is what makes the Jacobian
    dimensionless. The x sign is flipped so that x increases eastward and
    det J > 0 -- projections are orientation-reversing in pixel indices (RA
    increases to the left) and a negative determinant is a distraction, not
    physics.
    """
    a = core.DoubleVector(np.atleast_1d(alpha_deg) * DEG)
    d = core.DoubleVector(np.atleast_1d(delta_deg) * DEG)
    x, y = fsm.angle_to_xy(a, d)
    res_deg = fsm.res / DEG
    return np.stack([-np.asarray(x), np.asarray(y)]) * res_deg


def jacobian(proj, alpha_deg, delta_deg, step_deg=0.02, **map_kw):
    """Local Jacobian d(map coords) / d(orthonormal sky coords), shape (N, 2, 2).

    The sky basis is the orthonormal (east, north) frame at each point, i.e.
    displacements (cos(delta) d_alpha, d_delta). Using d_alpha directly instead
    would fold in a factor of cos(delta) that has nothing to do with the
    projection and would make every projection look sheared near the pole.

    Central differences. `step_deg` trades truncation error against roundoff;
    0.02 deg sits in a broad plateau where the answer is stable to ~1e-6, and
    the projections are smooth on that scale everywhere we evaluate them.
    """
    fsm = _flat_map(proj, **map_kw)
    alpha = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    delta = np.atleast_1d(np.asarray(delta_deg, dtype=float))

    # East step must shrink as cos(delta) to stay a fixed angular distance.
    d_alpha = step_deg / np.cos(np.deg2rad(delta))
    east = (_xy(fsm, alpha + d_alpha, delta) - _xy(fsm, alpha - d_alpha, delta))
    east /= 2 * step_deg
    north = (_xy(fsm, alpha, delta + step_deg) - _xy(fsm, alpha, delta - step_deg))
    north /= 2 * step_deg

    return np.stack([east, north], axis=-1).transpose(1, 0, 2)


def singular_values(jac):
    """(s1, s2) with s1 >= s2, for a stack of 2x2 Jacobians.

    Closed form rather than np.linalg.svd: it is ~50x faster over millions of
    pixels and avoids the sign conventions of the singular vectors, which we
    never use.
    """
    jac = np.asarray(jac, dtype=float)
    a, b = jac[..., 0, 0], jac[..., 0, 1]
    c, d = jac[..., 1, 0], jac[..., 1, 1]
    # Singular values of a 2x2: |det| and the Frobenius norm determine them.
    det = np.abs(a * d - b * c)
    fro2 = a * a + b * b + c * c + d * d
    disc = np.sqrt(np.maximum(fro2 * fro2 - 4 * det * det, 0.0))
    s1 = np.sqrt(np.maximum(0.5 * (fro2 + disc), 0.0))
    s2 = np.sqrt(np.maximum(0.5 * (fro2 - disc), 0.0))
    return s1, s2





