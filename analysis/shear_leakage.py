"""E-to-B leakage is the projection's shear, with the coefficient 2.

What this establishes
---------------------
The production suite measures BB/EE on unlensed sims -- where the true B is
zero -- for four flat-sky projections on seven apodized RA/dec rectangles.
They span four decades. This script shows that a single number computed from projection
geometry alone predicts every one of them:

    BB / EE  =  2 <g^2>,    g = (s1 - s2) / (s1 + s2),

with (s1, s2) the singular values of the projection's local Jacobian and <.>
the mask-squared-weighted mean over the footprint. The 2 is not fitted. It
falls out of the derivation below; the coefficient fitted to the usable
(mask, projection) pairs and its scatter are the `numbers` macros `fitcoef`,
`fitrms` and `fitworst`, which the `numbers` output computes from these
functions; nothing is typed.

Why 2, exactly
--------------
Locally a projection is a linear map J from the sky tangent plane to the map
plane. Polarization is sampled from HEALPix in the sky's (east, north) basis
and then rotated into the grid basis ("flattening"). Write the polar
decomposition J = R(psi_R) S, R a rotation and S symmetric. The `flattening`
decision picks the angle: `polar` rotates by psi_R itself (paper Eq.
polar_angle), so the rotation is removed exactly; `spt3g` uses
`maps.flatten_pol`, whose angle misses psi_R by a residual Delta of
(g^2/2) sin 4beta + O(g^3), beta the angle of the principal axes to the grid.
`flattening_residual` returns Delta for either choice -- identically zero
for `polar` -- and a residual rotation adds sin^2(2 Delta) ~ g^4 sin^2 4beta
to the leakage, which `predicted_spectra` carries through the shear moments.
Up to that known residual only the symmetric part S survives.

S cannot be removed. A sheared spin-2 field is not a rotated spin-2 field.
Working in the principal frame of S, a map-plane Fourier mode k comes from the
sky mode q = J^T k, and the two are misaligned by

    dpsi(chi) = arctan((s2/s1) tan chi) - chi  ~  -g sin(2 chi),

with chi the angle of k to the major axis. The sky has E only, so the plane's
spin-2 mode acquires B = E sin(2 dpsi). NaMaster bins in |l| annuli, which
averages over chi uniformly, and <sin^2(2 dpsi)> = 4 g^2 <sin^2 2chi> = 2 g^2.
That is the whole law. The mask enters only as the weight: pseudo-C_l weights
power by the apodized mask squared, so g^2 is averaged with weight mask^2 --
and the next section shows that plain mask^2, not a plane-area version of it,
is what the data want.

Why the reduction has to be g = (s1-s2)/(s1+s2)
----------------------------------------------
Only the *ratio* s1/s2 can matter -- a pure magnification (s1 = s2) is
conformal and leaks nothing -- and the leading term in the misalignment is
(s1-s2)/(s1+s2). Reductions that agree with it to first order but not beyond
do measurably worse: <ln^2(s1/s2)>, which differs at third order, and
<(s1/s2 - 1)^2>, which already differs at second order (`report` prints
the three fits). The weight matters as much as the reduction.
Weighting g^2 by mask^2 |det J| or mask^2 |det J|^2 -- the plane-area
measures one might expect from the pseudo-C_l normalisation -- scatters more
(`report` prints all three). Plain mask^2 wins
because the extra powers of |det J| are cancelled by the response of a falling
spectrum to the same stretch; that cancellation is only near-exact, and what
survives it is the l dependence in point 3 below.

Where the law stops
-------------------
Three places, all real and all reported rather than smoothed over.

  1. STG is conformal: s1 = s2 up to the truncation error of the finite-
     difference Jacobian (g ~ 2e-8, falling as step^2), so the law predicts
     zero. Its measured BB/EE is the pipeline's own E-to-B floor, strongly
     l-dependent and largest in the top band (`report` prints it by band).
     STG is excluded from the fit and used to state that floor, which is then
     subtracted before any band-by-band claim, and no point within 20x of it
     is judged.

  2. The law holds only where the footprint the pipeline analysed is the
     footprint the geometric moment averages over. The flat grid is sized to
     hold the largest rectangle with its taper in every projection
     (`flatsky_masks` records each cut's `margin_pix`), and that rectangle
     stops at dec -79.1, 11 deg short of the south celestial pole, so every
     (footprint, projection) pair tests the law. The RA step of
     geometry.jacobian goes as 1 / cos(delta), so the differenced Jacobian
     fails inside the last fraction of a degree of the pole; every moment is
     also accumulated with a `POLAR_CUT_DEG` cap excluded, as a check.

  3. The band average hides an l dependence. Band by band, SFL's ratio to
     2<g^2> rises with l and BICEP's falls. The
     leading-order law is l-independent by construction and cannot describe
     that. `predicted_spectra` restores what it drops -- the exact
     misalignment kernel, the fact that a plane mode k reads the sky
     spectrum at |J^T k| rather than |k|, and the flattening's residual
     rotation Delta -- and reproduces both trends band by band, over all masks and
     sheared projections that sit clear of the floor (`numbers` quotes
     SFL's and BICEP's measured-over-model ratio band by band as
     `bandSFLratio` and `bandBICEPratio`). SFL is equal-area, so the
     azimuthal stretch is symmetric about 1 and the convexity of a falling
     spectrum lifts the high-shear pixels at high l. BICEP is not: its shear
     comes with magnification of up to a factor of four, so the same convexity
     pushes the other way. Same law, opposite second-order term.

     SFL and BICEP leak in proportion to orthogonal second moments of the
     footprint. Expand both about the projection centre. SFL has
     x = (alpha - alpha_0) cos delta, so its shear is g ~ (tan|delta_0| / 2)
     times the *east* offset. BICEP has x = (alpha - alpha_0) cos delta_0, so
     its shear is g ~ (tan|delta_0| / 2) times the *north* offset. Same
     coefficient, orthogonal directions: the two would agree only on a
     footprint with <east^2> = <north^2>. Every rectangle here is self-similar
     to the 1500d field, whose east extent (+-50 deg of RA at cos 56 deg) is
     about twice its north extent (+-14 deg), so at leading order SFL leaks
     about four times what BICEP does across the whole sweep.

Inputs
------
Shear moments: `spectra.Run.moments`, the `shear_moments` output, evaluated on
the pixels of the pipeline's own `healpix_masks` (the seven apodized
rectangles of `src/stages/_footprints.py`, the 1500d field among them).

Spectra: the bandpowers via `spectra.Run.load` -- QU leg, simulation windows
removed. `predicted_spectra` is accordingly band-limited at the run's own
lmax.

Theory: `theory_spectra`'s unlensed_EE, needed only for the l-dependent
refinement.

As a recipe this script is the `fig_shear_leakage` output: it prints the
report and draws `shear_leakage.pdf` (and `fig_shear_leakage.png`) under
`--out`.

Run:  bash env.sh python3 analysis/shear_leakage.py
          --bandpowers results/fiducial/bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra results/fiducial/theory_spectra
          --shear-moments results/fiducial/shear_moments --out <dir>

Cost
----
The moments are read, not computed. The l-dependent model takes about 30 s.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as PS
import spectra as S
import theory as T

# Bands to test the law in. The first is the headline band average; the rest
# resolve the l dependence the average hides. The band-by-band set runs to
# l = 4500 -- the run carries sky power to its lmax, and 4800 is where the
# 2 arcmin pixel takes over, so 4000-4500 is the last band that is still a
# measurement of the projection.
BANDS = [(1000, 3000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500)]

# The conformal projection has no shear; what it measures is the pipeline floor.
FLOOR_PROJ = "ProjSTG"

# geometry.jacobian steps in RA by step_deg / cos(delta) to keep the step a
# fixed angular distance, and that step diverges at the pole, so the
# differenced Jacobian is meaningless in the last fraction of a degree. Every
# moment is therefore also accumulated with a polar cap excluded, as a check.
# The southernmost footprint edge, dec -79.1, is 11 deg from the pole.
POLAR_CUT_DEG = 0.5

# Reductions of (s1, s2) to a single shear scalar. All agree at leading order.
REDUCTIONS = {
    "g2": (r"$\langle g^2 \rangle,\ g=(s_1-s_2)/(s_1+s_2)$",
           lambda s1, s2: ((s1 - s2) / (s1 + s2)) ** 2),
    "q2": (r"$\langle (s_1/s_2 - 1)^2 \rangle$",
           lambda s1, s2: (s1 / s2 - 1.0) ** 2),
    "lnq2": (r"$\langle \ln^2(s_1/s_2) \rangle$",
             lambda s1, s2: np.log(s1 / s2) ** 2),
}


# --------------------------------------------------------------------------
# Geometry: mask-weighted shear moments
# --------------------------------------------------------------------------

def flattening_residual(jac, flattening):
    """The flattening's rotation minus the polar rotation of J, in radians, per pixel.

    Zero for the `polar` option, which rotates by psi_R itself. For `spt3g`,
    `maps.flatten_pol` rotates the polarization by rot/2, with
    rot = atan2(-da/dx, da/dy) + atan2(-dd/dy, -dd/dx) built from the inverse
    map. The derivatives of (alpha, delta) by pixel coordinates are the
    inverse of `geometry.jacobian` up to positive factors (cos delta, the
    pixel size) and the x flip `geometry._xy` applies, none of which move an
    atan2, so rot follows from J alone. The residual against the polar angle
    phi of J = R(phi) S is (g^2/2) sin 4beta + O(g^3), signed, beta the angle
    of the principal axes of S from sky east toward north.
    """
    if flattening == "polar":
        return np.zeros(jac.shape[:-2])
    a, b = jac[..., 0, 0], jac[..., 0, 1]
    c, d = jac[..., 1, 0], jac[..., 1, 1]
    det = a * d - b * c
    rot = np.arctan2(d / det, -b / det) + np.arctan2(-a / det, -c / det)
    phi = np.arctan2(c - b, a + d)
    return rot / 2 + phi


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def measured_ratios(run, bands=BANDS):
    """Band-averaged BB/EE per (band, mask, proj). NaN where the run failed.

    Averages the numerator and the denominator separately -- that is what the
    quoted ratio means, and it weights low l more heavily because EE falls.
    Any NaN inside a band poisons that entry rather than being nan-meaned
    around. The run has no NaNs: every mask is finite for every projection.
    """
    ee, bb = run.load("EE"), run.load("BB")
    ell = run.ell
    out = np.full((len(bands), len(S.SWEEP), len(S.PROJS)), np.nan)
    for b, (lo, hi) in enumerate(bands):
        sel = (ell > lo) & (ell < hi)
        for i in range(len(S.SWEEP)):
            for j in range(len(S.PROJS)):
                e, x = ee[i, j][:, sel], bb[i, j][:, sel]
                if not (np.isfinite(e).all() and np.isfinite(x).all()):
                    continue
                if not np.isfinite(e.mean()) or e.mean() <= 0:
                    continue
                out[b, i, j] = x.mean() / e.mean()
    return out


# --------------------------------------------------------------------------
# The l-dependent refinement
# --------------------------------------------------------------------------

def _cells(mom, i, j):
    """Occupied (s1, s2, weight) cells of one (mask, projection)."""
    h = mom["hist"][i, j]
    occ = h > 0
    return mom["cell_s1"][i, j][occ], mom["cell_s2"][i, j][occ], h[occ]


def _hat_weights(x, grid):
    """Linear-interpolation weights of the points `x` on the sorted `grid`.

    Returns (lo, hi, t): the value of `np.interp(x, grid, c, left=c[1],
    right=0)` is (1 - t) c[lo] + t c[hi], with t = 0 where x lands on or below
    the grid's ends and both weights dropped (t = nan) where x lies above it.
    Below the grid the weight goes to index 1, as `left=c[1]` does: index 0
    is the l = 0 slot, which CAMB zeroes.
    """
    n = grid.size
    lo = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, n - 2)
    hi = lo + 1
    t = (x - grid[lo]) / (grid[hi] - grid[lo])
    below = x < grid[0]
    lo = np.where(below, 1, lo)
    t = np.where(below, 0.0, t)
    t = np.where(x > grid[-1], np.nan, t)
    return lo, hi, t


def coupling_matrix(mom, i, j, ell, ell_prime, *, band_limit, n_chi=128, chunk=16):
    """The forward model as a matrix: plane C_l = R @ sky C_l', per spin block.

    Eq. (forward) is linear in the sky spectrum, so for one (mask, projection)
    it is a matrix R(l, l') from a sky spectrum sampled on `ell_prime` to the
    expected mask-decoupled flat-sky spectrum at `ell`, normalised to absolute
    C_l. Row l is a weighted sum over footprint cells c and mode azimuths chi
    of a hat function on the `ell_prime` grid centred at the sheared
    wavenumber l r_c(chi), r = |J^T l| / l, so `R @ cl` reproduces linear
    interpolation of `cl` exactly; `ell_prime` needs to resolve the spectrum,
    which the integer-l theory grid does. Each output l draws on the band of
    sky l' between l s2_min and l s1_max. Columns above `band_limit` are zero
    -- the sims' own band limit, `spectra.Run.lmax`; pass None for a sky
    without one.

    The blocks, keyed "XY<-ZW":

      TT<-TT           kernel 1: the scalar remapping alone.
      EE<-EE, BB<-BB   kernel cos^2(2 dpsi + 2 Delta) -> after the azimuthal
                       average, cos^2(2 dpsi) (1 - f) + sin^2(2 dpsi) f.
      BB<-EE, EE<-BB   the complement, sin^2(2 dpsi) (1 - f) + cos^2(2 dpsi) f.
      TE<-TE           kernel cos(2 dpsi + 2 Delta) -> cos(2 dpsi) <cos 2 Delta>.
      TB<-TE           kernel sin(2 dpsi + 2 Delta) -> cos(2 dpsi) <sin 2 Delta>.

    dpsi(chi) is the Fourier-space misalignment, Delta the flattening's
    residual rotation (zero under `polar`), f the cell's mean sin^2(2 Delta)
    and <sin 2 Delta> its signed mean (`shear_moments.footprint_moments`). The cross terms sin(4 dpsi) sin(4 Delta) / 2,
    sin(2 dpsi) sin(2 Delta) and sin(2 dpsi) cos(2 Delta) are odd in chi about
    the principal axis while the stretch is even, so they average away, for
    any spectrum and to all orders in g. The spin blocks are symmetric under
    E <-> B because the sky's B is rotated by the same angle as its E. For TE
    the cell mean of cos(2 Delta) is taken as sqrt(1 - f), exact for a Delta
    uniform within the cell; Delta is O(g^2) and nearly constant across a
    cell, so the error is O(Delta^4) times the within-cell scatter.

    TB<-TE is first order in Delta, so O(g^2) per pixel, where the leakage
    feels Delta only at second order. Delta is odd under a mirror about the
    projection's central meridian, so over a footprint
    symmetric about that meridian the signed mean, and TB<-TE, vanish; every
    footprint in this suite is one.

    The pseudo-C_l weight per unit sky area is W^2 |det J|^2 -- one |det J|
    because C_plane(k) = |det J| C_sky(J^T k), one more because the W^2
    integral runs over plane area -- and the normalisation is the same
    integral without the spectrum, W^2 |det J|.
    """
    s1, s2, w = _cells(mom, i, j)
    occ = mom["hist"][i, j] > 0
    f = mom["cell_sin2_2delta"][i, j][occ][:, None]
    f_signed = mom["cell_sin_2delta"][i, j][occ][:, None]
    chi = (np.arange(n_chi) + 0.5) * (np.pi / n_chi)

    chi_sky = np.arctan2(s2[:, None] * np.sin(chi), s1[:, None] * np.cos(chi))
    dpsi = (chi_sky - chi + np.pi / 2) % np.pi - np.pi / 2
    leak = np.sin(2 * dpsi) ** 2
    kern_b = leak * (1 - f) + (1 - leak) * f
    kernels = {
        "TT<-TT": np.ones_like(leak),
        "EE<-EE": 1 - kern_b,
        "BB<-EE": kern_b,
        "TE<-TE": np.cos(2 * dpsi) * np.sqrt(1 - f),
        "TB<-TE": np.cos(2 * dpsi) * f_signed,
    }
    stretch = np.sqrt(s1[:, None] ** 2 * np.cos(chi) ** 2
                      + s2[:, None] ** 2 * np.sin(chi) ** 2)
    # Cell weight, azimuthal mean and normalisation folded into each kernel.
    scale = (w * (s1 * s2) ** 2 / (w * s1 * s2).sum())[:, None] / n_chi
    vals = {k: (v * scale)[:, None, :] for k, v in kernels.items()}

    grid = np.asarray(ell_prime, dtype=float)
    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    n_p = grid.size
    out = {k: np.zeros((ell.size, n_p)) for k in kernels}
    for k0 in range(0, ell.size, chunk):
        sl = slice(k0, k0 + chunk)
        nl = ell[sl].size
        lo, hi, t = _hat_weights(ell[sl][None, :, None] * stretch[:, None, :],
                                 grid)
        keep = np.isfinite(t)
        row = np.broadcast_to(np.arange(nl)[None, :, None], t.shape)[keep] * n_p
        t = t[keep]
        idx = np.concatenate([row + lo[keep], row + hi[keep]])
        for name, v in vals.items():
            v = np.broadcast_to(v, keep.shape)[keep]
            wts = np.concatenate([v * (1 - t), v * t])
            out[name][sl] = np.bincount(idx, weights=wts,
                                        minlength=nl * n_p).reshape(nl, n_p)
    if band_limit is not None:
        for m in out.values():
            m[:, grid > band_limit] = 0.0
    out["BB<-BB"] = out["EE<-EE"]
    out["EE<-BB"] = out["BB<-EE"]
    return out


def predicted_spectra(mom, i, j, ell, ell_theory, cl_theory, *, band_limit, n_chi=128):
    """Forward-model BB(l) and EE(l), absolute C_l, from a sky with E only.

    The E column of `coupling_matrix`: BB = R_{BB<-EE} C^EE, EE = R_{EE<-EE}
    C^EE. That restores the three things the leading-order law drops:

      * the exact misalignment kernel sin^2(2 dpsi) rather than 4 dpsi^2,
      * the fact that a plane mode k reads the sky spectrum at |J^T k|, which
        depends on the angle of k to the shear's major axis, and
      * the pipeline's own flattening angle, through its residual rotation
        Delta (`flattening_residual`), which the shear moments carry and
        which is zero under `flattening: polar`.

    The second is what makes the ratio depend on l at all.

    `band_limit` kills sky power above the simulations' own band limit,
    `spectra.Run.lmax`. Without it the topmost bands are over-predicted, and
    they are exactly where the stretch |J^T k| / |k| pushes hardest against
    the cut.
    """
    R = coupling_matrix(mom, i, j, ell, ell_theory, band_limit=band_limit,
                        n_chi=n_chi)
    cl = np.asarray(cl_theory, dtype=float)
    return R["BB<-EE"] @ cl, R["EE<-EE"] @ cl


def diagonal_transfer_error(R, ell, ell_prime, cl_cal, cl_test, per_band=1):
    """What a diagonal transfer function gets wrong about a spectrum it was not
    calibrated on.

    A pipeline that calibrates on simulations of `cl_cal` sees the forward
    model only through its bandpower ratio F_b = <R C_cal>_b / <C_cal>_b and
    undoes it by dividing the observed bandpower by F_b. Applied to a sky
    `cl_test`, whose observed bandpower is <R C_test>_b, that returns
    <R C_test>_b / F_b; the fractional error of that estimate of <C_test>_b is
    returned, one entry per band. A band is `per_band` consecutive rows of
    `R`, which is one block of `coupling_matrix(..., ell, ell_prime)`; spectra
    live on `ell_prime`.

    F_b inherits the acoustic peaks: smearing over the band of sky l' fills
    troughs and clips peaks, so the error oscillates with the input spectrum's
    phase about a smooth offset, and a single l samples an arbitrary phase.

    Two identities pin the scalar block for smooth inputs. A constant
    (white) spectrum maps to int W^2 |det J|^2 / int W^2 |det J|, the footprint
    mean of |det J| under the estimator's weight, below the band limit. And
    C ~ l^-2 (D_l constant) passes through unchanged for any J: the azimuthal
    mean of r^-2 is 1 / (s1 s2), because r^-2 dchi / (s1 s2) is the uniform
    measure in the sky azimuth, so one power of |det J| cancels. The spin
    blocks keep the second identity up to their kernel's footprint mean,
    which is the same over sky azimuth as over plane azimuth. For an
    equal-area projection white and l^-2 inputs therefore give the same error.
    """
    def band(x):
        return x.reshape(-1, per_band).mean(-1)
    cal = band(np.interp(ell, ell_prime, cl_cal))
    test = band(np.interp(ell, ell_prime, cl_test))
    transfer = band(R @ cl_cal) / cal
    return band(R @ cl_test) / (transfer * test) - 1


def predicted_ratios(run, mom, bands=BANDS, n_chi=128, verbose=True):
    """Band-averaged BB/EE from `predicted_spectra`, same bands as the data,
    band-limited at the run's own lmax."""
    th = T.load(run.theory_spectra)
    ell_theory = th["ell"].astype(float)
    cl_theory = th["unlensed_EE"]
    ell = run.ell
    lo, hi = min(b[0] for b in bands), max(b[1] for b in bands)
    grid = ell[(ell > lo) & (ell < hi)]
    out = np.full((len(bands), len(S.SWEEP), len(S.PROJS)), np.nan)
    for i in range(len(S.SWEEP)):
        for j in range(len(S.PROJS)):
            bb, ee = predicted_spectra(mom, i, j, grid, ell_theory, cl_theory,
                                       band_limit=run.lmax, n_chi=n_chi)
            for b, (blo, bhi) in enumerate(bands):
                sel = (grid > blo) & (grid < bhi)
                out[b, i, j] = bb[sel].mean() / ee[sel].mean()
        if verbose:
            print(f"  forward model: mask {S.FSKY_PERCENT[i]:.2f}% done", flush=True)
    return out


# --------------------------------------------------------------------------
# Fitting and reporting
# --------------------------------------------------------------------------

def fit_constant(x, y):
    """Proportionality constant y = k x, fitted in log space, plus scatter.

    Log space, not least squares: the points span four decades and an ordinary
    least-squares fit through the origin is set entirely by the two largest.
    The quantity that matters is fractional agreement, so give every point
    equal weight in log y.
    """
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[good], y[good]
    k = np.exp(np.mean(np.log(y / x)))
    resid = y / (k * x) - 1.0
    return k, float(np.std(resid)), float(np.max(np.abs(resid))), int(good.sum())


def _fmt_table(title, values, fmt="{: .4e}"):
    lines = [title, "proj    " + "".join(f"{p:>11}" for p in
                                         [f"{v:.2f}%" for v in S.FSKY_PERCENT])]
    for j, proj in enumerate(S.PROJS):
        row = f"{S.PROJ_LABEL[proj]:<8}"
        for i in range(len(S.SWEEP)):
            v = values[i, j]
            row += "        nan" if not np.isfinite(v) else fmt.format(v)
        lines.append(row)
    return "\n".join(lines)


def law_mask(meas):
    """(mask, proj) entries that test the law: finite, and not the conformal floor projection."""
    keep = np.isfinite(meas)
    keep[:, S.PROJS.index(FLOOR_PROJ)] = False
    return keep


def report(mom, meas, pred=None):
    """Everything the paper needs, printed."""
    jstg = S.PROJS.index(FLOOR_PROJ)
    g2 = mom["g2"]                                        # (mask, proj)
    band0 = meas[0]
    floor = meas[:, :, jstg]                              # (band, mask)

    print()
    print("=" * 78)
    print("SHEAR MOMENTS AND MEASURED LEAKAGE")
    print("=" * 78)
    print(_fmt_table("<g^2>, g = (s1-s2)/(s1+s2), mask^2 weighted", g2))
    print()
    print(_fmt_table(f"measured BB/EE, {BANDS[0][0]} < l < {BANDS[0][1]}", band0))
    print()
    print(_fmt_table("measured BB/EE  /  <g^2>   (STG omitted: its <g^2> is a "
                     "numerical zero)",
                     np.where(np.arange(len(S.PROJS))[None, :] == jstg,
                              np.nan, band0 / g2), fmt="{: 11.4f}"))

    print()
    print("=" * 78)
    print("THE LAW")
    print("=" * 78)
    keep = law_mask(band0)
    x, y = g2[keep], band0[keep]
    k, rms, worst, n = fit_constant(x, y)
    r2 = y / (2 * x) - 1.0
    print("predicted coefficient   2  (exact, from <sin^2(2 dpsi)> = 2 g^2)")
    print(f"fitted coefficient      {k:.4f}   ({n} points, STG excluded)")
    print(f"fit / prediction        {k / 2:.4f}")
    print(f"scatter about y = 2x    {np.std(r2) * 100:.2f}% rms, "
          f"{np.max(np.abs(r2)) * 100:.2f}% worst")
    print(f"scatter about the fit   {rms * 100:.2f}% rms, {worst * 100:.2f}% worst")
    print(f"dynamic range           {np.log10(x.max() / x.min()):.2f} decades in <g^2>, "
          f"{np.log10(y.max() / y.min()):.2f} in BB/EE")
    slope, inter = np.polyfit(np.log(x), np.log(y), 1)
    print(f"free power law          BB/EE = {np.exp(inter):.4f} <g^2>^{slope:.4f}")
    print("     (the exponent is the test of linearity; 1 means the leakage is")
    print("      linear in <g^2>, not merely monotonic in it)")

    print()
    print("STG: the floor, not the law")
    print(f"  geometric <g^2>       {g2[:, jstg].min():.2e} to {g2[:, jstg].max():.2e}")
    print("     (that is the finite-difference floor of geometry.jacobian at")
    print("      step_deg = 0.02, which scales as step^2; STG is conformal, so")
    print("      its true shear is exactly zero)")
    print(f"  law would predict     {2 * g2[:, jstg].min():.2e} to {2 * g2[:, jstg].max():.2e}")
    print(f"  measured BB/EE        {band0[:, jstg].min():.2e} to {band0[:, jstg].max():.2e}")
    print(f"  -> the pipeline's own E-to-B floor is {np.median(band0[:, jstg]):.1e} "
          f"over 1000 < l < 3000")
    print("  The floor is steeply l-dependent (it is what limits the top bands):")
    print(f"     {'band':<14}" + "".join(f"{v:.2f}%".rjust(11) for v in S.FSKY_PERCENT))
    for b, (lo, hi) in enumerate(BANDS[1:], start=1):
        print(f"     {lo}-{hi:<9}" + "".join(f" {floor[b, i]:.3e}" for i in range(len(S.SWEEP))))

    print()
    print(f"Excluding a {POLAR_CUT_DEG} deg polar cap, where the differenced Jacobian fails:")
    print(f"  {'proj':<8}" + "".join(f"{v:.2f}%".rjust(11) for v in S.FSKY_PERCENT))
    for j, proj in enumerate(S.PROJS):
        print(f"  {S.PROJ_LABEL[proj]:<8}" + "".join(
            f"{mom['g2_nopole'][i, j] / mom['g2'][i, j] - 1:10.4f} "
            for i in range(len(S.SWEEP))))
    print("  (fractional change in <g^2>)")

    print()
    print("Extremes of the local Jacobian, s1_max / s2_min:")
    print(f"  {'proj':<8}" + "".join(f"{v:.2f}%".rjust(14) for v in S.FSKY_PERCENT))
    for j, proj in enumerate(S.PROJS):
        print(f"  {S.PROJ_LABEL[proj]:<8}" + "".join(
            f" {mom['s1_max'][i, j]:6.3f}/{mom['s2_min'][i, j]:<6.3f}"
            for i in range(len(S.SWEEP))))

    print()
    nan = [(i, j) for i in range(len(S.SWEEP)) for j in range(len(S.PROJS))
           if not np.isfinite(band0[i, j])]
    print(f"  Non-finite bandpowers anywhere in the suite: {len(nan)}")
    print(f"  Pairs entering the fit: {int(law_mask(band0).sum())} of "
          f"{(len(S.PROJS) - 1) * len(S.SWEEP)} sheared (mask, projection)")

    print()
    print("=" * 78)
    print("WHICH REDUCTION OF (s1, s2) COLLAPSES THE DATA BEST")
    print("=" * 78)
    print(f"{'reduction':<34}{'k':>9}{'rms':>9}{'worst':>9}")
    names = {"g2": "<g^2>, g=(s1-s2)/(s1+s2)",
             "q2": "<(s1/s2 - 1)^2>",
             "lnq2": "<ln^2(s1/s2)>"}
    for key in REDUCTIONS:
        k_, rms_, worst_, _ = fit_constant(mom[key][keep], y)
        print(f"{names[key]:<34}{k_:9.4f}{rms_ * 100:8.2f}%{worst_ * 100:8.2f}%")
    print()
    print("Weighting of the same reduction (mask^2 vs mask^2 |det J|^n):")
    for key, name in [("g2", "mask^2"), ("g2_det", "mask^2 |det J|"),
                      ("g2_det2", "mask^2 |det J|^2")]:
        k_, rms_, worst_, _ = fit_constant(mom[key][keep], y)
        print(f"{name:<34}{k_:9.4f}{rms_ * 100:8.2f}%{worst_ * 100:8.2f}%")

    print()
    print("=" * 78)
    print("BAND BY BAND")
    print("=" * 78)
    print("(measured BB/EE - STG floor) / 2<g^2>; 1.00 means the l-independent")
    print("law holds in that band. Raw values in parentheses where they differ.")
    hdr = f"{'band':<14}" + "".join(f"{S.PROJ_LABEL[p]:>18}" for p in S.PROJS
                                    if p != FLOOR_PROJ)
    # The smallest footprint, the 1500d field and the largest.
    field = S.SWEEP.index(S.FIELD)
    for i in [0, field, len(S.SWEEP) - 1]:
        print(f"--- {S.FSKY_PERCENT[i]:.2f}% mask ---")
        print(hdr)
        for b, (lo, hi) in enumerate(BANDS):
            row = f"{lo}-{hi:<9}"
            for j, proj in enumerate(S.PROJS):
                if proj == FLOOR_PROJ:
                    continue
                raw = meas[b, i, j] / (2 * g2[i, j])
                sub = (meas[b, i, j] - floor[b, i]) / (2 * g2[i, j])
                row += "               nan" if not np.isfinite(raw) else \
                    f"{sub:11.3f} ({raw:.2f})"
            print(row)

    if pred is not None:
        print()
        print("Restoring the l dependence (predicted_spectra), measured/model:")
        for i in [field, len(S.SWEEP) - 1]:
            print(f"--- {S.FSKY_PERCENT[i]:.2f}% mask ---")
            print(f"{'band':<14}" + "".join(
                f"{S.PROJ_LABEL[p]:>18}" for p in S.PROJS if p != FLOOR_PROJ))
            for b, (lo, hi) in enumerate(BANDS):
                row = f"{lo}-{hi:<9}"
                for j, proj in enumerate(S.PROJS):
                    if proj == FLOOR_PROJ:
                        continue
                    row += f"  {meas[b, i, j]:.5f}/{pred[b, i, j]:.5f}"
                print(row)
        # Only judge the model where the measurement is not floor-limited.
        clean = (law_mask(band0)[None, :, :]
                 & np.isfinite(meas) & np.isfinite(pred)
                 & (meas > 20 * floor[:, :, None]))
        r = meas[clean] / pred[clean]
        bad = np.argsort(-np.abs(r - 1))[:5]
        idx = np.array(np.nonzero(clean)).T
        print("\n  worst five points (band, mask, proj, measured/model):")
        for b_ in bad:
            bi, mi, pi = idx[b_]
            print(f"    {BANDS[bi]}  {S.FSKY_PERCENT[mi]:.2f}%  "
                  f"{S.PROJ_LABEL[S.PROJS[pi]]:<6} {r[b_]:.4f}")
        print(f"\n  all bands, all masks, three sheared projections, restricted to "
              f"points\n  more than 20x above the STG floor: measured/model = "
              f"{np.mean(r):.4f} +- {np.std(r):.4f}\n  ({clean.sum()} points, "
              f"worst {np.max(np.abs(r - 1)) * 100:.1f}%)")
        r0 = band0[keep] / pred[0][keep]
        print(f"\n  and on the 1000-3000 band average alone, the same points that "
              f"scatter\n  {np.std(r2) * 100:.2f}% about 2<g^2> scatter "
              f"{np.std(r0) * 100:.2f}% about the model: the residual\n  structure "
              "in the leading-order law is the l dependence, not noise.")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def figure(mom, meas, out):
    """The leakage relation over every usable sheared pair (left) and its
    residual by footprint size (right), at text width on the paper's rc.

    Writes `shear_leakage.pdf`, which the paper includes, and
    `fig_shear_leakage.png` under `out`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(PS.rc())
    colors = PS.PROJ_COLORS
    marks = PS.PROJ_MARKERS
    ms = 1.4 * PS.MS
    sheared = [p for p in S.PROJS if p != FLOOR_PROJ]

    g2 = mom["g2"]
    band0 = meas[0]
    keep = law_mask(band0)
    k, rms, _, _ = fit_constant(g2[keep], band0[keep])
    # The fit's rms band, expressed as a residual about the derived 2.
    fit_lo, fit_hi = 100 * (k / 2 * (1 - rms) - 1), 100 * (k / 2 * (1 + rms) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(PS.TEXT_W, 2.45),
                             layout="constrained")

    # (a) the law, without the STG floor that would add irrelevant decades.
    ax = axes[0]
    x = g2[keep]
    span = np.array([x.min() / 2.5, x.max() * 2.5])
    ax.plot(span, 2 * span, "-", color=PS.SECONDARY[0], lw=PS.LW, zorder=1)
    for j, proj in enumerate(S.PROJS):
        if proj == FLOOR_PROJ:
            continue
        sel = keep[:, j]
        # Hollow, not surface-filled, so coinciding markers stay visible.
        ax.plot(g2[sel, j], band0[sel, j], ls="none", marker=marks[proj],
                color=colors[proj], mfc="none", ms=ms, mew=PS.MEW, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*span)
    ax.set_ylim(2 * span[0], 2 * span[1])
    ax.set_xlabel(r"$\langle g^2\rangle$")
    ax.set_ylabel(r"$C_\ell^{BB}/C_\ell^{EE}$")

    # (b) residual about the derived coefficient, by footprint size.
    ax = axes[1]
    residuals = []
    for j, proj in enumerate(S.PROJS):
        if proj == FLOOR_PROJ:
            continue
        sel = keep[:, j]
        residual = 100 * (band0[sel, j] / (2 * g2[sel, j]) - 1)
        residuals.append(residual)
        ax.plot(S.FSKY_PERCENT[sel], residual, marker=marks[proj], color=colors[proj],
                mfc="white", ms=ms, mew=PS.MEW, lw=PS.LW, zorder=3)
    ax.axhline(0, color=PS.SECONDARY[0], lw=PS.LW, zorder=2)
    ax.axhspan(fit_lo, fit_hi, color=PS.SECONDARY[2], alpha=0.25, lw=0,
               zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"footprint $f_{\rm sky}$ [\%]")
    ax.set_ylabel(r"$C_\ell^{BB}/C_\ell^{EE} \,/\, 2\langle g^2\rangle - 1$ [\%]")
    residuals = np.concatenate(residuals)
    padding = 0.12 * (np.max(residuals) - np.min(residuals))
    ax.set_ylim(min(np.min(residuals), fit_lo) - padding,
                max(np.max(residuals), fit_hi) + padding)

    # One legend for both panels.
    handles = [Line2D([], [], ls="none", marker=marks[p], color=colors[p],
                      mfc="white", ms=ms, mew=PS.MEW) for p in sheared]
    labels = [S.PROJ_LABEL[p] for p in sheared]
    handles += [Line2D([], [], color=PS.SECONDARY[0], lw=PS.LW),
                Patch(facecolor=PS.SECONDARY[2], alpha=0.25, lw=0)]
    labels += [r"$C_\ell^{BB}/C_\ell^{EE} = 2\langle g^2\rangle$",
               rf"fitted $k = {k:.3f}$, $\pm${100 * rms:.1f}\% rms"]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "shear_leakage.pdf")
    fig.savefig(out / "fig_shear_leakage.png", dpi=200)
    print(f"\nwrote {out / 'shear_leakage.pdf'}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "bandpowers", "simulation_windows", "theory_spectra", "shear_moments")
    ap.add_argument("--out", type=Path, required=True)
    run = S.Run.from_args(args := ap.parse_args())

    mom = run.moments()
    meas = measured_ratios(run)
    pred = predicted_ratios(run, mom)
    report(mom, meas, pred)
    figure(mom, meas, args.out)


if __name__ == "__main__":
    main()
