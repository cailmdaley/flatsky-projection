"""Tom Crawford's 2026 proj0/proj5 measurement, and our prediction of it.

What this is
------------
In July 2026 Tom measured C_l(proj0)/C_l(proj5) on the 1500d field and found it
rising to about 2.5 by l ~ 4500 -- the observation that restarted this project.
His bandpowers sit on disk as pickles. This module loads them and hands the
geometry forward model a closed test: predict his proj0 from his proj5.

Why his proj5 is a legitimate input spectrum
--------------------------------------------
The forward model says ZEA's own bias over this footprint is well under a
percent -- ZEA is azimuthal about the field centre and barely shears. So his
measured proj5 bandpowers *are* the sky spectrum as seen through his beam, his
pixel window, his mask and his binning, to that accuracy. Feeding them back in
means every instrumental factor we do not know is already carried, and it
cancels again when we take the predicted ratio. What is left to predict is the
shear, which is the only thing this model claims to know.

The honest caveat
-----------------
Predicting the ratio at multipole l requires the input spectrum out to l*s1,
and s1 reaches 1.5 over this footprint. His bandpowers stop at l = 6000, so
above l = 4000 the prediction leans on a power-law extrapolation of his own
high-l slope. That extrapolation is stated explicitly rather than hidden, and
it is why the prediction and the measurement should be compared over
3000 < l < 5000 rather than at the very top of his range.

Files, all read-only, under /sptlocal/user/tcrawfor/:
    proj0_distortion_{tt,ee}[_cutcorner|_iso_shrink].pkl
each a dict of cls_proj0, cls_proj5, shape (4 sims, 60 bandpowers).
The plots he circulated are proj0_distortion_1500d_{23,24}jul26*.png.
"""

from pathlib import Path
import pickle

import numpy as np

TCRAWFOR = Path("/sptlocal/user/tcrawfor")

# 60 bandpowers spanning 0 < l < 6000; centres inferred from his plots, which
# run 0-6000 over exactly these bins.
DELTA_L = 100.0
N_BINS = 60
ELL = (np.arange(N_BINS) + 0.5) * DELTA_L

# The production 1500d map geometry, read off real files in
# /sptlocal/user/tcrawfor/winter_1500d_tau_decon_half1.g3 and matching
# spt3g_software/std_processing/python/field_stubs.py:124-134.
# NOTE this is -57.5, not the -56.0 the 2021 sims used. SFL's Jacobian does not
# depend on the declination of the projection centre at all, so the SFL number
# is unaffected; ZEA's does, mildly.
FIELD_CENTER = dict(alpha_center_deg=0.0, delta_center_deg=-57.5)
# RA 310..50 and dec -70..-42, from spt3g_software/sources/python/fields.py:451-455
FIELD_1500D = dict(ra_half_deg=50.0, dec_min_deg=-70.0, dec_max_deg=-42.0)

VARIANTS = ["", "_cutcorner", "_iso_shrink"]


def load(spec="ee", variant=""):
    """(proj0, proj5) bandpowers from his pickles.

    The arrays are shaped (4, 60), but **rows 1-3 are identically zero**: his
    pickles hold one realization, not four. The shape comes from the ancestor
    script `proj0_distortion_check_cmbtt_20jul26.py`, which allocates four and
    dies in the first pass on a deliberate `print(notavariable)`. Averaging over
    the leading axis therefore divides numerator and denominator by the same 4
    and the returned ratio is still exact -- but nothing here has any realization
    averaging in it, so his bandpowers carry full single-realization scatter.
    """
    path = TCRAWFOR / f"proj0_distortion_{spec}{variant}.pkl"
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return np.asarray(d["cls_proj0"]).mean(0), np.asarray(d["cls_proj5"]).mean(0)


def as_input_spectrum(cl_proj5, lmax=12000):
    """His proj5 bandpowers as a dense C_l(l), power-law continued past l=6000.

    Interpolation is in log C_l against l, because the spectrum falls by four
    orders of magnitude across the range and linear interpolation would put
    visible kinks in it. The continuation slope is fitted in log-log to his top
    ten bandpowers.
    """
    ell = ELL
    cl = np.asarray(cl_proj5, dtype=float)
    good = cl > 0
    ell, cl = ell[good], cl[good]

    out_ell = np.arange(lmax + 1, dtype=float)
    out = np.exp(np.interp(out_ell, ell, np.log(cl)))

    # Power-law continuation above his last bandpower.
    hi = slope_fit = np.polyfit(np.log(ell[-10:]), np.log(cl[-10:]), 1)
    beyond = out_ell > ell[-1]
    out[beyond] = np.exp(slope_fit[1] + slope_fit[0] * np.log(out_ell[beyond]))
    out[0] = 0.0
    return out_ell, out, float(hi[0])


def bin_to_his_bandpowers(ell_fine, y_fine):
    """Average a finely sampled curve into his 100-wide bandpowers."""
    out = np.full(N_BINS, np.nan)
    for i in range(N_BINS):
        lo, hi = i * DELTA_L, (i + 1) * DELTA_L
        sel = (ell_fine >= lo) & (ell_fine < hi)
        if sel.any():
            out[i] = np.nanmean(y_fine[sel])
    return out
