"""Predict Crawford's proj0/proj5 ratio exactly, from his own configuration.

What changed, and why it matters
--------------------------------
Crawford's generating script was found on disk at
`/sptlocal/user/tcrawfor/code/spt3g_software/scratch/tcrawfor/proj0_distortion_check_cmbtt_24jul26.py`.
Reading it turns three previously-estimated quantities into known ones, and
the prediction below has no fitted or guessed input left anywhere.

**His input spectrum is analytic.** He does not project a sky simulation. He
draws a Gaussian random field directly on a flat ZEA grid with
`quick_flatsky_routines.cmb_flatsky`, using lensed Planck 2018 C_l, and
declares that field to be the proj5 map. So the "sky" spectrum is known
exactly out to his band limit of l = 7999, and the earlier worry -- that
predictions above l = 4000 leaned on extrapolating his measured bandpowers --
is gone. His "EE" runs are the same T-only machinery with the EE spectrum
substituted into the TT slot, so they test spectral slope, not polarization.

**The operator is the composite Jacobian.** Because the input lives on the ZEA
plane rather than on the sky, the only thing acting is the plane-to-plane map,
and `geometry.relative_jacobian` gives it directly. The predicted ratio is then
P_proj0 / P_proj5 with P_proj5 *equal to the input by construction* -- there is
no sky spectrum to model and nothing to cancel.

**His masks are reconstructible.** `crawford_masks` rebuilds all three from the
real SPT-3G apodization mask, so the cut-corner and iso-shrink predictions are
made over his actual geometry instead of over guessed shapes.

What the model still does not contain
-------------------------------------
Nothing about sampling. He reprojects with `interp=False, rebin=1`, i.e.
nearest-neighbour with no sub-pixel averaging, which is the configuration
Umilta identified in 2021 as the one that aliases. Any excess the geometry
does not account for is therefore a candidate resampling signature, and
`run_crawford_sims.py` measures it by flipping that one switch.
"""

import numpy as np
from pathlib import Path

import crawford as C
import crawford_masks as CM
import geometry as G

# The file `quick_flatsky_routines.camb2cl` reads, in our own checkout.
CAMB = Path("/home/cailmdaley/code/python/spt3g_software/simulations/data/camb/"
            "planck18_TTEEEE_lowl_lowE_lensing_highacc/"
            "planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat")

# `camb2cl(8000)` truncates at index 8000, and `cmb_flatsky` zeroes the grid
# above max(ell) = 7999. That is a hard band limit and the model must impose it.
BAND_LIMIT = 7999

PROJ0_KW = dict(delta_center_deg=CM.DELTA_CENTER_PROJ0)
PROJ5_KW = dict(delta_center_deg=CM.DELTA_CENTER_PROJ5)


def theory_input(spec="tt", lmax=BAND_LIMIT):
    """His input C_l, built the way `camb2cl` builds it.

    Lensed Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing, D_l in uK^2 on
    disk, returned as C_l with l = 0 and 1 zeroed. `spec='ee'` returns the EE
    column, which is what he substitutes into the TT slot for his "EE" runs.
    """
    dat = np.loadtxt(CAMB)
    ell_file = dat[:, 0].astype(int)
    col = {"tt": 1, "ee": 2, "bb": 3, "te": 4}[spec]
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.zeros(lmax + 1)
    sel = ell_file <= lmax
    l = ell_file[sel]
    cl[l] = dat[sel, col] * (2 * np.pi) / (l * (l + 1.0))
    return ell, cl


def bin_like_crawford(ell_fine, y_fine, n_bins=C.N_BINS, delta_l=C.DELTA_L):
    """Bin a curve the way `cl_flatsky` does: flat average over 2D modes.

    It bins the 2D power grid by `floor(l/delta_l)` and divides by the number
    of modes, so within a band each multipole carries weight proportional to
    the number of Fourier cells at that |l|, i.e. proportional to l. Using a
    plain unweighted mean instead shifts the effective band centre low by a
    few multipoles -- negligible in the middle of the range, not negligible at
    l < 500 where the spectrum is steep.
    """
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (ell_fine >= i * delta_l) & (ell_fine < (i + 1) * delta_l)
        if sel.any():
            w = ell_fine[sel]
            out[i] = np.sum(w * y_fine[sel]) / np.sum(w)
    return out


def geometry_of(variant="full", ngrid=5000, res_arcmin=1.0):
    """(s1, s2, mask^2) of the composite proj5 -> proj0 map over his footprint."""
    d = CM.cached(variant, ngrid, res_arcmin)
    a = G.relative_jacobian(
        "ProjSFL", "ProjZEA",
        d["ra0"].astype(float), d["dec0"].astype(float),
        to_kw=PROJ0_KW, from_kw=PROJ5_KW)
    s1, s2 = G.singular_values(a)
    return s1, s2, d["w0"].astype(float) ** 2


def predict(variant="full", spec="tt", ngrid=5000, res_arcmin=1.0,
            n_psi=96, n_bins=64):
    """Predicted proj0/proj5 bandpower ratio, on his 60 bandpowers.

    Returns (ell_bandpower, ratio_predicted).
    """
    s1, s2, w = geometry_of(variant, ngrid, res_arcmin)
    ell_th, cl_th = theory_input(spec)

    ell = np.arange(1.0, C.N_BINS * C.DELTA_L + 1.0)
    proj0 = G.predicted_spectrum(ell, s1, s2, w, ell_th, cl_th,
                                 band_limit=BAND_LIMIT,
                                 n_bins=n_bins, n_psi=n_psi)
    proj5 = np.interp(ell, ell_th, cl_th)

    return C.ELL, bin_like_crawford(ell, proj0) / bin_like_crawford(ell, proj5)


def measured(variant="", spec="tt"):
    """His measured ratio, averaged over his 4 realizations."""
    p0, p5 = C.load(spec=spec, variant=variant)
    return C.ELL, p0 / p5


VARIANT_FILE = {"full": "", "cutcorner": "_cutcorner",
                "iso_shrink": "_iso_shrink"}
