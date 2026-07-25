"""Re-derive the projection-comparison spectra from the surviving 2021 bandpowers.

Why this exists: the 2021 pipeline's own plotting script
(/sptlocal/analysis/pol_project2020/workflow/scripts/plot_namaster_spectra.py)
has two defects. It writes output filenames from the whole `perc_str` list
instead of one element (the crash that parked the project), and its loading
loop assigns `cl_flat_EE[ii]` inside the projection loop, so every projection
slot ends up holding the last projection read. Nothing here reuses it.

What the 2021 pipeline actually computed, for the record:
  - Input: HEALPix Nside 8192 *unlensed* TQU sims (so the true B is zero).
  - map2alm to lmax 4000, monopole and dipole filtered.
  - Projected to a flat grid with bilinear interpolation, oversampled 8x,
    rebinned to 2 arcmin pixels.
  - spt3g maps.FlattenPol applied, i.e. the projection-induced spin-2
    rotation is undone before spectra are taken.
  - NaMaster NmtFieldFlat with purify_b=True on apodized circular masks
    (C2, 2.5 deg apodization), seven radii spanning f_sky 0.75% to 10%.
  - 20 sims, bandpowers of width dl = 10 out to l = 4000.

purify_b=True is the load-bearing detail: mask-induced E-to-B leakage is
removed by construction, so residual BB is projection-induced, not mask-induced.
"""

import numpy as np
from pathlib import Path

SPECTRA_DIR = Path("/sptlocal/analysis/pol_project2020/results/spectra")
THEORY_EE = Path(__file__).resolve().parent.parent / "Clee.dat"

PROJS = ["ProjSFL", "ProjSIN", "ProjSTG", "ProjZEA", "ProjBICEP"]
# How the paper names them, and what they are.
PROJ_LABEL = {
    "ProjSFL": "SFL",
    "ProjSIN": "SIN",
    "ProjSTG": "STG",
    "ProjZEA": "ZEA",
    "ProjBICEP": "BICEP",
}
PROJ_KIND = {
    "ProjSFL": "pseudocylindrical, equal-area",
    "ProjSIN": "zenithal",
    "ProjSTG": "conformal",
    "ProjZEA": "zenithal equal-area",
    "ProjBICEP": "cylindrical",
}
PERC = np.array([0.75, 1.00, 1.25, 1.5, 3.0, 5.0, 10.0])

NSIMS, NBINS, DELTA_L = 20, 400, 10
RES_ARCMIN = 2.0
NSIDE_IN = 8192

BIN_LO = np.arange(NBINS) * DELTA_L
BIN_HI = (np.arange(NBINS) + 1) * DELTA_L
ELL = 0.5 * (BIN_LO + BIN_HI)


def load(spec="EE"):
    """Bandpowers as (n_masks, n_projs, NSIMS, NBINS)."""
    out = np.full((len(PERC), len(PROJS), NSIMS, NBINS), np.nan)
    for i in range(len(PERC)):
        for j, proj in enumerate(PROJS):
            path = SPECTRA_DIR / f"Cl_flatsky_namaster_unlensed_{spec}_{proj}_mask{i}.txt"
            arr = np.loadtxt(path)
            if arr.shape != (NSIMS, NBINS):
                raise ValueError(f"{path.name}: {arr.shape} != {(NSIMS, NBINS)}")
            out[i, j] = arr
    return out


def bin_theory(cl):
    """Average an unbinned C_l over each bandpower.

    Binning matters: the acoustic peaks move fast compared to dl = 10, so
    evaluating theory at bin centres instead of averaging over the bin
    manufactures percent-level wiggles that look like a projection effect.
    """
    binned = np.empty(NBINS)
    for i, (lo, hi) in enumerate(zip(BIN_LO, BIN_HI)):
        seg = cl[lo:hi]
        binned[i] = np.nanmean(seg)
    return binned


def flat_pixwin(ell, res_arcmin=RES_ARCMIN, n_phi=512):
    """Azimuthally averaged square-pixel window, in C_l (i.e. already squared).

    Rebinning an oversampled grid down to `res_arcmin` pixels is a top-hat
    average, whose 2D transform is sinc(lx d/2) sinc(ly d/2). NaMaster bins in
    |l| annuli, so average |W|^2 around each annulus.
    """
    d = np.deg2rad(res_arcmin / 60.0)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    lx = np.outer(ell, np.cos(phi))
    ly = np.outer(ell, np.sin(phi))
    w = np.sinc(lx * d / (2 * np.pi)) * np.sinc(ly * d / (2 * np.pi))
    return (w ** 2).mean(axis=1)


def healpix_pixwin(ell, nside=NSIDE_IN):
    """HEALPix pixel window in C_l, interpolated onto bandpower centres."""
    import healpy as hp
    pw = hp.pixwin(nside, pol=True)[1]  # polarization window
    l = np.arange(len(pw))
    return np.interp(ell, l, pw) ** 2


def theory_ee():
    """Binned unlensed CAMB EE, matching the sims' input spectrum."""
    cl = np.loadtxt(THEORY_EE)
    cl[0] = 0.0
    return bin_theory(cl)


def transfer(ell=ELL):
    """Total expected multiplicative suppression of C_l from pixelization."""
    return healpix_pixwin(ell) * flat_pixwin(ell)
