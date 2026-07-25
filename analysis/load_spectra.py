"""Load the surviving 2021 NaMaster bandpowers into one array.

The pipeline wrote one text file per (spectrum, projection, mask):
    Cl_flatsky_namaster_unlensed_{EE,BB}_{proj}_mask{0..6}.txt
each holding Nsims=20 rows x 400 bandpowers (delta_ell = 10, ell_max = 4000).

This module deliberately re-derives everything from the raw text files rather
than reusing workflow/scripts/plot_namaster_spectra.py, which has an indexing
bug that collapses all projections onto one.
"""

import numpy as np
from pathlib import Path

SPECTRA_DIR = Path("/sptlocal/analysis/pol_project2020/results/spectra")

PROJS = ["ProjSFL", "ProjSIN", "ProjSTG", "ProjZEA", "ProjBICEP"]
PERC = [0.75, 1.00, 1.25, 1.5, 3.0, 5.0, 10.0]
PERC_STR = ["0p75", "1p0", "1p25", "1p5", "3", "5", "10"]
NSIMS = 20
NBINS = 400
DELTA_L = 10

# Bandpower edges as defined in the pipeline: NmtBinFlat(l0, lf) with
# l0 = 10*i, lf = 10*(i+1). Effective ell is the bin centre.
BIN_LO = np.arange(NBINS) * DELTA_L
BIN_HI = (np.arange(NBINS) + 1) * DELTA_L
ELL = 0.5 * (BIN_LO + BIN_HI)


def load(spec="EE"):
    """Return array of shape (n_masks, n_projs, NSIMS, NBINS)."""
    out = np.full((len(PERC), len(PROJS), NSIMS, NBINS), np.nan)
    for i in range(len(PERC)):
        for j, proj in enumerate(PROJS):
            path = SPECTRA_DIR / f"Cl_flatsky_namaster_unlensed_{spec}_{proj}_mask{i}.txt"
            if path.exists():
                arr = np.loadtxt(path)
                if arr.shape != (NSIMS, NBINS):
                    raise ValueError(f"{path.name}: got {arr.shape}, want {(NSIMS, NBINS)}")
                out[i, j] = arr
    return out


if __name__ == "__main__":
    ee = load("EE")
    bb = load("BB")
    print(f"EE {ee.shape}  finite: {np.isfinite(ee).all()}")
    print(f"BB {bb.shape}  finite: {np.isfinite(bb).all()}")

    # The question that matters first: are the projections actually distinct?
    print("\nAre projections distinct? (mask3 = f_sky 1.5%, mean over sims)")
    m = 3
    ref = ee[m, 0].mean(axis=0)
    for j, proj in enumerate(PROJS):
        cur = ee[m, j].mean(axis=0)
        rel = np.abs(cur - ref) / np.abs(ref)
        band = slice(20, 300)  # ell 200-3000
        print(f"  {proj:10s}  max|dC/C| vs ProjSFL over ell 200-3000: {rel[band].max():.4f}")

    print("\nBB amplitude by projection (mask3, mean over sims, ell 500-2000):")
    band = slice(50, 200)
    for j, proj in enumerate(PROJS):
        print(f"  {proj:10s}  mean C_l^BB = {bb[m, j].mean(axis=0)[band].mean():.3e}")
