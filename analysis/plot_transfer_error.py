"""What a diagonal transfer function misses that the coupling matrix keeps.

Pure prediction: no simulations enter. `shear_leakage.coupling_matrix` writes
Eq. (forward) as a matrix R(l, l') from the sky spectrum to the expected
flat-sky spectrum, for one projection on one footprint -- here SPT's 1500d
field, for the most sheared projection (SFL) and an equal-area zenithal one
with little shear (ZEA).

Plotted is the error of a diagonal correction. F_b = <R C>_b / <C>_b is
calibrated on the unlensed Planck TT and EE, per dl = 50 bandpower, and
applied to a white spectrum; the curve is the recovered bandpower over the
truth, minus one (`shear_leakage.diagonal_transfer_error`). Both projections
are equal-area, so a D_l-flat (tSZ-like) spectrum gives the same curves. F_b
encodes the CMB's own slope and acoustic phase, which a foreground does not
share.

Run inside the project container:
    OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=12G bash env.sh python3 analysis/plot_transfer_error.py

Output: figures/transfer_error.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as PS
import shear_leakage as SL
import spectra as S
import theory as T

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "figures" / "transfer_error.pdf"

PROJS = ("ProjSFL", "ProjZEA")
STEP, PER_BAND = 5.0, 10            # dl = 50 bandpowers from 5-l sampling
ELL_LO, ELL_HI = 500.0, 4800.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = T.load()
    lp = th["ell"].astype(float)
    white = np.where(lp >= 2, 1.0, 0.0)
    mom = SL.field_moments_1500d()
    ell_b = np.arange(ELL_LO + STEP / 2, ELL_HI, STEP)
    ell_c = ell_b.reshape(-1, PER_BAND).mean(-1)

    plt.rcParams.update(PS.rc())
    fig, ax_e = plt.subplots(figsize=(PS.COL_W, 2.4), layout="constrained")

    for proj in PROJS:
        j = S.PROJS.index(proj)
        col = PS.PROJ_COLORS[proj]
        R = SL.coupling_matrix(mom, 0, j, ell_b, lp, band_limit=None)
        for X, ls in (("TT", "-"), ("EE", "--")):
            err = SL.diagonal_transfer_error(R[f"{X}<-{X}"], ell_b, lp,
                                             th[f"unlensed_{X}"], white,
                                             per_band=PER_BAND)
            ax_e.plot(ell_c, 100 * err, color=col, ls=ls, lw=PS.LW)
            print(f"  {X} diagonal-correction error at l = 1000/2000/3000/4000:"
                  + "".join(f" {100 * np.interp(L, ell_c, err):+.1f}%"
                            for L in (1000, 2000, 3000, 4000)))

    ax_e.axhline(0.0, color=PS.RULE, lw=PS.LW_REF, zorder=1)
    ax_e.set_xlim(ELL_LO, ELL_HI)
    ax_e.set_xlabel(r"$\ell$")
    ax_e.set_ylabel(r"$\hat C_b / C_b - 1$ [\%]")
    handles = ([Line2D([], [], color=PS.PROJ_COLORS[p]) for p in PROJS]
               + [Line2D([], [], color=PS.INK_SOFT, ls=ls) for ls in ("-", "--")])
    labels = [PS.PROJ_LABEL[p] for p in PROJS] + [r"$TT$", r"$EE$"]
    fig.legend(handles, labels, ncol=4, loc="outside upper center")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    main()
