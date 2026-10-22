"""The projection as a coupling matrix, and what a diagonal correction misses.

Pure prediction: no simulations enter. `shear_leakage.coupling_matrix` writes
Eq. (forward) as a matrix R(l, l') from the sky spectrum to the expected
flat-sky spectrum, for one projection on one footprint -- here SPT's 1500d
field, for the most sheared projection (SFL) and an equal-area zenithal one
with little shear (ZEA).

Left: one row of R, as the kernel l R(l, l') against l'/l. A plane multipole
l reads the sky at l r, r = |J^T l| / l, over the footprint and every mode
azimuth, so the row is the distribution of r weighted by W^2 |det J|^2 and
the spin kernel -- supported on [s2_min, s1_max] and independent of l below
the band limit, so every row is this curve. Its width is the shear: SFL
draws on sky multipoles from about two thirds of l to one and a half times
it, ZEA on a few percent either side (the script prints both). The B<-E
row is the same band through the sin^2 kernel, which vanishes on the
principal axes.

Right: the error of a diagonal correction. F_b = <R C>_b / <C>_b calibrated
on the unlensed Planck TT and EE, per dl = 50 bandpower, applied to a white
spectrum; plotted is the recovered bandpower over the truth, minus one
(`shear_leakage.diagonal_transfer_error`). Both projections are equal-area,
so a D_l-flat (tSZ-like) spectrum gives the same curves. F_b encodes the
CMB's own slope and acoustic phase, which a foreground does not share.

Run inside the project container:
    OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=12G bash env.sh python3 analysis/plot_coupling_matrix.py

Output: figures/coupling_matrix.pdf
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
FIGURE = ROOT / "figures" / "coupling_matrix.pdf"

PROJS = ("ProjSFL", "ProjZEA")
ELL_ROW = 4000.0                    # the row drawn; without a band limit every row is the same
RHO_BIN = 0.008                     # display bins in l'/l: 32 sky multipoles at ELL_ROW
STEP, PER_BAND = 5.0, 10            # dl = 50 bandpowers from 5-l sampling
ELL_LO, ELL_HI = 500.0, 4800.0


def kernel_row(R_row, ell_prime, ell):
    """l R(l, l') rebinned in rho = l'/l, as a density in rho."""
    rho = ell_prime / ell
    edges = np.arange(0.5, 1.6 + RHO_BIN, RHO_BIN)
    mass, _ = np.histogram(rho, bins=edges, weights=R_row)
    return 0.5 * (edges[1:] + edges[:-1]), mass / RHO_BIN


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
    fig, (ax_k, ax_e) = plt.subplots(1, 2, figsize=(PS.TEXT_W, 2.5),
                                     layout="constrained")

    for proj in PROJS:
        j = S.PROJS.index(proj)
        col = PS.PROJ_COLORS[proj]
        R = SL.coupling_matrix(mom, 0, j, [ELL_ROW], lp, band_limit=None)
        for block, ls in (("EE<-EE", "-"), ("BB<-EE", "--")):
            rho, k = kernel_row(R[block][0], lp, ELL_ROW)
            k = np.where(k > 0, k, np.nan)
            ax_k.plot(rho, k, color=col, ls=ls, lw=PS.LW)
        print(f"{PS.PROJ_LABEL[proj]}: row sums at l = {ELL_ROW:.0f}: "
              f"EE<-EE {R['EE<-EE'].sum():.5f}  BB<-EE {R['BB<-EE'].sum():.5f}"
              f"  support l'/l in [{lp[R['EE<-EE'][0] > 0].min() / ELL_ROW:.3f},"
              f" {lp[R['EE<-EE'][0] > 0].max() / ELL_ROW:.3f}]")

        R = SL.coupling_matrix(mom, 0, j, ell_b, lp, band_limit=None)
        for X, ls in (("TT", "-"), ("EE", "--")):
            err = SL.diagonal_transfer_error(R[f"{X}<-{X}"], ell_b, lp,
                                             th[f"unlensed_{X}"], white,
                                             per_band=PER_BAND)
            ax_e.plot(ell_c, 100 * err, color=col, ls=ls, lw=PS.LW)
            print(f"  {X} diagonal-correction error at l = 1000/2000/3000/4000:"
                  + "".join(f" {100 * np.interp(L, ell_c, err):+.1f}%"
                            for L in (1000, 2000, 3000, 4000)))

    ax_k.set_yscale("log")
    ax_k.set_xlim(0.6, 1.45)
    ax_k.set_ylim(1e-3, 1e2)
    ax_k.axvline(1.0, color=PS.RULE, lw=PS.LW_REF, zorder=1)
    ax_k.set_xlabel(r"$\ell'/\ell$")
    ax_k.set_ylabel(r"$\ell\, R(\ell, \ell')$")
    ax_k.legend([Line2D([], [], color=PS.INK_SOFT, ls="-"),
                 Line2D([], [], color=PS.INK_SOFT, ls="--")],
                [r"$EE \leftarrow EE$", r"$BB \leftarrow EE$"],
                loc="upper left")

    ax_e.axhline(0.0, color=PS.RULE, lw=PS.LW_REF, zorder=1)
    ax_e.set_xlim(ELL_LO, ELL_HI)
    ax_e.set_xlabel(r"$\ell$")
    ax_e.set_ylabel(r"diagonal correction, $\hat C_b / C_b - 1$ [\%]")
    ax_e.legend([Line2D([], [], color=PS.INK_SOFT, ls="-"),
                 Line2D([], [], color=PS.INK_SOFT, ls="--")],
                [r"$TT$", r"$EE$"], loc="lower left")

    fig.legend([Line2D([], [], color=PS.PROJ_COLORS[p]) for p in PROJS],
               [PS.PROJ_LABEL[p] for p in PROJS], ncol=len(PROJS),
               loc="outside upper center")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    main()
