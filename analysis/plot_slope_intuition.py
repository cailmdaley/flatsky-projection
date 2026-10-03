"""What the slope-squared coefficient does, in one projection and three spectra.

Pure prediction: no simulations enter. One projection (SFL) on one footprint
(the 5% disk) fixes the geometry -- the same cells of (s1, s2, W^2) that the
forward model of Section 2.3 integrates over. Three input spectra of quite
different shape are then passed through the *scalar* version of that model,

    ratio(l) = [ sum_c w_c |det J_c|^2 < C(l r_c(chi)) >_chi / sum_c w_c |det J_c| ]
               / C(l),      r(chi) = sqrt(s1^2 cos^2 chi + s2^2 sin^2 chi),

which is Equation (eq:forward) with the misalignment kernel set to one. The
geometry is held fixed across the three curves, so every difference between
them is the spectrum's own shape read through n^2/4:

  * lensed TT -- acoustic peaks imprint, the steep damping tail amplifies;
  * lensed EE -- sharper peaks, deeper troughs, so a larger swing;
  * a featureless power law C ~ l^-4 -- constant slope, so a constant offset,
    the clean n(n+2)/4 <g^2> the expansion predicts.

Run inside the project container:
    OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=12G -p CPUQuota=400% bash env.sh python3 analysis/plot_slope_intuition.py

Output: figures/slope_intuition.pdf
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
FIGURE = ROOT / "figures" / "slope_intuition.pdf"

PROJ = "ProjSFL"
MASK_INDEX = 5          # PERC[5] = 5.0%, the paper's reference footprint
ELL = np.arange(200.0, 4801.0, 10.0)   # the spectra figure's range
POWER_LAW_INDEX = -4.0


def scalar_ratio(mom, i, j, ell, ell_theory, cl_theory, n_chi=128, chunk=64):
    """Footprint-averaged predicted / input scalar spectrum.

    The spin-two kernels of `shear_leakage.predicted_spectra` are replaced by
    unity; everything else -- the (s1, s2) cells, the W^2 |det J|^2 numerator
    weight, the |det J| normalisation, the azimuthal average of the sheared
    wavenumber -- is identical, so the three curves differ only by C(l).
    """
    s1, s2, w = SL._cells(mom, i, j)
    chi = (np.arange(n_chi) + 0.5) * (np.pi / n_chi)
    stretch = np.sqrt(s1[:, None] ** 2 * np.cos(chi) ** 2
                      + s2[:, None] ** 2 * np.sin(chi) ** 2)

    cl = np.asarray(cl_theory, dtype=float)
    weight = w * (s1 * s2) ** 2
    norm = (w * s1 * s2).sum()

    out = np.empty(ell.size)
    for k in range(0, ell.size, chunk):
        sl = slice(k, k + chunk)
        sampled = np.interp(ell[sl][None, :, None] * stretch[:, None, :],
                            ell_theory, cl, left=cl[1], right=0.0)
        out[sl] = (weight @ sampled.mean(-1)) / norm
    return out / np.interp(ell, ell_theory, cl)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = T.load()
    ell_theory = th["ell"].astype(float)

    with np.errstate(divide="ignore"):
        power_law = np.where(ell_theory > 0,
                             np.power(np.maximum(ell_theory, 1.0),
                                      POWER_LAW_INDEX), 0.0)

    inputs = [
        (r"lensed $TT$", th["lensed_TT"], PS.SECONDARY[0], "-"),
        (r"lensed $EE$", th["lensed_EE"], PS.SECONDARY[1], "--"),
        (rf"$C_\ell \propto \ell^{{{POWER_LAW_INDEX:.0f}}}$",
         power_law, PS.SECONDARY[2], ":"),
    ]

    mom = SL.load_moments()
    i, j = MASK_INDEX, S.PROJS.index(PROJ)

    plt.rcParams.update(PS.rc())
    fig, ax = plt.subplots(figsize=(PS.COL_W, 2.4), layout="constrained")

    for lab, cl, col, ls in inputs:
        ratio = scalar_ratio(mom, i, j, ELL, ell_theory, cl)
        ax.plot(ELL, ratio, ls=ls, color=col,
                lw=1.4 * PS.LW if ls == ":" else PS.LW, label=lab)
        print(f"{lab:<28} ratio at l = 1000 ... 4800: "
              + "  ".join(f"{np.interp(l, ELL, ratio):.3f}"
                          for l in (1000, 2000, 3000, 4000, 4800)))

    ax.axhline(1.0, color=PS.RULE, lw=PS.LW_REF, zorder=1)
    ax.set_xlim(ELL[0], ELL[-1])
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{\rm plane}/C_\ell^{\rm sky}$")
    fig.legend(ncol=3, loc="outside upper center")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    main()
