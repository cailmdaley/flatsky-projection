"""Sanity plot for the regenerated theory spectra: is the lensing BB floor real?

TT, EE (lensed + unlensed), and BB (lensed + unlensed) as D_l = l(l+1)C_l/2pi,
log-log, so three things are checkable by eye at once: the acoustic peaks
land where they should, lensing measurably smooths EE at high ell (it eats
into the damping tail), and unlensed BB is a hard zero while lensed BB traces
the familiar broad hump peaking around ell ~ 1000 -- the "lensing floor" that
the projection-effects figures compare leakage against.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory as T

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


def dl(ell, cl):
    """Raw C_l -> D_l = l(l+1)C_l/2pi. Zero at ell=0 is undefined; mask it."""
    out = np.full(len(ell), np.nan)
    out[1:] = ell[1:] * (ell[1:] + 1) / (2 * np.pi) * cl[1:]
    return out


def main():
    d = T.load()
    ell = d["ell"]
    h = d["header"]  # T.load() already unpacks this to a plain dict

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(ell, dl(ell, d["lensed_TT"]), color="0.2", lw=1.3, label="TT (lensed)")
    ax.plot(ell, dl(ell, d["unlensed_EE"]), color="C0", lw=1.1, ls="--",
             label="EE (unlensed)")
    ax.plot(ell, dl(ell, d["lensed_EE"]), color="C0", lw=1.3, label="EE (lensed)")
    ax.plot(ell, dl(ell, d["lensed_BB"]), color="C3", lw=1.6, label="BB (lensed)")

    # The BB peak, called out explicitly -- this is the number the figures
    # that motivated this task need to quote.
    peak_i = np.nanargmax(dl(ell, d["lensed_BB"]))
    peak_ell, peak_val = ell[peak_i], dl(ell, d["lensed_BB"])[peak_i]
    ax.scatter([peak_ell], [peak_val], color="C3", zorder=5, s=25)
    ax.annotate(
        f"lensing BB floor peak:\n" + r"$D_\ell \approx$" + f" {peak_val:.3f}"
        r" $\mu\mathrm{K}^2$ at $\ell \approx$" + f" {peak_ell}",
        xy=(peak_ell, peak_val), xytext=(peak_ell * 1.6, peak_val * 4),
        fontsize=9, color="C3",
        arrowprops=dict(arrowstyle="-", color="C3", lw=0.8),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, h["lmax"])
    ax.set_ylim(1e-7, 1e4)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi\ \ [\mu\mathrm{K}^2]$")
    ax.set_title("Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing, CAMB "
                  f"{h['camb_version']}", loc="left", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    path = OUT / "theory_spectra.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")

    print(f"\nLensing BB floor peak: D_l = {peak_val:.4f} uK^2 at ell = {peak_ell}")
    print("Reference values (lensed BB, D_l in uK^2):")
    for l in [100, 500, 1000, 1500, 2000, 3000]:
        print(f"  ell={l:5d}  D_BB = {dl(ell, d['lensed_BB'])[l]:.4e} uK^2")


if __name__ == "__main__":
    main()
