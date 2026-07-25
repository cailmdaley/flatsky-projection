"""First-look figures from the recovered 2021 bandpowers.

Two panels, each answering a question the collaboration is asking right now.

Left: C_l^EE(SFL) / C_l^EE(ZEA). SPT calls these proj0 and proj5, so this is
the exact statistic Tom Crawford plotted for the 1500d field. Theory and all
pixel windows cancel in the ratio, so nothing here depends on a normalization
choice.

Right: C_l^BB for every projection at f_sky = 1.5%. The input sims are
unlensed and E-only, and NaMaster ran with purify_b=True, so all of this is
projection-induced E-to-B leakage.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spectra as S

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


def smooth(y, n=5):
    """Boxcar over bandpowers, NaN-safe, purely cosmetic."""
    k = np.ones(n) / n
    ok = np.isfinite(y)
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    return np.where(den > 0, num / den, np.nan)


def main():
    ee = S.load("EE")
    bb = S.load("BB")
    ell = S.ELL

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ---- left: SFL / ZEA, the proj0/proj5 statistic ----------------------
    ax = axes[0]
    i_sfl, i_zea = S.PROJS.index("ProjSFL"), S.PROJS.index("ProjZEA")
    # mask 6 (f_sky 10%) is 15% NaN for SFL; NaMaster fails there. Drop it.
    use = [0, 1, 2, 3, 4, 5]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(use)))

    ax.axhline(1.0, color="0.3", lw=0.8, zorder=1)
    for c, m in zip(cmap, use):
        r = np.nanmean(ee[m, i_sfl], axis=0) / np.nanmean(ee[m, i_zea], axis=0)
        ax.plot(ell, smooth(r), color=c, lw=1.4,
                label=rf"$f_{{\rm sky}}={S.PERC[m]:g}\%$")

    ax.set_xlim(100, 4000)
    ax.set_ylim(0.55, 2.0)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{EE}(\mathrm{SFL})\ /\ C_\ell^{EE}(\mathrm{ZEA})$")
    ax.set_title("SFL vs ZEA — SPT's proj0 vs proj5", loc="left")
    ax.legend(ncol=2, fontsize=9)

    # ---- right: E-to-B leakage by projection -----------------------------
    ax = axes[1]
    m = 3  # f_sky = 1.5%
    dl = ell * (ell + 1) / (2 * np.pi)
    order = ["ProjBICEP", "ProjSFL", "ProjSIN", "ProjZEA", "ProjSTG"]
    colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(order)))

    for c, proj in zip(colors, order):
        j = S.PROJS.index(proj)
        y = np.nanmean(bb[m, j], axis=0) * dl
        ax.plot(ell, smooth(y), color=c, lw=1.5,
                label=f"{S.PROJ_LABEL[proj]} — {S.PROJ_KIND[proj]}")

    ax.set_yscale("log")
    ax.set_xlim(100, 4000)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{BB}/2\pi\ \ [\mu\mathrm{K}^2]$")
    ax.set_title(r"Projection-induced $E\to B$, $f_{\rm sky}=1.5\%$", loc="left")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    path = OUT / "revival_first_look.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")

    # ---- numbers worth stating in prose ----------------------------------
    print("\nE->B leakage, f_sky=1.5%, mean C_l^BB over 500<l<2000:")
    band = (ell > 500) & (ell < 2000)
    ref = None
    for proj in order:
        j = S.PROJS.index(proj)
        v = np.nanmean(np.nanmean(bb[m, j], axis=0)[band])
        if ref is None:
            ref = v
        print(f"  {S.PROJ_LABEL[proj]:6s} {v:.3e}   ({v/ref:8.2e} x BICEP)")

    print("\nSFL/ZEA high-l behaviour by sky fraction (mean over 3000<l<4000):")
    hi = (ell > 3000) & (ell < 4000)
    for mm in use:
        r = np.nanmean(ee[mm, i_sfl], axis=0) / np.nanmean(ee[mm, i_zea], axis=0)
        lo = (ell > 200) & (ell < 1000)
        print(f"  f_sky={S.PERC[mm]:5g}%:  high-l {np.nanmean(r[hi]):.3f}"
              f"   low-l {np.nanmean(r[lo]):.3f}"
              f"   wiggle rms {np.nanstd(r[(ell>200)&(ell<3000)]):.3f}")


if __name__ == "__main__":
    main()
