"""The measured spectra themselves, projection by projection, against the
parameter-free forward model.

This is the paper's central figure. Everything else in the analysis is a
reduction -- a band average, a ratio, a fit. Here the distortion is shown
where it actually lives: in the bandpowers.

One mask (the 5% disk), five projections, two spectra.

  (a) EE, the measured bandpowers against the input spectrum. Order-unity
      differences between projections are already visible by eye at high l.
  (b) EE divided by the input spectrum -- the scalar remap of
      Eq. (scalar_ratio) made visible. SFL and BICEP lose a third of their
      power at l = 1000 and gain half of it by l = 4000; STG, which has no
      shear at all, loses power monotonically to its magnification; ZEA,
      equal-area and least sheared, stays within a few percent of unity
      until the band limit.
  (c) BB, which is pure leakage: the sims are unlensed, so the true B is
      zero. Five projections spread over five orders of magnitude, with STG
      -- conformal, zero shear -- sitting on the pipeline's own E-to-B floor
      rather than on any prediction.

The smooth curves in (b) and (c) are the same forward model,
`shear_leakage.predicted_spectra`: the sky spectrum read at the sheared
wavenumber |J^T l| with the exact sin^2/cos^2(2 dpsi) kernel, averaged over
mode azimuth and over the footprint with weight W^2 |det J|^2 per unit sky
area, and band-limited at the sims' l = 4000. No free parameters, no fit.

Normalisation. The model returns an unnormalised numerator; the pseudo-C_l
denominator is the plane-area mask integral, sum(W^2 |det J|) per unit sky
area, one power of |det J| lower than the numerator's. Getting this right
matters only for a projection whose magnification varies across the
footprint -- for BICEP the wrong power moves the curve by 20%, for the two
equal-area projections not at all.

Pixel windows. Both spectra are divided by the Nside 8192 HEALPix window and
the 2 arcmin flat-sky top-hat (`spectra.transfer`). What remains is the
bilinear interpolation transfer, which no analytic window captures; it is
common to every projection and to EE and BB alike, and it is why the
measurements sit a few percent under the curves, growing toward l = 4000.

Run:  bash env.sh python3 analysis/plot_projection_spectra.py
Out:  figures/projection_spectra.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shear_leakage as SL
import spectra as S
import theory as T

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "figures" / "projection_spectra.pdf"

MASK_INDEX = 5          # PERC[5] = 5.0%, the paper's reference footprint
REBIN = 5               # display binning: 5 x dl=10 -> dl=50
ELL_MIN, ELL_MAX = 200.0, 3900.0

COLORS = {"ProjSFL": "#d62728", "ProjSIN": "#1f77b4", "ProjSTG": "#7f7f7f",
          "ProjZEA": "#2ca02c", "ProjBICEP": "#ff7f0e"}
MARKS = {"ProjSFL": "o", "ProjSIN": "s", "ProjSTG": "x", "ProjZEA": "^",
         "ProjBICEP": "D"}


def rebin(y, n=REBIN):
    """Average adjacent bandpowers for display. Length must divide evenly."""
    y = np.asarray(y, dtype=float)
    return y[: (len(y) // n) * n].reshape(-1, n).mean(axis=1)


def model_curves(mom, i, j, ell, ell_theory, cl_theory):
    """Forward-modelled BB(l), EE(l) normalised to absolute C_l.

    The numerator carries W^2 |det J|^2 per unit sky area (one |det J| from
    C_plane = |det J| C_sky(J^T l), one more because the mask-squared integral
    runs over plane area); the pseudo-C_l normalisation divides by the same
    integral without the spectrum, i.e. by W^2 |det J| per unit sky area.
    """
    h = mom["hist"][i, j]
    occ = h > 0
    s1, s2, w = mom["cell_s1"][i, j][occ], mom["cell_s2"][i, j][occ], h[occ]
    norm = (w * s1 * s2).sum()
    bb, ee = SL.predicted_spectra(mom, i, j, ell, ell_theory, cl_theory)
    return bb / norm, ee / norm


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.22,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })

    i = MASK_INDEX
    mom = SL.load_moments()
    ee = S.load("EE")
    bb = S.load("BB")
    ell = S.ELL
    cl_ee = S.theory("unlensed_EE")
    cl_bb_lensed = S.theory("lensed_BB")
    transfer = S.transfer()
    th = T.load()
    ell_theory, cl_theory = th["ell"].astype(float), th["unlensed_EE"]

    ellb = rebin(ell)
    keep = (ellb > ELL_MIN) & (ellb < ELL_MAX)
    dl = ellb * (ellb + 1) / (2 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.25, 1.0]})

    # ---- (a) EE bandpowers -------------------------------------------------
    ax = axes[0]
    ax.plot(ellb[keep], (rebin(cl_ee) * dl)[keep], "k-", lw=1.4, zorder=1,
            label="input $EE$")
    for j, proj in enumerate(S.PROJS):
        y = rebin(ee[i, j].mean(0) / transfer) * dl
        ax.plot(ellb[keep], y[keep], MARKS[proj], color=COLORS[proj], ms=2.6,
                mfc="none", mew=0.9, label=S.PROJ_LABEL[proj])
    ax.set_yscale("log")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{EE}/2\pi$  [$\mu$K$^2$]")
    ax.legend(fontsize=8, ncol=3, loc="lower left")

    # ---- (b) EE ratio to input, with the remap prediction ------------------
    ax = axes[1]
    ax.axhline(1.0, color="k", lw=1.0, zorder=1)
    for j, proj in enumerate(S.PROJS):
        r = rebin(ee[i, j].mean(0) / transfer / cl_ee)
        ax.plot(ellb[keep], r[keep], MARKS[proj], color=COLORS[proj], ms=2.6,
                mfc="none", mew=0.9, label=S.PROJ_LABEL[proj])
        _, m_ee = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        ax.plot(ellb[keep], rebin(m_ee / cl_ee)[keep], "-", color=COLORS[proj],
                lw=1.3, alpha=0.85)
    ax.set_ylim(0.25, 2.05)
    ax.set_ylabel(r"measured $C_\ell^{EE}$ / input $C_\ell^{EE}$")
    ax.legend(fontsize=8, ncol=5, loc="lower left", columnspacing=1.0)

    # ---- (c) BB, which is all leakage --------------------------------------
    ax = axes[2]
    ax.plot(ellb[keep], (rebin(cl_bb_lensed) * dl)[keep], "k--", lw=1.2,
            zorder=1, label="lensed $BB$ (reference)")
    for j, proj in enumerate(S.PROJS):
        y = rebin(bb[i, j].mean(0) / transfer) * dl
        ax.plot(ellb[keep], y[keep], MARKS[proj], color=COLORS[proj], ms=2.6,
                mfc="none", mew=0.9, label=S.PROJ_LABEL[proj])
        m_bb, _ = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        if np.nanmax(m_bb) > 0:
            ax.plot(ellb[keep], (rebin(m_bb) * dl)[keep], "-",
                    color=COLORS[proj], lw=1.3, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 3e1)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{BB}/2\pi$  [$\mu$K$^2$]")
    ax.legend(fontsize=8, ncol=3, loc="lower right")

    axes[0].set_xlim(ELL_MIN, ELL_MAX)
    fig.tight_layout(pad=1.0, h_pad=0.6)
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")

    # A few numbers for the caption and the prose, printed rather than typed.
    print(f"\n5% disk, measured / input EE and measured/model, "
          f"pixel windows divided out:")
    for j, proj in enumerate(S.PROJS):
        r = ee[i, j].mean(0) / transfer / cl_ee
        _, m_ee = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        m_bb, _ = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        row = []
        for L in (1000, 2000, 3000, 4000):
            k = np.argmin(abs(ell - L))
            row.append(f"{L}: {r[k]:.3f} ({r[k] / (m_ee[k] / cl_ee[k]):.3f})")
        print(f"  {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row))
    sel = (ell > 500) & (ell < 3000)
    print("\nBB at 5% disk, 500 < l < 3000 mean [uK^2]:")
    for j, proj in enumerate(S.PROJS):
        m_bb, _ = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        meas = (bb[i, j].mean(0) / transfer)[sel].mean()
        print(f"  {S.PROJ_LABEL[proj]:<6s} measured {meas:9.3e}   "
              f"model {m_bb[sel].mean():9.3e}   "
              f"<g^2> {mom['g2'][i, j]:9.3e}")


if __name__ == "__main__":
    main()
