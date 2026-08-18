"""The measured spectra themselves, projection by projection, against the
parameter-free forward model.

This is the paper's central pair of figures. Everything else in the analysis
is a reduction -- a band average, a ratio, a fit. Here the distortion is
shown where it actually lives: in the bandpowers.

One mask (the 5% disk), five projections, two spectra, split into two
standalone figures so the EE story and the BB story can each live in their
own subsection:

  EE figure (top) the measured bandpowers against the input spectrum.
      Order-unity differences between projections are already visible by eye
      at high l.
  EE figure (bottom) EE divided by the input spectrum -- the scalar remap of
      Eq. (scalar_ratio) made visible. SFL and BICEP lose a third of their
      power at l = 1000 and gain half of it by l = 4000; STG, which has no
      shear at all, loses power monotonically to its magnification; ZEA,
      equal-area and least sheared, stays within a few percent of unity
      until the band limit.
  BB figure, which is pure leakage: the sims are unlensed, so the true B is
      zero. Five projections spread over five orders of magnitude, with STG
      -- conformal, zero shear -- sitting on the pipeline's own E-to-B floor
      rather than on any prediction.

The smooth curves are the same forward model, `shear_leakage.predicted_spectra`:
the sky spectrum read at the sheared wavenumber |J^T l| with the exact
sin^2/cos^2(2 dpsi) kernel, averaged over mode azimuth and over the footprint
with weight W^2 |det J|^2 per unit sky area, and band-limited at the sims'
l = 4000. No free parameters, no fit.

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
Out:  figures/projection_spectra_ee.pdf, figures/projection_spectra_bb.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shear_leakage as SL
import spectra as S
import theory as T
import paper_style as PS

ROOT = Path(__file__).resolve().parent.parent
FIGURE_EE = ROOT / "figures" / "projection_spectra_ee.pdf"
FIGURE_BB = ROOT / "figures" / "projection_spectra_bb.pdf"

MASK_INDEX = 5          # PERC[5] = 5.0%, the paper's reference footprint
REBIN = 5               # display binning: 5 x dl=10 -> dl=50
ELL_MIN, ELL_MAX = 200.0, 3900.0

COLORS = PS.PROJ_COLORS
MARKS = PS.PROJ_MARKERS

MS, MEW = 3.0, 0.9


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


def load_data():
    """Everything both figures need, loaded once."""
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

    return dict(i=i, mom=mom, ee=ee, bb=bb, ell=ell, cl_ee=cl_ee,
                cl_bb_lensed=cl_bb_lensed, transfer=transfer,
                ell_theory=ell_theory, cl_theory=cl_theory,
                ellb=ellb, keep=keep, dl=dl)


def set_style(plt):
    plt.rcParams.update({
        "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.18,
        "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": PS.RULE, "axes.linewidth": 0.7,
        "xtick.color": PS.INK_SOFT, "ytick.color": PS.INK_SOFT,
        "legend.frameon": False,
    })


def plot_ee(plt, d):
    """EE bandpowers (top) and EE/input ratio (bottom), self-sufficient."""
    i, mom, ee = d["i"], d["mom"], d["ee"]
    ell, ell_theory, cl_theory = d["ell"], d["ell_theory"], d["cl_theory"]
    cl_ee, transfer = d["cl_ee"], d["transfer"]
    ellb, keep, dl = d["ellb"], d["keep"], d["dl"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True,
                              gridspec_kw={"height_ratios": [1.0, 1.25]})

    # ---- (a) EE bandpowers -------------------------------------------------
    ax = axes[0]
    input_line, = ax.plot(ellb[keep], (rebin(cl_ee) * dl)[keep], "-",
                           color=PS.INK, lw=1.4, zorder=1, label="input $EE$")
    proj_handles = []
    for j, proj in enumerate(S.PROJS):
        y = rebin(ee[i, j].mean(0) / transfer) * dl
        h, = ax.plot(ellb[keep], y[keep], MARKS[proj], color=COLORS[proj],
                      ms=MS, mfc="none", mew=MEW, label=S.PROJ_LABEL[proj])
        proj_handles.append(h)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{EE}/2\pi$  [$\mu$K$^2$]")

    # ---- (b) EE ratio to input, with the remap prediction ------------------
    ax = axes[1]
    ax.axhline(1.0, color=PS.RULE, lw=0.8, zorder=1)
    for j, proj in enumerate(S.PROJS):
        r = rebin(ee[i, j].mean(0) / transfer / cl_ee)
        ax.plot(ellb[keep], r[keep], MARKS[proj], color=COLORS[proj], ms=MS,
                mfc="none", mew=MEW)
        _, m_ee = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        ax.plot(ellb[keep], rebin(m_ee / cl_ee)[keep], "-", color=COLORS[proj],
                lw=1.3, alpha=0.9)
    ax.set_ylim(0.25, 2.05)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"measured $C_\ell^{EE}$ / input $C_\ell^{EE}$")

    axes[0].set_xlim(ELL_MIN, ELL_MAX)

    # Self-sufficient legend: five projections plus the input-EE reference.
    handles = proj_handles + [input_line]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS] + ["input $EE$"]
    fig.legend(handles, labels, fontsize=8.5, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.3,
               handletextpad=0.5, frameon=False)

    fig.tight_layout(pad=1.0, h_pad=0.6, rect=(0, 0, 1, 0.945))
    FIGURE_EE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE_EE)
    print(f"wrote {FIGURE_EE}")


def plot_bb(plt, d):
    """BB, pure leakage -- single-panel figure, columnwidth-friendly."""
    i, mom = d["i"], d["mom"]
    bb, ell = d["bb"], d["ell"]
    ell_theory, cl_theory = d["ell_theory"], d["cl_theory"]
    cl_bb_lensed, transfer = d["cl_bb_lensed"], d["transfer"]
    ellb, keep, dl = d["ellb"], d["keep"], d["dl"]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))

    ref_line, = ax.plot(ellb[keep], (rebin(cl_bb_lensed) * dl)[keep], "--",
                         color=PS.INK, lw=1.2, zorder=1,
                         label="lensed $BB$ (reference)")
    proj_handles = []
    for j, proj in enumerate(S.PROJS):
        y = rebin(bb[i, j].mean(0) / transfer) * dl
        h, = ax.plot(ellb[keep], y[keep], MARKS[proj], color=COLORS[proj],
                      ms=MS, mfc="none", mew=MEW, label=S.PROJ_LABEL[proj])
        proj_handles.append(h)
        m_bb, _ = model_curves(mom, i, j, ell, ell_theory, cl_theory)
        if np.nanmax(m_bb) > 0:
            ax.plot(ellb[keep], (rebin(m_bb) * dl)[keep], "-",
                    color=COLORS[proj], lw=1.3, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 3e1)
    ax.set_xlim(ELL_MIN, ELL_MAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{BB}/2\pi$  [$\mu$K$^2$]")

    # Self-sufficient legend: five projections plus the lensed-BB reference.
    handles = proj_handles + [ref_line]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS] + ["lensed $BB$ (reference)"]
    fig.legend(handles, labels, fontsize=8.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.3,
               handletextpad=0.5, frameon=False)

    fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.88))
    FIGURE_BB.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE_BB)
    print(f"wrote {FIGURE_BB}")


def print_numbers(d):
    """A few numbers for the caption and the prose, printed rather than typed."""
    i, mom, ee, bb = d["i"], d["mom"], d["ee"], d["bb"]
    ell, ell_theory, cl_theory = d["ell"], d["ell_theory"], d["cl_theory"]
    cl_ee, transfer = d["cl_ee"], d["transfer"]

    print(f"\n5% disk, measured / input EE and measured/model, "
          f"pixel windows divided out:")
    for j, proj in enumerate(S.PROJS):
        r = ee[i, j].mean(0) / transfer / cl_ee
        _, m_ee = model_curves(mom, i, j, ell, ell_theory, cl_theory)
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


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_style(plt)
    d = load_data()
    plot_ee(plt, d)
    plot_bb(plt, d)
    print_numbers(d)


if __name__ == "__main__":
    main()
