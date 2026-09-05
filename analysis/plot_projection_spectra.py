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
      power at l = 1000 and more than double it by l = 4500; STG, which has
      no shear at all, loses power monotonically to its magnification; ZEA,
      equal-area and least sheared, stays within a few percent of unity.
  BB figure (top), which is pure leakage: the sims are unlensed, so the true
      B is zero. Five projections spread over close to six decades on the 5%
      disk, with STG -- conformal, zero shear -- sitting on the pipeline's own
      E-to-B floor rather than on any prediction. The span is computed, not
      remembered: `paper_numbers.py` writes it as the \bbdecades macro.
  BB figure (bottom) BB/EE, the leakage relation itself, against the same
      forward model. STG is left off this axis: with zero shear its ratio is
      the floor, not a leakage, and it is quoted in the text instead.

The smooth curves are the same forward model, `shear_leakage.predicted_spectra`:
the sky spectrum read at the sheared wavenumber |J^T l| with the exact
sin^2/cos^2(2 dpsi) kernel, averaged over mode azimuth and over the footprint
with weight W^2 |det J|^2 per unit sky area, and band-limited at the sims'
l = 6000. No free parameters, no fit.

Normalisation. The model returns an unnormalised numerator; the pseudo-C_l
denominator is the plane-area mask integral, sum(W^2 |det J|) per unit sky
area, one power of |det J| lower than the numerator's. Getting this right
matters only for a projection whose magnification varies across the
footprint -- for BICEP the wrong power moves the curve by 20%, for the two
equal-area projections not at all.

Range. The production run carries sky power to l = 6000, but the 2 arcmin
pixel has a Nyquist multipole of 5400 and the resampling residual arrives
before it: above l ~ 4800 every projection, STG included, rises onto one
common curve that is the pixel and not the geometry. These figures stop at
l = 4800 and make no claim beyond it.

Pixel windows. The production bandpowers already have the 2 arcmin flat-sky
top-hat divided out and the spurious HEALPix window restored (see
`spectra`), so `spectra.transfer` is unity here. What remains is the bilinear
interpolation transfer, which no analytic window captures; it is common to
every projection and to EE and BB alike, and it is why the measurements sit a
few percent under the curves, growing with l.

Run:  systemd-run --user --scope -p MemoryMax=12G bash env.sh python3 analysis/plot_projection_spectra.py
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
# The production bandpowers are already dl = 50; nothing is rebinned here.
ELL_MIN, ELL_MAX = 300.0, 4800.0
BAND_LIMIT = S.LMAX     # 6000: the run's own harmonic band limit

COLORS = PS.PROJ_COLORS
MARKS = PS.PROJ_MARKERS

MS, MEW = 3.0, 0.9


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
    bb, ee = SL.predicted_spectra(mom, i, j, ell, ell_theory, cl_theory,
                                  band_limit=BAND_LIMIT)
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

    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    dl = ell * (ell + 1) / (2 * np.pi)

    m_bb = np.empty((len(S.PROJS), len(ell)))
    m_ee = np.empty_like(m_bb)
    for j in range(len(S.PROJS)):
        m_bb[j], m_ee[j] = model_curves(mom, i, j, ell, ell_theory, cl_theory)

    return dict(i=i, mom=mom, ee=ee, bb=bb, ell=ell, cl_ee=cl_ee,
                cl_bb_lensed=cl_bb_lensed, transfer=transfer,
                m_bb=m_bb, m_ee=m_ee, keep=keep, dl=dl)


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
    i, ee, transfer = d["i"], d["ee"], d["transfer"]
    cl_ee, m_ee = d["cl_ee"], d["m_ee"]
    ell, keep, dl = d["ell"], d["keep"], d["dl"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.25]})

    # ---- (a) EE bandpowers -------------------------------------------------
    ax = axes[0]
    input_line, = ax.plot(ell[keep], (cl_ee * dl)[keep], "-",
                          color=PS.INK, lw=1.4, zorder=1, label="input $EE$")
    proj_handles = []
    for j, proj in enumerate(S.PROJS):
        y = (ee[i, j].mean(0) / transfer) * dl
        h, = ax.plot(ell[keep], y[keep], MARKS[proj], color=COLORS[proj],
                     ms=MS, mfc="none", mew=MEW, label=S.PROJ_LABEL[proj])
        proj_handles.append(h)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{EE}/2\pi$  [$\mu$K$^2$]")

    # ---- (b) EE ratio to input, with the remap prediction ------------------
    ax = axes[1]
    ax.axhline(1.0, color=PS.RULE, lw=0.8, zorder=1)
    for j, proj in enumerate(S.PROJS):
        r = ee[i, j].mean(0) / transfer / cl_ee
        ax.plot(ell[keep], r[keep], MARKS[proj], color=COLORS[proj], ms=MS,
                mfc="none", mew=MEW)
        ax.plot(ell[keep], (m_ee[j] / cl_ee)[keep], "-", color=COLORS[proj],
                lw=1.3, alpha=0.9)
    ax.set_ylim(0.25, 2.6)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{EE}$ / input $C_\ell^{EE}$")

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
    """BB bandpowers (top) and the BB/EE leakage relation (bottom).

    STG is drawn in the top panel, where its points are the pipeline floor and
    the reader can see how far below everything else that floor sits, and
    omitted from the ratio panel, where a floor divided by EE would read as a
    leakage. Its value is quoted in the text instead.
    """
    i, bb, ee, transfer = d["i"], d["bb"], d["ee"], d["transfer"]
    cl_bb_lensed, cl_ee = d["cl_bb_lensed"], d["cl_ee"]
    m_bb, m_ee = d["m_bb"], d["m_ee"]
    ell, keep, dl = d["ell"], d["keep"], d["dl"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    # ---- (a) BB bandpowers -------------------------------------------------
    ax = axes[0]
    ref_line, = ax.plot(ell[keep], (cl_bb_lensed * dl)[keep], "--",
                        color=PS.INK, lw=1.2, zorder=1,
                        label="lensed")
    proj_handles = []
    for j, proj in enumerate(S.PROJS):
        y = (bb[i, j].mean(0) / transfer) * dl
        h, = ax.plot(ell[keep], y[keep], MARKS[proj], color=COLORS[proj],
                     ms=MS, mfc="none", mew=MEW, label=S.PROJ_LABEL[proj])
        proj_handles.append(h)
        if np.nanmax(m_bb[j]) > 0:
            ax.plot(ell[keep], (m_bb[j] * dl)[keep], "-", color=COLORS[proj],
                    lw=1.3, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 3e1)
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{BB}/2\pi$  [$\mu$K$^2$]")

    # ---- (b) BB / EE, the leakage relation ---------------------------------
    ax = axes[1]
    ax.plot(ell[keep], (cl_bb_lensed / cl_ee)[keep], "--",
                            color=PS.INK, lw=1.2, zorder=1)
    drawn = [(cl_bb_lensed / cl_ee)[keep]]
    for j, proj in enumerate(S.PROJS):
        if proj == SL.FLOOR_PROJ:
            continue
        r = bb[i, j].mean(0) / ee[i, j].mean(0)
        ax.plot(ell[keep], r[keep], MARKS[proj], color=COLORS[proj], ms=MS,
                mfc="none", mew=MEW)
        ax.plot(ell[keep], (m_bb[j] / m_ee[j])[keep], "-", color=COLORS[proj],
                lw=1.3, alpha=0.9)
        drawn += [r[keep], (m_bb[j] / m_ee[j])[keep]]
    ax.set_yscale("log")
    # Span what is actually drawn -- the four sheared projections, their
    # models, and the lensed reference -- with a small margin, rather than a
    # fixed range with two empty decades at the bottom.
    lo = np.nanmin([np.nanmin(y) for y in drawn])
    hi = np.nanmax([np.nanmax(y) for y in drawn])
    ax.set_ylim(lo / 1.6, hi * 1.6)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{BB} / C_\ell^{EE}$")

    axes[0].set_xlim(ELL_MIN, ELL_MAX)

    # One dashed entry for both panels: which lensed reference is drawn is
    # settled by the axis the reader is looking at.
    handles = proj_handles + [ref_line]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS] + ["lensed"]
    fig.legend(handles, labels, fontsize=8.5, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.3,
               handletextpad=0.5, frameon=False)

    fig.tight_layout(pad=1.0, h_pad=0.6, rect=(0, 0, 1, 0.925))
    FIGURE_BB.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE_BB)
    print(f"wrote {FIGURE_BB}")


def print_numbers(d):
    """A few numbers for the caption and the prose, printed rather than typed."""
    i, ee, bb, transfer = d["i"], d["ee"], d["bb"], d["transfer"]
    ell, cl_ee = d["ell"], d["cl_ee"]
    m_bb, m_ee = d["m_bb"], d["m_ee"]

    print("\n5% disk, measured / input EE (measured / model in parentheses):")
    for j, proj in enumerate(S.PROJS):
        r = ee[i, j].mean(0) / transfer / cl_ee
        row = []
        for L in (1000, 2000, 3000, 4000, 4500):
            k = np.argmin(abs(ell - L))
            row.append(f"{L}: {r[k]:.3f} ({r[k] / (m_ee[j][k] / cl_ee[k]):.3f})")
        print(f"  {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row))
    sel = (ell > 500) & (ell < 3000)
    print("\nBB at 5% disk, 500 < l < 3000 mean [uK^2]:")
    for j, proj in enumerate(S.PROJS):
        meas = (bb[i, j].mean(0) / transfer)[sel].mean()
        print(f"  {S.PROJ_LABEL[proj]:<6s} measured {meas:9.3e}   "
              f"model {m_bb[j][sel].mean():9.3e}   "
              f"<g^2> {d['mom']['g2'][i, j]:9.3e}")
    wide = (ell > 500) & (ell < 4800)
    print("\nBB/EE over 500 < l < 4800, measured / model:")
    for j, proj in enumerate(S.PROJS):
        if proj == SL.FLOOR_PROJ:
            continue
        r = (bb[i, j].mean(0) / ee[i, j].mean(0))[wide] / (m_bb[j] / m_ee[j])[wide]
        print(f"  {S.PROJ_LABEL[proj]:<6s} {r.mean():.4f} +- {r.std():.4f}")


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
