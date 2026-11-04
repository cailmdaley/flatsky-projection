"""The measured spectra themselves, projection by projection, against the
parameter-free forward model.

This is the paper's central figure. Everything else in the analysis is a
reduction -- a band average, a ratio, a fit. Here the distortion is shown
where it actually lives: in the bandpowers.

One mask (the 5% disk), four projections, two channels, stacked on one
multipole axis so that the reader compares them bin for bin:

  top     EE divided by the input spectrum -- the scalar channel, the remap of
          Eq. (scalar_ratio) made visible. It oscillates with the acoustic
          structure because it tracks the local slope of C(l): SFL and BICEP
          lose a third of their power at l = 1000 and more than double it by
          l = 4500; STG, which has no shear at all, loses power monotonically
          to its magnification; ZEA, equal-area and least sheared, stays
          within a few percent of unity.
  bottom  BB/EE, the leakage channel, which is pure leakage: the sims are
          unlensed, so the true B is zero. It is flat at 2<g^2> because the
          misalignment kernel has no slope dependence. STG is left off: with
          zero shear its BB is the pipeline's own E-to-B floor, not a
          leakage, and it is quoted in the text instead. The lensed BB/EE of
          the real sky is drawn for scale.

The smooth curves are the same forward model, `shear_leakage.predicted_spectra`:
the sky spectrum read at the sheared wavenumber |J^T l| with the exact
sin^2/cos^2(2 dpsi + 2 Delta) kernel, Delta the residual of the pipeline's own
flattening angle, averaged over mode azimuth and over the footprint
with weight W^2 |det J|^2 per unit sky area, and band-limited at the sims'
l = 6000. No free parameters, no fit.

Normalisation. The pseudo-C_l denominator is the plane-area mask integral,
sum(W^2 |det J|) per unit sky area, one power of |det J| lower than the numerator's. Getting this right
matters only for a projection whose magnification varies across the
footprint -- for BICEP the wrong power moves the curve by 20%, for the two
equal-area projections not at all.

Range. The production run carries sky power to l = 6000, but the 2 arcmin
pixel has a Nyquist multipole of 5400 and the resampling residual arrives
before it: above l ~ 4800 every projection, STG included, rises onto one
common curve that is the pixel and not the geometry. The figure stops at
l = 4800 and makes no claim beyond it.

Pixel windows. The production bandpowers already have the 2 arcmin flat-sky
top-hat divided out (see `spectra`). What remains is the bilinear
interpolation transfer, which no analytic window captures; it is common to
every projection and to EE and BB alike, and it is why the measurements sit a
few percent under the curves, growing with l.

Error bars are the standard error of the 20-seed mean; they are drawn behind
filled markers, so where one is smaller than its marker it does not show.

Run:  OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=12G -p CPUQuota=400% bash env.sh python3 analysis/plot_projection_spectra.py
Out:  figures/projection_spectra.pdf
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
FIGURE = ROOT / "figures" / "projection_spectra.pdf"

MASK_INDEX = 5          # PERC[5] = 5.0%, the paper's reference footprint
# The production bandpowers are already dl = 50; nothing is rebinned here.
ELL_MIN, ELL_MAX = 300.0, 4800.0
BAND_LIMIT = S.LMAX     # 6000: the run's own harmonic band limit

COLORS = PS.PROJ_COLORS
MARKS = PS.PROJ_MARKERS

def model_curves(mom, i, j, ell, ell_theory, cl_theory):
    """Forward-modelled BB(l), EE(l) in absolute C_l."""
    return SL.predicted_spectra(mom, i, j, ell, ell_theory, cl_theory,
                                band_limit=BAND_LIMIT)


def load_data():
    """Everything the figure and the printed numbers need, loaded once."""
    i = MASK_INDEX
    mom = SL.load_moments()
    ee = S.load("EE")
    bb = S.load("BB")
    ell = S.ELL
    cl_ee = S.theory("unlensed_EE")
    cl_bb_lensed = S.theory("lensed_BB")
    th = T.load()
    ell_theory, cl_theory = th["ell"].astype(float), th["unlensed_EE"]

    keep = (ell > ELL_MIN) & (ell < ELL_MAX)

    m_bb = np.empty((len(S.PROJS), len(ell)))
    m_ee = np.empty_like(m_bb)
    for j in range(len(S.PROJS)):
        m_bb[j], m_ee[j] = model_curves(mom, i, j, ell, ell_theory, cl_theory)

    return dict(i=i, mom=mom, ee=ee, bb=bb, ell=ell, cl_ee=cl_ee,
                cl_bb_lensed=cl_bb_lensed,
                m_bb=m_bb, m_ee=m_ee, keep=keep)


def sem(x):
    """Standard error of the mean over the seed axis."""
    return np.nanstd(x, axis=0, ddof=1) / np.sqrt(x.shape[0])


def points(ax, x, y, err, proj):
    """One projection's bandpowers: filled markers over their error bars."""
    ax.errorbar(x, y, yerr=err, fmt=PS.PROJ_MARKERS[proj],
                color=PS.PROJ_COLORS[proj], mfc="white",
                ms=PS.MS, mew=PS.MEW, elinewidth=PS.MEW, capsize=0, zorder=3)


def plot(plt, d):
    """EE / input (top) and BB / EE (bottom) on one multipole axis."""
    from matplotlib.lines import Line2D

    i, ee, bb = d["i"], d["ee"], d["bb"]
    cl_ee, cl_bb_lensed = d["cl_ee"], d["cl_bb_lensed"]
    m_ee, m_bb = d["m_ee"], d["m_bb"]
    ell, keep = d["ell"], d["keep"]
    x = ell[keep]

    fig, axes = plt.subplots(2, 1, figsize=(PS.TEXT_W, 4.3), sharex=True,
                             layout="constrained",
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    # ---- scalar channel: EE / input --------------------------------------
    ax = axes[0]
    ax.axhline(1.0, color=PS.RULE, lw=PS.LW_REF, zorder=1)
    drawn = []
    for j, proj in enumerate(S.PROJS):
        r = ee[i, j] / cl_ee
        model = (m_ee[j] / cl_ee)[keep]
        ax.plot(x, model, "-", color=PS.PROJ_COLORS[proj], lw=PS.LW, zorder=2)
        points(ax, x, r.mean(0)[keep], sem(r)[keep], proj)
        drawn += [r.mean(0)[keep], model]
    ax.set_yscale("log")
    lo = min(np.nanmin(y) for y in drawn)
    hi = max(np.nanmax(y) for y in drawn)
    # Log, so a factor of two of loss and of gain sit equally far from unity;
    # the range always spans what is drawn.
    ax.set_ylim(min(lo / 1.08, 0.5), hi * 1.08)
    ax.yaxis.set_major_locator(plt.FixedLocator([0.5, 0.7, 1, 1.5, 2, 3]))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel(r"$C_\ell^{EE} / C_\ell^{EE,\,\mathrm{in}}$")

    # ---- leakage channel: BB / EE ----------------------------------------
    ax = axes[1]
    lensed = (cl_bb_lensed / cl_ee)[keep]
    ax.plot(x, lensed, "--", color=PS.SECONDARY[1], lw=PS.LW_REF, zorder=1)
    drawn = [lensed]
    for j, proj in enumerate(S.PROJS):
        if proj == SL.FLOOR_PROJ:
            continue
        r = bb[i, j] / ee[i, j]
        model = (m_bb[j] / m_ee[j])[keep]
        ax.plot(x, model, "-", color=PS.PROJ_COLORS[proj], lw=PS.LW, zorder=2)
        points(ax, x, r.mean(0)[keep], sem(r)[keep], proj)
        drawn += [r.mean(0)[keep], model]
    ax.set_yscale("log")
    lo = min(np.nanmin(y) for y in drawn)
    hi = max(np.nanmax(y) for y in drawn)
    ax.set_ylim(lo / 1.6, hi * 1.6)
    ax.set_ylabel(r"$C_\ell^{BB} / C_\ell^{EE}$")
    ax.set_xlabel(r"$\ell$")
    ax.set_xlim(ELL_MIN, ELL_MAX)
    fig.align_ylabels(axes)

    # One legend for both panels: projection identity, then what a point,
    # a curve and the dashed line are.
    handles = [Line2D([], [], ls="none", marker=PS.PROJ_MARKERS[p],
                      color=PS.PROJ_COLORS[p], mfc="white", ms=PS.MS + 0.6,
                      mew=PS.MEW) for p in S.PROJS]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS]
    handles += [Line2D([], [], ls="none", marker="o", color=PS.SECONDARY[1],
                       mfc="white", ms=PS.MS + 0.6, mew=PS.MEW),
                Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW),
                Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW_REF, ls="--")]
    labels += ["simulations, mean of 20", "forward model",
               r"lensed $BB/EE$"]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


def print_numbers(d):
    """A few numbers for the caption and the prose, printed rather than typed."""
    i, ee, bb = d["i"], d["ee"], d["bb"]
    ell, cl_ee = d["ell"], d["cl_ee"]
    m_bb, m_ee = d["m_bb"], d["m_ee"]

    print("\n5% disk, measured / input EE (measured / model in parentheses):")
    for j, proj in enumerate(S.PROJS):
        r = ee[i, j].mean(0) / cl_ee
        row = []
        for L in (1000, 2000, 3000, 4000, 4500):
            k = np.argmin(abs(ell - L))
            row.append(f"{L}: {r[k]:.3f} ({r[k] / (m_ee[j][k] / cl_ee[k]):.3f})")
        print(f"  {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row))
    sel = (ell > 500) & (ell < 3000)
    print("\nBB at 5% disk, 500 < l < 3000 mean [uK^2]:")
    for j, proj in enumerate(S.PROJS):
        meas = (bb[i, j].mean(0))[sel].mean()
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

    plt.rcParams.update(PS.rc())
    d = load_data()
    plot(plt, d)
    print_numbers(d)


if __name__ == "__main__":
    main()
