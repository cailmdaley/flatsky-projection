"""The spectra on the 5% disk, projection by projection, against the
parameter-free forward model.

This is the paper's central figure: the distortion where it lives, in the
bandpowers. One mask, four projections, three spectra, one column each, with
the spectrum itself above and its ratio below on the same multipole axis.

  top     y = l^6 C_l, TT, EE, BB, on linear axes. l^6 puts the peak near
          l = 2000 and keeps the damping tail well above the axis out to
          l = 4800, so the acoustic peaks can be seen to wash out under SFL
          and BICEP. The black line is the sims' input, the binned unlensed
          CAMB spectrum; the input BB is zero because the sims are unlensed.
          ZEA's BB is two orders of magnitude below SFL's and BICEP's, so it
          is drawn scaled (`SCALE`) and a legend in the panel says so.
  bottom  TT / input, EE / input, BB / EE. The two scalar-like ratios share a
          y axis, because they share a shape: both track the local slope of
          C(l) and rise above l = 3000. BB/EE is pure leakage, since the true
          B is zero, and is flat at 2<g^2> because the misalignment kernel has
          no slope dependence. The lensed BB/EE of the real sky is drawn for
          scale.

STG is left out of the BB column: with zero shear its BB is the pipeline's own
E-to-B floor, not a leakage, and it is quoted in the text instead.

The curves are the forward model of `shear_leakage.coupling_matrix`: the sky
spectrum read at the sheared wavenumber |J^T l|, with the exact
sin^2/cos^2(2 dpsi + 2 Delta) kernel for E and B and kernel 1 for T, Delta the
residual of the pipeline's own flattening angle, averaged over mode azimuth
and over the footprint with weight W^2 |det J|^2 per unit sky area, and
band-limited at the sims' l = 6000. No free parameters, no fit.

Normalisation. The pseudo-C_l denominator is the plane-area mask integral,
sum(W^2 |det J|) per unit sky area, one power of |det J| lower than the
numerator's. Getting this right matters only for a projection whose
magnification varies across the footprint: for BICEP the wrong power moves the
curve by 20%, for the two equal-area projections not at all.

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

Points are the mean over the 20 realizations. Four projections in a 2 inch
column leave no room for all 90 bandpowers, so each projection shows every
`STRIDE`-th one, offset from the others. An error bar, the standard error of
the mean, is drawn only where it outreaches its marker.

TT points come from `spectra.load_scalar`. `measured_tt` returns None until
the `scalar_tt_bandpowers` output exists, and until then the TT column draws
curves and no markers.

Run:  ~/code/python/spt3g_software/build/env-shell.sh python analysis/plot_projection_spectra.py
Out:  figures/projection_spectra.pdf
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
FIGURE = ROOT / "figures" / "projection_spectra.pdf"
SCALAR_TT_DIR = ROOT / "results" / "highell" / "scalar_tt_bandpowers"

MASK_INDEX = 5          # PERC[5] = 5.0%, the paper's reference footprint
ELL_MIN, ELL_MAX = 300.0, 4800.0
BAND_LIMIT = S.LMAX     # 6000: the run's own harmonic band limit

SPECTRA = ("TT", "EE", "BB")
POWER = 6
INPUT_LW = 0.6          # thin and on top, because ZEA lies along the input
SCALE = {("BB", "ProjZEA"): 100}
STRIDE = 4
FIG_H = 4.0


def measured_tt():
    """Measured TT on the reference disk, (n_projs, NSIMS, NBINS), or None."""
    if not SCALAR_TT_DIR.is_dir():
        return None
    return S.load_scalar("TT")[MASK_INDEX]


def load_data():
    """Bandpowers, inputs and model curves, per spectrum, in absolute C_l."""
    i = MASK_INDEX
    mom = SL.load_moments()
    th = T.load()
    ell_th = th["ell"].astype(float)
    ell = S.ELL

    model = {s: np.empty((len(S.PROJS), len(ell))) for s in SPECTRA}
    for j in range(len(S.PROJS)):
        R = SL.coupling_matrix(mom, i, j, ell, ell_th, band_limit=BAND_LIMIT)
        model["BB"][j] = R["BB<-EE"] @ th["unlensed_EE"]
        model["EE"][j] = R["EE<-EE"] @ th["unlensed_EE"]
        model["TT"][j] = R["TT<-TT"] @ th["unlensed_TT"]

    return dict(
        i=i, mom=mom, ell=ell, model=model,
        measured={"TT": measured_tt(), "EE": S.load("EE")[i],
                  "BB": S.load("BB")[i]},
        input={"TT": S.theory("unlensed_TT"), "EE": S.theory("unlensed_EE"),
               "BB": np.zeros_like(ell)},
        lensed_bb=S.theory("lensed_BB"),
    )


def sem(x):
    """Standard error of the mean over the seed axis."""
    return np.nanstd(x, axis=0, ddof=1) / np.sqrt(x.shape[0])


def add_visible_errors(series):
    """Error bars where they outreach their marker, on axes already sized.

    Marker size is in points, so which bars survive depends on the final
    geometry, and on a log axis on the height of the point as well.
    """
    for ax, x, y, err, proj in series:
        ax.autoscale(False)
        to_pt = 72 / ax.figure.dpi
        ends = [ax.transData.transform(np.c_[x, y + s * err])[:, 1]
                for s in (-1, 1)]
        reach = 0.5 * (ends[1] - ends[0]) * to_pt
        ax.errorbar(x, y, yerr=np.where(reach > PS.MS / 2, err, np.nan),
                    fmt="none", ecolor=PS.PROJ_COLORS[proj],
                    elinewidth=PS.MEW, capsize=0, zorder=2.9)


def plot(plt, d):
    """TT, EE, BB side by side: the spectrum above, its ratio below."""
    from matplotlib.lines import Line2D

    ell = d["ell"]
    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    x = ell[keep]

    fig, axes = plt.subplots(2, len(SPECTRA), figsize=(PS.TEXT_W, FIG_H),
                             sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    axes[1, 1].sharey(axes[1, 0])

    series = []                 # (ax, x, y, err, proj), for the error bars

    def curve_and_points(ax, j, model, seeds, scale=1.0):
        """One projection: its model curve, and its points if it has any."""
        proj = S.PROJS[j]
        ax.plot(x, scale * model[keep], "-", lw=PS.LW,
                color=PS.PROJ_COLORS[proj], zorder=2)
        if seeds is None:
            return
        every = slice(j, None, STRIDE)
        xs = x[every]
        y = (scale * seeds.mean(0)[keep])[every]
        ax.plot(xs, y, PS.PROJ_MARKERS[proj], color=PS.PROJ_COLORS[proj],
                mfc="white", ms=PS.MS, mew=PS.MEW, zorder=3)
        series.append((ax, xs, y, (scale * sem(seeds)[keep])[every], proj))

    for col, spec in enumerate(SPECTRA):
        top, bottom = axes[:, col]
        measured = d["measured"][spec]
        cl_in = d["input"][spec]
        weight = ell ** POWER
        exponent = int(np.floor(np.log10((weight * d["model"][spec])[:, keep].max())))
        unit = 10.0 ** exponent

        top.plot(x, (weight * cl_in / unit)[keep], color="k", lw=INPUT_LW,
                 zorder=2.5)
        for j, proj in enumerate(S.PROJS):
            if spec == "BB" and proj == SL.FLOOR_PROJ:
                continue
            curve_and_points(top, j, weight * d["model"][spec][j],
                             None if measured is None else weight * measured[j],
                             SCALE.get((spec, proj), 1) / unit)
        top.set_ylabel(rf"$\ell^{{{POWER}}}\, C_\ell^{{{spec}}}$ "
                       rf"[$10^{{{exponent}}}\ \mu{{\rm K}}^2$]")
        top.margins(y=0.04)

        if spec == "BB":
            bottom.plot(x, (d["lensed_bb"] / d["input"]["EE"])[keep], "--",
                        color=PS.SECONDARY[1], lw=PS.LW_REF, zorder=1)
            for j, proj in enumerate(S.PROJS):
                if proj == SL.FLOOR_PROJ:
                    continue
                curve_and_points(bottom, j,
                                 d["model"]["BB"][j] / d["model"]["EE"][j],
                                 d["measured"]["BB"][j] / d["measured"]["EE"][j])
            bottom.set_ylabel(r"$C_\ell^{BB} / C_\ell^{EE}$")
        else:
            bottom.axhline(1.0, color="k", lw=INPUT_LW, zorder=2.5)
            for j in range(len(S.PROJS)):
                curve_and_points(bottom, j, d["model"][spec][j] / cl_in,
                                 None if measured is None else measured[j] / cl_in)
            bottom.set_ylabel(rf"$C_\ell^{{{spec}}} / "
                              rf"C_\ell^{{{spec},\,\mathrm{{in}}}}$")
        bottom.set_yscale("log")
        bottom.set_xlabel(r"$\ell$")

    # The TT and EE ratios share this axis, so it is set once, after both.
    ratio = axes[1, 0].yaxis
    ratio.set_major_locator(plt.FixedLocator([0.7, 1, 1.5, 2, 3]))
    ratio.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
    ratio.set_minor_formatter(plt.NullFormatter())

    axes[0, 0].set_xlim(ELL_MIN, ELL_MAX)
    axes[0, 0].set_xticks([1000, 2000, 3000, 4000])
    fig.align_ylabels(axes)

    handles = [Line2D([], [], ls="none", marker=PS.PROJ_MARKERS[p],
                      color=PS.PROJ_COLORS[p], mfc="white", ms=PS.MS + 0.6,
                      mew=PS.MEW) for p in S.PROJS]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS]
    handles += [Line2D([], [], color="k", lw=INPUT_LW),
                Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW),
                Line2D([], [], ls="none", marker="o", color=PS.SECONDARY[1],
                       mfc="white", ms=PS.MS + 0.6, mew=PS.MEW),
                Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW_REF, ls="--")]
    labels += ["input", "forward model", "simulations, mean of 20",
               r"lensed $BB/EE$"]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")

    bb = axes[0, SPECTRA.index("BB")]
    scaled = [(proj, s) for (spec, proj), s in SCALE.items() if spec == "BB"]
    bb.legend([Line2D([], [], color=PS.PROJ_COLORS[proj],
                      marker=PS.PROJ_MARKERS[proj], mfc="white",
                      ms=PS.MS + 0.6, mew=PS.MEW) for proj, _ in scaled],
              [rf"{S.PROJ_LABEL[proj]} $\times\,{s}$" for proj, s in scaled],
              loc="upper right")

    fig.canvas.draw()
    add_visible_errors(series)

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


def print_numbers(d):
    """A few numbers for the caption and the prose, printed rather than typed."""
    i, ell = d["i"], d["ell"]
    ee, bb = d["measured"]["EE"], d["measured"]["BB"]
    m_ee, m_bb = d["model"]["EE"], d["model"]["BB"]
    cl_ee = d["input"]["EE"]

    print("\n5% disk, measured / input EE (measured / model in parentheses):")
    for j, proj in enumerate(S.PROJS):
        r = ee[j].mean(0) / cl_ee
        row = []
        for L in (1000, 2000, 3000, 4000, 4500):
            k = np.argmin(abs(ell - L))
            row.append(f"{L}: {r[k]:.3f} ({r[k] / (m_ee[j][k] / cl_ee[k]):.3f})")
        print(f"  {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row))
    sel = (ell > 500) & (ell < 3000)
    print("\nBB at 5% disk, 500 < l < 3000 mean [uK^2]:")
    for j, proj in enumerate(S.PROJS):
        print(f"  {S.PROJ_LABEL[proj]:<6s} measured {bb[j].mean(0)[sel].mean():9.3e}   "
              f"model {m_bb[j][sel].mean():9.3e}   "
              f"<g^2> {d['mom']['g2'][i, j]:9.3e}")
    wide = (ell > 500) & (ell < 4800)
    print("\nBB/EE over 500 < l < 4800, measured / model:")
    for j, proj in enumerate(S.PROJS):
        if proj == SL.FLOOR_PROJ:
            continue
        r = (bb[j].mean(0) / ee[j].mean(0))[wide] / (m_bb[j] / m_ee[j])[wide]
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
