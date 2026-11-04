"""The spectra themselves on the 5% disk: TT, EE and BB, projection by projection.

Every other spectrum figure in the paper is a ratio. This one shows what the
ratios are ratios of: the bandpowers, the input they were made from, and the
forward model, on linear axes.

Each panel plots y = l^p C_l with an integer power p (`POWERS`), so that a
spectrum falling over several decades reads on a linear scale. p = 6 in all
three: it puts the peak near l = 2000 and keeps the damping tail, where the
projections separate by a factor of two, well above the axis out to l = 4800.

Data
----
  points   Mean over the 20 realizations of `spectra.load` (QU leg), 5% disk.
           Four projections on a 2 inch axis leave no room for all 90
           bandpowers, so each projection shows every `STRIDE`-th one, offset
           from the others.
  black    The sims' input, the binned unlensed CAMB spectrum. The input BB is
           zero: the sims are unlensed.
  curves   The parameter-free forward model: `shear_leakage.predicted_spectra`
           for EE and BB, and the TT<-TT block of `coupling_matrix` for TT, both
           band-limited at the run's l = 6000.

STG is left out of BB, as in `plot_projection_spectra`: with zero shear its BB
is the pipeline's own floor, not a leakage. ZEA's BB is two orders of magnitude
below SFL's and BICEP's, so it is drawn scaled (`SCALE`) and a legend in the
panel says so.

TT has no measurement yet. `measured_tt` is the one place the panel reads one;
it returns None until the production files carry a `tt` array, and until then
the panel draws curves and no markers.

Run:  ~/code/python/spt3g_software/build/env-shell.sh python analysis/plot_direct_spectra.py
Out:  figures/direct_spectra.pdf
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
FIGURE = ROOT / "figures" / "direct_spectra.pdf"

MASK_INDEX = 5          # PERC[5] = 5.0%, the paper's reference footprint
ELL_MIN, ELL_MAX = 300.0, 4800.0
BAND_LIMIT = S.LMAX     # 6000: the run's own harmonic band limit

PANELS = ("TT", "EE", "BB")
POWERS = {"TT": 6, "EE": 6, "BB": 6}
INPUT_LW = 0.6          # thin and on top, because ZEA lies along the input
SCALE = {("BB", "ProjZEA"): 100}
STRIDE = 4
FIG_H = 2.3


def measured_tt():
    """Measured TT on the reference disk, (n_projs, NSIMS, NBINS), or None.

    None while the production bandpower files carry no `tt` array. The array
    is read by `spectra.load`, indexed as `ee` and `bb` are.
    """
    with np.load(S._path(S.PROJS[0], 0)) as z:
        if "tt" not in z.files:
            return None
    return S.load("TT")[MASK_INDEX]


def load_data():
    """Bandpowers, inputs and model curves, per panel, in absolute C_l."""
    i = MASK_INDEX
    mom = SL.load_moments()
    th = T.load()
    ell_th = th["ell"].astype(float)
    ell = S.ELL

    model = {s: np.empty((len(S.PROJS), len(ell))) for s in PANELS}
    for j in range(len(S.PROJS)):
        model["BB"][j], model["EE"][j] = SL.predicted_spectra(
            mom, i, j, ell, ell_th, th["unlensed_EE"], band_limit=BAND_LIMIT)
        model["TT"][j] = SL.coupling_matrix(
            mom, i, j, ell, ell_th, band_limit=BAND_LIMIT
        )["TT<-TT"] @ th["unlensed_TT"]

    return dict(
        ell=ell, model=model,
        measured={"TT": measured_tt(), "EE": S.load("EE")[i],
                  "BB": S.load("BB")[i]},
        input={"TT": S.theory("unlensed_TT"), "EE": S.theory("unlensed_EE"),
               "BB": np.zeros_like(ell)},
    )


def sem(x):
    """Standard error of the mean over the seed axis."""
    return x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def visible_error(ax, err):
    """`err` where its bar outreaches a marker, NaN where the marker hides it."""
    lo, hi = ax.get_ylim()
    pt_per_unit = ax.get_window_extent().height * 72 / ax.figure.dpi / (hi - lo)
    return np.where(err * pt_per_unit > PS.MS / 2, err, np.nan)


def plot(plt, d):
    """TT, EE, BB side by side: input, forward model, measured bandpowers."""
    from matplotlib.lines import Line2D

    ell = d["ell"]
    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    x = ell[keep]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(PS.TEXT_W, FIG_H),
                             sharex=True, layout="constrained")

    series = []                 # (ax, x, y, err, proj), drawn once axes are sized
    for ax, spec in zip(axes, PANELS):
        p = POWERS[spec]
        weight = x ** p
        measured = d["measured"][spec]
        top = (weight * d["model"][spec][:, keep]).max()
        exponent = int(np.floor(np.log10(top)))
        unit = 10.0 ** exponent

        ax.plot(x, weight * d["input"][spec][keep] / unit, color="k",
                lw=INPUT_LW, zorder=2.5)
        for j, proj in enumerate(S.PROJS):
            if spec == "BB" and proj == SL.FLOOR_PROJ:
                continue
            f = weight * SCALE.get((spec, proj), 1) / unit
            ax.plot(x, f * d["model"][spec][j][keep], "-", lw=PS.LW,
                    color=PS.PROJ_COLORS[proj], zorder=2)
            if measured is not None:
                m = measured[j][:, keep]
                every = slice(j, None, STRIDE)
                series.append((ax, x[every], (f * m.mean(0))[every],
                               (f * sem(m))[every], proj))

        ax.set_xlim(ELL_MIN, ELL_MAX)
        ax.set_xticks([1000, 2000, 3000, 4000])
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(rf"$\ell^{{{p}}}\, C_\ell^{{{spec}}}$ "
                      rf"[$10^{{{exponent}}}\ \mu{{\rm K}}^2$]")
        ax.margins(y=0.04)

    handles = [Line2D([], [], ls="none", marker=PS.PROJ_MARKERS[p],
                      color=PS.PROJ_COLORS[p], mfc="white", ms=PS.MS + 0.6,
                      mew=PS.MEW) for p in S.PROJS]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS]
    handles += [Line2D([], [], color="k", lw=INPUT_LW),
                Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW),
                Line2D([], [], ls="none", marker="o", color=PS.SECONDARY[1],
                       mfc="white", ms=PS.MS + 0.6, mew=PS.MEW)]
    labels += ["input", "forward model", "simulations, mean of 20"]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")

    bb = axes[PANELS.index("BB")]
    scaled = [(proj, s) for (spec, proj), s in SCALE.items() if spec == "BB"]
    bb.legend([Line2D([], [], color=PS.PROJ_COLORS[proj],
                      marker=PS.PROJ_MARKERS[proj], mfc="white",
                      ms=PS.MS + 0.6, mew=PS.MEW) for proj, _ in scaled],
              [rf"{S.PROJ_LABEL[proj]} $\times\,{s}$" for proj, s in scaled],
              loc="upper right")

    # Marker size is in points, so which error bars survive depends on the
    # final axes geometry.
    fig.canvas.draw()
    for ax, xs, y, err, proj in series:
        ax.errorbar(xs, y, yerr=visible_error(ax, err),
                    fmt=PS.PROJ_MARKERS[proj], color=PS.PROJ_COLORS[proj],
                    mfc="white", ms=PS.MS, mew=PS.MEW, elinewidth=PS.MEW,
                    capsize=0, zorder=3)

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PS.rc())
    plot(plt, load_data())


if __name__ == "__main__":
    main()
