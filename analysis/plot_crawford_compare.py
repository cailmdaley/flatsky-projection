"""Crawford's proj0/proj5 ratio: what geometry explains, and what his estimator adds.

The figure this makes
---------------------
Six cells, TT above EE, one column per mask. Each cell is a ratio panel over a
strip showing the mode-coupling bias of the two masks that went into it.

Read it column by column.

  full mask     Projection geometry alone lands on his measurement: 1.026 for
                TT and 0.997 for EE at l = 4050. The coupling correction is
                sub-percent, and the bias strip is flat at one. Nothing else is
                needed, and nothing else is claimed.

  cut corner    Geometry alone over-predicts by 19% (TT). His mask surgery
  iso shrink    zeroes pixels outright, destroying the apodization along the
                new boundary, and `cl_flatsky` corrects only by <mask^2> --
                no mode decoupling at all. Folding in the exact coupling of
                his own masks closes both variants: 0.996 and 0.976 for TT,
                0.988 and 1.010 for EE.

So the apparent failure of the model on the variants is not a failure of the
projection model. It is the estimator, and it is calculable in closed form
from the masks themselves.

Why the bias strip earns its space
----------------------------------
The strip shows the multiplicative error the estimator makes on each spectrum
separately: Ctilde/C for the proj0 mask and for the proj5 mask. By l = 5000
those reach 2 (cut corner) and 8 (iso shrink) for TT. Neither spectrum is
usable on its own. The ratio survives only because the two errors are similar
and largely divide out -- the residual, bias0/bias5, is exactly the factor
between the dashed and solid curves above.

The caveat that travels with the two right-hand columns
-------------------------------------------------------
`mask_coupling` evaluates the isotropic part of the coupling, which it argues
is safe when the correction is small. On the full mask that argument holds
completely. On iso_shrink it does not: a correction of a factor of eight has
an anisotropic partner that is not negligible, and the strip is what shows
this. The variant columns should be read as "mode coupling of this size is
present and is the right explanation", not as a percent-level prediction.

Input is the JSON written by `run_mask_coupling.py`; this script computes
nothing and will refuse to run if that file is missing.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawford as C          # noqa: E402
import crawford_masks as CM   # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
COUPLING = CM.CACHE / "mask_coupling.json"

SPECS = ("tt", "ee")
COLUMNS = (("full", "full mask"),
           ("cutcorner", "cut corner (hard edge)"),
           ("iso_shrink", "iso shrink (hard edge)"))
L_QUOTE = 4050.0                      # the multipole the annotations quote

DATA = "#2E2A26"      # measured bandpowers -- the anchor, kept neutral
GEOM = "#5A8B8B"      # geometry only
FULL = "#A05638"      # geometry + mask coupling
BIAS0 = "#C08B5C"     # estimator bias of the proj0 mask
BIAS5 = "#5B4B6B"     # estimator bias of the proj5 mask
RULE = "0.55"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "figure.autolayout": False,
})


def load():
    """The cached coupling results, keyed '{spec}/{variant}'."""
    if not COUPLING.exists():
        raise SystemExit(
            f"missing {COUPLING}\nrun `python analysis/run_mask_coupling.py` "
            "(about 10 minutes) before plotting.")
    d = json.loads(COUPLING.read_text())
    want = {f"{s}/{v}" for s in SPECS for v, _ in COLUMNS}
    missing = want - set(d)
    if missing:
        raise SystemExit(f"{COUPLING} is missing keys: {sorted(missing)}")
    for key in sorted(want):
        for field in ("measured", "uncoupled", "coupled", "bias0", "bias5"):
            arr = np.asarray(d[key][field], dtype=float)
            if arr.shape != (C.N_BINS,):
                raise SystemExit(f"{COUPLING}: {key}/{field} has shape "
                                 f"{arr.shape}, expected ({C.N_BINS},)")
            d[key][field] = arr
    return d


def ratio_panel(ax, r):
    """Measured, geometry-only and geometry+coupling on one set of bandpowers."""
    ax.axhline(1.0, color=RULE, lw=0.7, zorder=0)
    # The shaded wedge is the whole argument of the figure: it is what mode
    # coupling moves. It has no width in the full column and opens up in the
    # other two.
    ax.fill_between(C.ELL, r["uncoupled"], r["coupled"], color=FULL,
                    alpha=0.16, lw=0, zorder=1)
    ax.plot(C.ELL, r["coupled"], "-", color=FULL, lw=1.9, zorder=3,
            label=r"geometry $+$ mask coupling")
    ax.plot(C.ELL, r["uncoupled"], "--", color=GEOM, lw=1.7, dashes=(5, 2.6),
            zorder=4, label="geometry only")
    ax.plot(C.ELL, r["measured"], "-", color=DATA, lw=0.7, alpha=0.35, zorder=5)
    ax.plot(C.ELL, r["measured"], "o", color=DATA, ms=2.9, mew=0, zorder=6,
            label="Crawford 2026, measured")


def bias_panel(ax, r):
    """The estimator's own multiplicative error, mask by mask."""
    ax.axhline(1.0, color=RULE, lw=0.7, zorder=0)
    ax.plot(C.ELL, r["bias0"], "-", color=BIAS0, lw=1.4, label="proj0 mask")
    ax.plot(C.ELL, r["bias5"], "-", color=BIAS5, lw=1.4, label="proj5 mask")
    ax.set_yscale("log")


def annotate(ax, r):
    """measured/model at one multipole, coloured to key onto the two curves."""
    i = int(L_QUOTE // C.DELTA_L)
    geom = r["measured"][i] / r["uncoupled"][i]
    full = r["measured"][i] / r["coupled"][i]
    ax.text(0.035, 0.960, rf"measured\,/\,model at $\ell = {L_QUOTE:.0f}$",
            transform=ax.transAxes, va="top", fontsize=8, color="0.35")
    for dy, name, val, colour in ((0.865, "geometry only", geom, GEOM),
                                  (0.775, r"$+$ mask coupling", full, FULL)):
        ax.text(0.075, dy, name, transform=ax.transAxes, va="top",
                fontsize=8.5, color=colour)
        ax.text(0.395, dy, f"${val:.3f}$", transform=ax.transAxes, va="top",
                ha="right", fontsize=8.5, color=colour)


def main():
    d = load()

    fig = plt.figure(figsize=(13.6, 8.9))
    blocks = {
        "tt": fig.add_gridspec(2, 3, height_ratios=[3.0, 1.15], top=0.893,
                               bottom=0.505, hspace=0.07, wspace=0.075,
                               left=0.062, right=0.988),
        "ee": fig.add_gridspec(2, 3, height_ratios=[3.0, 1.15], top=0.452,
                               bottom=0.064, hspace=0.07, wspace=0.075,
                               left=0.062, right=0.988),
    }

    axes = {}
    for spec in SPECS:
        gs = blocks[spec]
        main_ref = bias_ref = None
        for j, (variant, _) in enumerate(COLUMNS):
            a = fig.add_subplot(gs[0, j], sharey=main_ref)
            b = fig.add_subplot(gs[1, j], sharey=bias_ref, sharex=a)
            main_ref = main_ref or a
            bias_ref = bias_ref or b
            axes[spec, variant] = (a, b)

    for spec in SPECS:
        for j, (variant, title) in enumerate(COLUMNS):
            a, b = axes[spec, variant]
            r = d[f"{spec}/{variant}"]
            ratio_panel(a, r)
            bias_panel(b, r)
            annotate(a, r)
            a.tick_params(labelbottom=False)
            a.set_xlim(0, 6000)
            b.set_xticks([0, 2000, 4000, 6000])
            if spec == "tt":
                a.set_title(title, pad=7)
            if spec == "ee":
                b.set_xlabel(r"$\ell$", labelpad=1)
            if j:
                a.tick_params(labelleft=False)
                b.tick_params(labelleft=False)

    # Row identity lives in the y-axis label, so the two blocks never need a
    # separate row heading.
    for spec in SPECS:
        a, b = axes[spec, "full"]
        up = spec.upper()
        a.set_ylabel(rf"$C_\ell^{{{up}}}(\mathrm{{proj0}})\,/\,"
                     rf"C_\ell^{{{up}}}(\mathrm{{proj5}})$")
        b.set_ylabel(r"$\tilde{C}_\ell/C_\ell$", labelpad=2)
        b.set_ylim(0.85, 16.0)
        b.set_yticks([1, 3, 10])
        b.set_yticklabels(["1", "3", "10"])
        b.minorticks_off()
        lo = min(np.nanmin(d[f"{spec}/{v}"][k]) for v, _ in COLUMNS
                 for k in ("measured", "uncoupled", "coupled"))
        hi = max(np.nanmax(d[f"{spec}/{v}"][k]) for v, _ in COLUMNS
                 for k in ("measured", "uncoupled", "coupled"))
        a.set_ylim(lo - 0.06 * (hi - lo), hi + 0.13 * (hi - lo))

    # Two legends, each parked where its own block has empty space.
    a, b = axes["tt", "full"]
    handles, labels = a.get_legend_handles_labels()
    order = [2, 1, 0]
    a.legend([handles[i] for i in order], [labels[i] for i in order],
             loc="lower right", frameon=False, handlelength=2.2,
             borderaxespad=0.6, labelspacing=0.35)
    b.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2,
             handlelength=1.6, columnspacing=1.4, borderaxespad=0.3)

    fig.suptitle("Projection geometry predicts Crawford's full-mask "
                 "proj0/proj5 ratio; his hard-edged variants need the "
                 "estimator's mode coupling as well", y=0.972, fontsize=13.5)
    fig.savefig(OUT / "crawford_compare.png")
    print(f"wrote {OUT / 'crawford_compare.png'}")

    # ---- the numbers, for the record ---------------------------------------
    show = [10, 20, 30, 40, 50]
    for spec in SPECS:
        for variant, _ in COLUMNS:
            r = d[f"{spec}/{variant}"]
            print(f"\n{spec.upper()} {variant}")
            print("  l           " + "".join(f"{C.ELL[i]:8.0f}" for i in show))
            print("  measured    " + "".join(f"{r['measured'][i]:8.3f}" for i in show))
            print("  meas/geom   " + "".join(
                f"{r['measured'][i] / r['uncoupled'][i]:8.3f}" for i in show))
            print("  meas/+coup  " + "".join(
                f"{r['measured'][i] / r['coupled'][i]:8.3f}" for i in show))
            print("  bias0       " + "".join(f"{r['bias0'][i]:8.2f}" for i in show))
            print("  bias5       " + "".join(f"{r['bias5'][i]:8.2f}" for i in show))


if __name__ == "__main__":
    main()
