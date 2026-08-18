"""Publication figures: geometric prediction vs Crawford's measured proj0/proj5 ratio.

Three figures, all built from `crawford_model.predict()` / `crawford_model.measured()`
plus a direct disk-mask sweep of `geometry.py` -- nothing here fits a parameter to his
data. See `crawford_model.py` for the model and the felt fiber
`science/spt/pol-project-paper/tt-scalar-convexity` (and its `derivation.md`) for the
physics writeup this figure set illustrates.

Figure 1 (crawford_tt_ee_comparison.png): predicted vs measured ratio, full mask,
TT and EE panels side by side, with a residual sub-panel and a shaded "resampling
regime" band above l ~ 5400.

Figure 2 (crawford_mask_variants.png): the two hard-edge mask variants (cutcorner,
iso_shrink), where zeroing pixels outright breaks `cl_flatsky`'s <mask^2> correction
and the measurement falls below the pure-geometry prediction.

Figure 3 (crawford_mask_size_trend.png): predicted ratio at l = 3000, 4000, 5000 vs
disk radius (concentric apodized C2/2.5deg disks about the projection centre, SFL
relative to ZEA, weighted by W^2 -- the same weighting NaMaster's pseudo-Cl
estimator uses, and what `analysis/disk_trend.py` applies at the seven actual
paper radii), with Crawford's real footprint <g^2> = 0.0442 marked and his
measured l~4000 value overlaid.

Run inside the project container:
    systemd-run --user --scope -p MemoryMax=4G bash env.sh python3 analysis/plot_crawford_comparison.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src" / "stages"))
import crawford as C            # noqa: E402
import crawford_model as CMOD   # noqa: E402
import geometry as G            # noqa: E402
from make_masks import apodize_disk_analytic  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

RESAMPLING_L = 5400.0   # only ~1-2 clean bandpowers remain above this
DIP_L = 1450.0
PEAK_L = 4000.0

DATA = "#2E2A26"
MODEL = "#A05638"
RESID_LINE = "#5A8B8B"
BAND = "#C7924A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.7,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.autolayout": False,
    "text.usetex": False,
})


# --------------------------------------------------------------------------- #
# Figure 1: prediction vs measurement, full mask, TT & EE
# --------------------------------------------------------------------------- #

def make_figure1():
    specs = [("tt", "TT"), ("ee", "EE")]
    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.08,
                          wspace=0.14, left=0.08, right=0.985, top=0.90,
                          bottom=0.11)

    printed = {}
    for j, (spec, label) in enumerate(specs):
        ell_pred, ratio_pred = CMOD.predict("full", spec)
        ell_meas, ratio_meas = CMOD.measured("", spec)
        assert np.allclose(ell_pred, ell_meas)
        ell = ell_pred
        resid = ratio_meas / ratio_pred
        printed[spec] = (ell, ratio_meas, ratio_pred, resid)

        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)

        ax.axvspan(RESAMPLING_L, 6000, color=BAND, alpha=0.14, lw=0, zorder=0)
        axr.axvspan(RESAMPLING_L, 6000, color=BAND, alpha=0.14, lw=0, zorder=0)

        ax.plot(ell, ratio_pred, "-", color=MODEL, lw=2.0, zorder=3,
                label="geometric prediction")
        ax.plot(ell, ratio_meas, "o", color=DATA, ms=4.0, mew=0, zorder=4,
                label="Crawford 2026, measured")
        ax.plot(ell, ratio_meas, "-", color=DATA, lw=0.6, alpha=0.35, zorder=2)

        axr.axhline(1.0, color="0.55", lw=0.8, zorder=0)
        axr.plot(ell, resid, "-", color=RESID_LINE, lw=1.6)
        axr.plot(ell, resid, "o", color=RESID_LINE, ms=2.6, mew=0)

        for l0, text in ((DIP_L, "acoustic dip"), (PEAK_L, "peak")):
            ax.axvline(l0, color="0.6", lw=0.7, ls=":", zorder=1)
            i0 = int(l0 // C.DELTA_L)
            ax.annotate(text, xy=(l0, ratio_meas[i0]), xycoords="data",
                       xytext=(6, 10), textcoords="offset points",
                       fontsize=8, color="0.35", ha="left", va="bottom")

        if j == 0:
            ax.text(0.985, 0.045, "resampling\nregime", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#8a5a1e")
        ax.set_title(label, pad=8)
        ax.tick_params(labelbottom=False)
        ax.set_xlim(0, 6000)
        axr.set_xlabel(r"multipole $\ell$")
        axr.set_ylim(0.85, 1.30)
        if j == 0:
            ax.set_ylabel(r"$C_\ell(\mathrm{proj0})\,/\,C_\ell(\mathrm{proj5})$")
            axr.set_ylabel("meas /\npred")
        else:
            ax.tick_params(labelleft=False)
            axr.tick_params(labelleft=False)

    axes0 = fig.axes[0]
    axes0.legend(loc="upper left", frameon=False, handlelength=2.0,
                borderaxespad=0.4)

    fig.suptitle("Zero-parameter projection geometry predicts Crawford's "
                 "1500d proj0/proj5 ratio", y=0.975, fontsize=13.5)
    path = OUT / "crawford_tt_ee_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return printed


# --------------------------------------------------------------------------- #
# Figure 2: hard-edge mask variants
# --------------------------------------------------------------------------- #

def make_figure2():
    variants = [("cutcorner", "cut corner (hard edge)"),
                ("iso_shrink", "iso shrink (hard edge)")]
    spec = "tt"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.14,
                        wspace=0.08)

    for ax, (variant, title) in zip(axes, variants):
        ell, ratio_pred = CMOD.predict(variant, spec)
        _, ratio_meas = CMOD.measured(CMOD.VARIANT_FILE[variant], spec)
        ax.axhline(1.0, color="0.55", lw=0.8, zorder=0)
        ax.plot(ell, ratio_pred, "-", color=MODEL, lw=2.0, zorder=3,
               label="geometric prediction")
        ax.plot(ell, ratio_meas, "o", color=DATA, ms=4.0, mew=0, zorder=4,
               label="Crawford 2026, measured")
        ax.plot(ell, ratio_meas, "-", color=DATA, lw=0.6, alpha=0.35, zorder=2)
        ax.fill_between(ell, ratio_meas, ratio_pred, where=ratio_meas < ratio_pred,
                        color=MODEL, alpha=0.10, lw=0, zorder=1)
        ax.set_title(title, pad=8)
        ax.set_xlabel(r"multipole $\ell$")
        ax.set_xlim(0, 6000)

    axes[0].set_ylabel(r"$C_\ell^{TT}(\mathrm{proj0})\,/\,C_\ell^{TT}(\mathrm{proj5})$")
    axes[0].legend(loc="upper left", frameon=False, handlelength=2.0)
    fig.suptitle("Hard-edge mask surgery pulls the measurement below pure "
                "geometry: mode coupling, not shape", y=0.975, fontsize=13)
    path = OUT / "crawford_mask_variants.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Figure 3: mask-size trend
# --------------------------------------------------------------------------- #

CENTER_DELTA = -56.0
CENTER_ALPHA = 0.0
CRAWFORD_G2 = 0.0442


DISK_NSIDE = 2048
DISK_APOTYPE = "C2"
DISK_APOSIZE_DEG = 2.5


def disk_geometry(radius_rad):
    """W^2-weighted (s1, s2, w^2) samples over an apodized (C2, 2.5 deg taper)
    disk of `radius_rad` about the projection centre, SFL relative to ZEA --
    both at the same centre, so radius and shear are the only thing moving.

    This replaced a flat, uniform-weighted, hard-edged Euclidean-disk
    approximation (ngrid=260 grid in tangent-plane offset degrees) that
    disagreed with the derivation note's own table by 10-20% at matched
    radius: an unapodized disk average of g^2 is dominated by whichever rim
    pixels the grid quantization happens to keep, since g grows fastest right
    at the edge. Apodizing removes that sensitivity, and querying a true
    spherical cap (`hp.query_disc`) removes the flat-tangent-plane
    approximation error on top of it. See `analysis/disk_trend.py` for the
    canonical version of this calculation, at the seven actual paper radii.
    """
    center_vec = hp.ang2vec(CENTER_ALPHA, CENTER_DELTA, lonlat=True)
    w_full = apodize_disk_analytic(DISK_NSIDE, center_vec, radius_rad,
                                   DISK_APOTYPE, DISK_APOSIZE_DEG)
    pix = np.nonzero(w_full > 0)[0]
    w = w_full[pix].astype(float)
    lon, lat = hp.pix2ang(DISK_NSIDE, pix, lonlat=True)

    a = G.relative_jacobian(
        "ProjSFL", "ProjZEA", lon, lat,
        to_kw=dict(delta_center_deg=CENTER_DELTA),
        from_kw=dict(delta_center_deg=CENTER_DELTA))
    s1, s2 = G.singular_values(a)
    g = (s1 - s2) / (s1 + s2)
    w2 = w ** 2
    g2 = float(np.average(g ** 2, weights=w2))
    return s1, s2, w2, g2


def disk_prediction(radius_rad, ell_targets, ell_th, cl_th):
    s1, s2, w2, g2 = disk_geometry(radius_rad)
    ell = np.asarray(ell_targets, dtype=float)
    proj0 = G.predicted_spectrum(ell, s1, s2, w2, ell_th, cl_th,
                                 band_limit=CMOD.BAND_LIMIT, n_bins=48, n_psi=64)
    proj5 = np.interp(ell, ell_th, cl_th)
    return g2, proj0 / proj5


def make_figure3():
    ell_targets = np.array([3000.0, 4000.0, 5000.0])
    ell_th, cl_th = CMOD.theory_input("tt")

    radii_rad = np.linspace(0.10, 0.65, 15)
    g2s, ratios = [], []
    for r in radii_rad:
        g2, ratio = disk_prediction(r, ell_targets, ell_th, cl_th)
        g2s.append(g2)
        ratios.append(ratio)
    g2s = np.array(g2s)
    ratios = np.array(ratios)   # (n_radii, 3)

    # Crawford's own footprint: real mask geometry already cached by predict().
    s1_c, s2_c, w_c = CMOD.geometry_of("full")
    g_c = (s1_c - s2_c) / (s1_c + s2_c)
    g2_footprint = float(np.average(g_c ** 2, weights=w_c))

    _, meas_ratio_4000 = CMOD.measured("", "tt")
    meas_at_4000 = meas_ratio_4000[int(PEAK_L // C.DELTA_L)]

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    colors = ["#5A8B8B", "#A05638", "#6B5B95"]
    for k, l0 in enumerate(ell_targets):
        ax.plot(g2s, ratios[:, k], "o-", color=colors[k], ms=4.5, lw=1.8,
               label=rf"$\ell = {l0:.0f}$")

    ax.axvline(CRAWFORD_G2, color="0.35", lw=1.2, ls="--", zorder=1,
              label=rf"Crawford footprint, $\langle g^2\rangle={CRAWFORD_G2}$")
    ax.plot([g2_footprint], [meas_at_4000], "*", color="#C0392B", ms=16,
           mec="black", mew=0.5, zorder=5,
           label=rf"measured, $\ell\!\sim\!{PEAK_L:.0f}$")

    ax.set_xlabel(r"footprint shear variance $\langle g^2 \rangle$")
    ax.set_ylabel(r"predicted $C_\ell^{TT}(\mathrm{proj0})\,/\,C_\ell^{TT}(\mathrm{proj5})$")
    ax.set_title("Only the largest, most-sheared footprint reaches ratio "
                "$\\approx 2$", pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    path = OUT / "crawford_mask_size_trend.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return g2s, ratios, g2_footprint, meas_at_4000


def main():
    printed = make_figure1()
    make_figure2()
    g2s, ratios, g2_footprint, meas_at_4000 = make_figure3()

    print("\n--- key numbers ---")
    for spec in ("tt", "ee"):
        ell, meas, pred, resid = printed[spec]
        for l0 in (DIP_L, PEAK_L, RESAMPLING_L, 5950.0):
            i = int(l0 // C.DELTA_L)
            print(f"{spec.upper()} l={ell[i]:6.0f}  measured={meas[i]:6.3f}  "
                 f"predicted={pred[i]:6.3f}  meas/pred={resid[i]:6.3f}")
    print(f"\nCrawford full-mask footprint <g^2> = {g2_footprint:.4f}")
    print(f"measured TT ratio near l={PEAK_L:.0f}: {meas_at_4000:.3f}")
    print("\ndisk sweep (radius trend, W^2-weighted apodized disks):")
    for r_rad, g2, row in zip(np.linspace(0.10, 0.65, 15), g2s, ratios):
        print(f"  R={r_rad:.3f} rad  <g^2>={g2:.4f}  "
             f"ratio(3000,4000,5000) = {row[0]:.3f} {row[1]:.3f} {row[2]:.3f}")


if __name__ == "__main__":
    main()
