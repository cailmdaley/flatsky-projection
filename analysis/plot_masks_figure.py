"""The 2021 mask suite, as each projection sees it.

Seven concentric apodized disks (unapodized sky fractions 0.75% to 10%) were
projected onto the same 110 x 110 deg, 2 arcmin flat-sky grid in five
projections. The scientific point of the figure is geometric: a disk on the
sphere is not a disk on the grid, and how badly it is stretched depends on the
projection. SFL and the BICEP equidistant-cylindrical projection stretch in x
by ~1/cos(dec) at the field centre and are near-degenerate; SIN, STG and ZEA
are zenithal and keep circles as circles near the centre.

One row of five panels, one per projection:
  - grey image of the largest apodized mask (the C2, 2.5 deg taper shows as the
    soft edge),
  - W = 0.5 contours for all seven disks, coloured by disk area,
  - a dashed circle at the largest disk's true angular radius (37.2 deg) --
    what a projection with no distortion at all would draw.

Masks are read one at a time from the 2021 g3 files (3300^2 float64 each) and
reduced to contour paths, which are cached; re-running is cheap and idempotent.

Outputs: figures/masks_projections.{pdf,png}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
CACHE = Path(__file__).resolve().parent / ".cache" / "mask_contours.npz"
MASK_DIR = Path("/sptlocal/analysis/pol_project2020/results/masks")

PROJS = ["ProjSFL", "ProjSIN", "ProjSTG", "ProjZEA", "ProjBICEP"]
LABELS = {"ProjSFL": "SFL", "ProjSIN": "SIN", "ProjSTG": "STG",
          "ProjZEA": "ZEA", "ProjBICEP": "BICEP"}
# Paper-wide projection identity (dataviz skill's validated categorical
# palette, slots 1-5 in SFL/SIN/STG/ZEA/BICEP order; passes adjacent-pair
# CVD + normal-vision floors on the light surface). Reconcile against
# analysis/plot_projection_spectra.py's COLORS, which predates this set.
PROJ_COLOR = {"ProjSFL": "#2a78d6", "ProjSIN": "#eb6834", "ProjSTG": "#1baf7a",
              "ProjZEA": "#eda100", "ProjBICEP": "#e87ba4"}
# From the 2021 config: disk radii (rad) and their nominal sky percentages.
DISK_RADII_RAD = [0.175, 0.2, 0.225, 0.25, 0.35, 0.45, 0.65]
DISK_PERC = [0.75, 1.00, 1.25, 1.5, 3.0, 5.0, 10.0]
NMASK = len(DISK_PERC)
BIN = 4  # mean-pool factor before contouring; the taper is 2.5 deg wide


def _pool(a, n=BIN):
    ny, nx = a.shape
    return a[: ny // n * n, : nx // n * n].reshape(ny // n, n, nx // n, n).mean((1, 3))


def _extract():
    """Contour paths (in degrees from field centre) plus one background image."""
    from spt3g import core, maps  # noqa: F401  (registers FlatSkyMap)
    segs, meta = {}, {}
    bg = {}
    for proj in PROJS:
        for im in range(NMASK):
            path = MASK_DIR / f"flatsky_mask{im}_{proj}.g3"
            frame = next(iter(core.G3File(str(path))))
            mp = frame["T"]
            res_deg = mp.res / core.G3Units.deg
            arr = _pool(np.asarray(mp, dtype=np.float32))
            del frame, mp
            half = 0.5 * arr.shape[0] * res_deg * BIN
            ext = (-half, half, -half, half)
            meta["extent"] = np.array(ext)
            x = np.linspace(ext[0], ext[1], arr.shape[1])
            y = np.linspace(ext[2], ext[3], arr.shape[0])
            cs = plt.contour(x, y, arr, levels=[0.5])
            paths = [np.asarray(v, dtype=np.float32) for v in cs.allsegs[0]]
            plt.close("all")
            for k, v in enumerate(paths):
                segs[f"{proj}_{im}_{k}"] = v
            if im == NMASK - 1:
                bg[proj] = arr.astype(np.float16)
            del arr
    np.savez_compressed(CACHE, **segs,
                        **{f"bg_{p}": bg[p] for p in bg}, **meta)


def _load():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        _extract()
    return np.load(CACHE)


def main():
    d = _load()
    extent = tuple(d["extent"])
    OUT.mkdir(exist_ok=True)

    fs = 8
    plt.rcParams.update({
        "font.size": fs, "font.family": "serif", "text.usetex": True,
        "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "axes.linewidth": 0.6, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.autolayout": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 200,
    })

    from matplotlib.colors import BoundaryNorm, ListedColormap

    base = plt.get_cmap("viridis")
    colors = [base(0.05 + 0.9 * i / (NMASK - 1)) for i in range(NMASK)]
    lcmap = ListedColormap(colors)
    bounds = np.arange(NMASK + 1) - 0.5
    norm = BoundaryNorm(bounds, NMASK)

    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.85), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, left=0.055, right=0.9,
                                              bottom=0.195, top=0.84))
    lim = 55.0

    for i, (ax, proj) in enumerate(zip(axes, PROJS)):
        bg = d[f"bg_{proj}"].astype(np.float32)
        ax.imshow(bg, origin="lower", extent=extent, cmap="Greys",
                  vmin=0, vmax=2.8, interpolation="nearest", rasterized=True,
                  zorder=1)
        r = np.degrees(DISK_RADII_RAD[-1])
        ax.add_patch(Circle((0, 0), r, fill=False, ec="0.15", lw=0.55,
                            ls=(0, (2.2, 1.8)), zorder=6))
        for im in range(NMASK):
            k = 0
            while f"{proj}_{im}_{k}" in d:
                v = d[f"{proj}_{im}_{k}"]
                ax.plot(v[:, 0], v[:, 1], color=colors[im], lw=0.75, zorder=5)
                k += 1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_xticklabels(["", "$-20$", "$0$", "$20$", ""])
        ax.tick_params(length=2.0, pad=1.5)
        ax.set_title(LABELS[proj], pad=6, fontweight="bold")
        ax.add_patch(Rectangle((0.0, 1.0), 1.0, 0.02, transform=ax.transAxes,
                                facecolor=PROJ_COLOR[proj], edgecolor="none",
                                clip_on=False, zorder=10))
        if i > 0:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel(r"$\Delta y$ [deg]", labelpad=1)
    axes[2].set_xlabel(r"map coordinate offset from field centre [deg]", labelpad=2)

    cax = fig.add_axes([0.912, 0.20, 0.013, 0.70])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=lcmap), cax=cax,
                      ticks=np.arange(NMASK))
    cb.ax.set_yticklabels([f"{p:g}" for p in DISK_PERC])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0, pad=2, labelsize=fs - 1.5)
    cb.set_label(r"disk area [\% of sky]", labelpad=3, fontsize=fs - 1)

    fig.savefig(OUT / "masks_projections.pdf")
    fig.savefig(OUT / "masks_projections.png", dpi=300)
    print("wrote", OUT / "masks_projections.pdf")


if __name__ == "__main__":
    main()
