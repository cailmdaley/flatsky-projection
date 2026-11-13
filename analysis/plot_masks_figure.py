"""The disk masks, as each projection's grid holds them.

Seven concentric apodized disks (nominal sky fractions 0.75% to 10%) are
projected onto the flat-sky grid in four projections. The point of the figure
is geometric: a disk on the sphere is not a disk on the grid, and how badly it
is stretched depends on the projection. SFL and the BICEP equidistant-
cylindrical projection stretch in x by ~1/cos(dec) at the field centre and are
near-degenerate; STG and ZEA are zenithal and keep circles as circles near the
centre.

One row of four panels, one per projection:
  - grey image of the largest apodized mask (the taper shows as the soft edge),
  - W = 0.5 contours for all seven disks, coloured by disk area,
  - a dashed circle at the largest disk's true angular radius -- what a
    projection with no distortion at all would draw.

Run:  bash env.sh python3 analysis/plot_masks_figure.py
          --mask-contours <mask_contours dir> --out <dir>
Out:  <out>/masks_projections.{pdf,png}
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

import spectra as S
from paper_style import PROJ_COLORS as PROJ_COLOR, PROJS, mask_ramp

LABELS = {"ProjSFL": "SFL", "ProjSTG": "STG",
          "ProjZEA": "ZEA", "ProjBICEP": "BICEP"}
NMASK = len(S.DISKS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "mask_contours")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = np.load(args.mask_contours / "mask_contours.npz")
    extent = tuple(d["extent"])
    args.out.mkdir(parents=True, exist_ok=True)

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

    # The mask ladder is a magnitude, not an identity: it takes the neutral
    # sequential ramp, never a projection hue.
    colors = mask_ramp(NMASK)
    lcmap = ListedColormap(colors)
    bounds = np.arange(NMASK + 1) - 0.5
    norm = BoundaryNorm(bounds, NMASK)

    fig, axes = plt.subplots(1, len(PROJS), figsize=(7.1, 2.25), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, left=0.055, right=0.9,
                                              bottom=0.195, top=0.84))
    lim = 55.0

    for i, (ax, proj) in enumerate(zip(axes, PROJS)):
        bg = d[f"bg_{proj}"].astype(np.float32)
        ax.imshow(bg, origin="lower", extent=extent, cmap="Greys",
                  vmin=0, vmax=2.8, interpolation="nearest", rasterized=True,
                  zorder=1)
        r = np.degrees(S.FOOTPRINTS[S.DISKS[-1]].radius_rad)
        ax.add_patch(Circle((0, 0), r, fill=False, ec="0.15", lw=0.55,
                            ls=(0, (2.2, 1.8)), zorder=6))
        for im, footprint in enumerate(S.DISKS):
            k = 0
            while f"{proj}_{footprint}_{k}" in d:
                v = d[f"{proj}_{footprint}_{k}"]
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
    cb.ax.set_yticklabels([f"{p:g}" for p in S.PERC])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0, pad=2, labelsize=fs - 1.5)
    cb.set_label(r"disk area [\% of sky]", labelpad=3, fontsize=fs - 1)

    fig.savefig(args.out / "masks_projections.pdf")
    fig.savefig(args.out / "masks_projections.png", dpi=300)
    print("wrote", args.out / "masks_projections.pdf")


if __name__ == "__main__":
    main()
