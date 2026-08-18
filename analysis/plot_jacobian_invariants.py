"""The anchor figure for Section 2: local shear g and magnification mu over the
1500d footprint, for all five projections, computed relative to the sky (not
relative to ZEA -- this is the single-projection Jacobian of Eq. scalar_remap,
the invariants introduced right after the itemization in sec:law).

Layout: 2 rows (g, mu) x 5 columns (SFL, SIN, STG, ZEA, BICEP), one shared
colorbar per row. The 10% mask's W=0.5 contour (reusing the cache built by
plot_masks_figure.py) is overlaid on every panel so the reader sees what the
analysis footprint actually samples, versus the wider field shown for context.

Run inside the project container:
    systemd-run --user --scope -p MemoryMax=4G bash env.sh python3 analysis/plot_jacobian_invariants.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import geometry as G
from paper_style import PROJ_COLORS
from spt3g import core, maps  # noqa: F401 -- registers FlatSkyMap / MapProjection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
CONTOUR_CACHE = Path(__file__).resolve().parent / ".cache" / "mask_contours.npz"

PROJS = ["ProjSFL", "ProjSIN", "ProjSTG", "ProjZEA", "ProjBICEP"]
LABELS = {"ProjSFL": "SFL", "ProjSIN": "SIN", "ProjSTG": "STG",
          "ProjZEA": "ZEA", "ProjBICEP": "BICEP"}

# Projection colour identity lives in paper_style; used here only as a small
# non-annotative accent (a tick above each column title) so the five
# projections read as the same entities across every figure in the paper.
LIM_DEG = 55.0
RES_DEG = 0.5
N = int(2 * LIM_DEG / RES_DEG) + 1  # 221, odd so pixel N//2 sits on the centre

MASK_INDEX = 6  # the 10% (largest) disk in the 2021 suite


def sky_grid(proj):
    """(alpha_deg, delta_deg) at each pixel of an N x N, RES_DEG grid centred
    on the field centre, plus the (dx, dy) map-plane offsets in degrees."""
    fsm = maps.FlatSkyMap(
        N, N, RES_DEG * core.G3Units.deg, proj=getattr(maps.MapProjection, proj),
        alpha_center=G.ALPHA_CENTER_DEG * core.G3Units.deg,
        delta_center=G.DELTA_CENTER_DEG * core.G3Units.deg,
    )
    ix = np.arange(N, dtype=float)
    xg, yg = np.meshgrid(ix, ix, indexing="xy")
    xv = core.DoubleVector(xg.ravel())
    yv = core.DoubleVector(yg.ravel())
    a, d = fsm.xy_to_angle(xv, yv)
    alpha = np.asarray(a) / core.G3Units.deg
    delta = np.asarray(d) / core.G3Units.deg
    alpha = np.where(alpha > 180, alpha - 360, alpha)
    dx = (xg - N // 2) * RES_DEG
    dy = (yg - N // 2) * RES_DEG
    return alpha.reshape(N, N), delta.reshape(N, N), dx, dy


def compute():
    g_maps, mu_maps = {}, {}
    for proj in PROJS:
        alpha, delta, dx, dy = sky_grid(proj)
        jac = G.jacobian(proj, alpha.ravel(), delta.ravel())
        s1, s2 = G.singular_values(jac)
        g = (s1 - s2) / (s1 + s2)
        mu = s1 * s2
        g_maps[proj] = g.reshape(N, N)
        mu_maps[proj] = mu.reshape(N, N)
    return g_maps, mu_maps, dx, dy


def main():
    plt.rcParams.update({
        "font.size": 8, "font.family": "serif", "text.usetex": True,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.autolayout": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 200,
    })

    g_maps, mu_maps, dx, dy = compute()
    extent = (dx.min(), dx.max(), dy.min(), dy.max())

    contours = np.load(CONTOUR_CACHE) if CONTOUR_CACHE.exists() else None

    g_all = np.concatenate([g_maps[p].ravel() for p in PROJS])
    g_vmax = np.nanpercentile(g_all, 97.0)
    mu_all = np.concatenate([mu_maps[p].ravel() for p in PROJS])
    mu_absmax = np.nanpercentile(np.abs(mu_all - 1.0), 97.0)
    mu_norm = TwoSlopeNorm(vmin=max(1 - mu_absmax, 1e-3), vcenter=1.0, vmax=1 + mu_absmax)
    print(f"g_vmax={g_vmax:.4f}  mu range=[{1 - mu_absmax:.4f}, {1 + mu_absmax:.4f}]")

    fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.15), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.10,
                                              left=0.065, right=0.88,
                                              bottom=0.11, top=0.92))

    im_g = im_mu = None
    for j, proj in enumerate(PROJS):
        axg, axm = axes[0, j], axes[1, j]

        im_g = axg.imshow(g_maps[proj], origin="lower", extent=extent,
                          cmap="viridis", vmin=0, vmax=g_vmax,
                          interpolation="nearest", rasterized=True, zorder=1)
        im_mu = axm.imshow(mu_maps[proj], origin="lower", extent=extent,
                           cmap="RdBu_r", norm=mu_norm,
                           interpolation="nearest", rasterized=True, zorder=1)

        for ax in (axg, axm):
            if contours is not None:
                k = 0
                while f"{proj}_{MASK_INDEX}_{k}" in contours:
                    v = contours[f"{proj}_{MASK_INDEX}_{k}"]
                    ax.plot(v[:, 0], v[:, 1], color="k", lw=0.6, zorder=5)
                    k += 1
            ax.set_xlim(-LIM_DEG, LIM_DEG)
            ax.set_ylim(-LIM_DEG, LIM_DEG)
            ax.set_aspect("equal")
            ax.set_xticks([-40, -20, 0, 20, 40])
            ax.set_yticks([-40, -20, 0, 20, 40])
            ax.tick_params(length=2.0, pad=1.5)

        axg.set_title(LABELS[proj], pad=6, fontweight="bold")
        # Color-identity bar above each column title, matching this
        # projection's hue in every other figure in the paper (see
        # plot_masks_figure.py).
        axg.add_patch(Rectangle((0.0, 1.0), 1.0, 0.02, transform=axg.transAxes,
                                facecolor=PROJ_COLORS[proj], edgecolor="none",
                                clip_on=False, zorder=10))
        axg.set_xticklabels([])
        axm.set_xticklabels(["", "$-20$", "$0$", "$20$", ""])
        if j > 0:
            axg.tick_params(labelleft=False)
            axm.tick_params(labelleft=False)
        else:
            axg.set_yticklabels(["", "$-20$", "$0$", "$20$", ""])
            axm.set_yticklabels(["", "$-20$", "$0$", "$20$", ""])

    axes[0, 0].set_ylabel(r"$g$" + "\n" + r"$\Delta y$ [deg]", labelpad=1)
    axes[1, 0].set_ylabel(r"$\mu$" + "\n" + r"$\Delta y$ [deg]", labelpad=1)
    axes[1, 2].set_xlabel(r"map coordinate offset from field centre [deg]", labelpad=2)

    cax_g = fig.add_axes([0.895, 0.53, 0.014, 0.39])
    cb_g = fig.colorbar(im_g, cax=cax_g)
    cb_g.set_label(r"shear $g = (s_1-s_2)/(s_1+s_2)$", fontsize=7, labelpad=3)
    cb_g.ax.tick_params(labelsize=6.5, length=2)

    cax_mu = fig.add_axes([0.895, 0.11, 0.014, 0.39])
    cb_mu = fig.colorbar(im_mu, cax=cax_mu)
    cb_mu.set_label(r"magnification $\mu = s_1 s_2$", fontsize=7, labelpad=3)
    cb_mu.ax.tick_params(labelsize=6.5, length=2)

    OUT.mkdir(exist_ok=True)
    path = OUT / "jacobian_invariants.pdf"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)

    print("\n--- key numbers ---")
    for proj in PROJS:
        g, mu = g_maps[proj], mu_maps[proj]
        print(f"{proj:12s} g: min={np.nanmin(g):.4f} max={np.nanmax(g):.4f}  "
             f"mu: min={np.nanmin(mu):.4f} max={np.nanmax(mu):.4f}")


if __name__ == "__main__":
    main()
