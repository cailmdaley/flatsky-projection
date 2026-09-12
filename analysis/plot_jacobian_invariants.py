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
from matplotlib.colors import LinearSegmentedColormap, Normalize

import geometry as G
from paper_style import INK, MASK_RAMP, PROJ_COLORS, RULE
from spt3g import core, maps  # noqa: F401 -- registers FlatSkyMap / MapProjection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
CONTOUR_CACHE = Path(__file__).resolve().parent / ".cache" / "mask_contours.npz"

PROJS = ["ProjSFL", "ProjSIN", "ProjSTG", "ProjZEA", "ProjBICEP"]
LABELS = {"ProjSFL": "SFL", "ProjSIN": "SIN", "ProjSTG": "STG",
          "ProjZEA": "ZEA", "ProjBICEP": "BICEP"}

# Projection colour identity lives in paper_style: each column is named in ink
# over a rule in its projection colour, so the hue is never asked to carry text.
#
# Colour for the fields themselves is magnitude, not identity, and draws from
# the non-projection side of the palette. g is one-hue sequential on the slate
# ramp, starting from a pale slate rather than white so that g = 0 (all of STG)
# reads as a value and not as an empty panel; mu is diverging on log2(mu), so
# a factor of two of magnification and of compression sit equally far from a
# neutral-grey midpoint at mu = 1. Sky that the projection cannot reach (past
# the pole for the cylindrical pair, beyond the horizon for SIN) is hatched.
G_CMAP = LinearSegmentedColormap.from_list("slate", ["#E9EDF0"] + MASK_RAMP)
MU_CMAP = LinearSegmentedColormap.from_list(
    "teal_grey_copper",
    ["#0F6E6A", "#5FA39E", "#B9D3D0", "#E6E5E1", "#E4C2A2", "#C98B57", "#8C4E1A"])
G_VMAX = 0.6          # linear; SFL and BICEP saturate only near the pole
LOG2MU_MAX = 1.0      # mu in [1/2, 2]
HATCH_COLOR = "#B5B5B0"
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
    g_maps, mu_maps, valid = {}, {}, {}
    for proj in PROJS:
        alpha, delta, dx, dy = sky_grid(proj)
        jac = G.jacobian(proj, alpha.ravel(), delta.ravel())
        s1, s2 = G.singular_values(jac)
        g = ((s1 - s2) / (s1 + s2)).reshape(N, N)
        mu = (s1 * s2).reshape(N, N)
        # Pixels the inverse projection cannot place on the sphere: SIN's
        # xy_to_angle returns NaN beyond the horizon; SFL and BICEP keep
        # counting declination past the pole (delta < -90), where the
        # Jacobian is meaningless.
        ok = np.isfinite(g) & np.isfinite(delta) & (delta > -89.5) & (np.abs(alpha) <= 180.0)
        g_maps[proj] = np.where(ok, g, np.nan)
        mu_maps[proj] = np.where(ok, mu, np.nan)
        valid[proj] = ok
    return g_maps, mu_maps, dx, dy


def main():
    plt.rcParams.update({
        "font.size": 8, "font.family": "serif", "text.usetex": True,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "axes.grid": False,
        "axes.edgecolor": RULE, "text.color": INK, "axes.labelcolor": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "hatch.linewidth": 0.4, "hatch.color": HATCH_COLOR,
        "figure.autolayout": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 200,
    })

    g_maps, mu_maps, dx, dy = compute()
    extent = (dx.min(), dx.max(), dy.min(), dy.max())

    contours = np.load(CONTOUR_CACHE) if CONTOUR_CACHE.exists() else None

    g_cmap = G_CMAP.copy()
    g_cmap.set_bad(alpha=0.0)
    mu_cmap = MU_CMAP.copy()
    mu_cmap.set_bad(alpha=0.0)
    g_norm = Normalize(vmin=0.0, vmax=G_VMAX)
    mu_norm = Normalize(vmin=-LOG2MU_MAX, vmax=LOG2MU_MAX)

    fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.2), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.10,
                                              left=0.065, right=0.88,
                                              bottom=0.11, top=0.90))

    im_g = im_mu = None
    for j, proj in enumerate(PROJS):
        axg, axm = axes[0, j], axes[1, j]

        for ax in (axg, axm):
            # Unreachable sky shows through the transparent NaNs as hatching.
            ax.add_patch(Rectangle((-LIM_DEG, -LIM_DEG), 2 * LIM_DEG, 2 * LIM_DEG,
                                   facecolor="white", edgecolor=HATCH_COLOR,
                                   hatch="/////", linewidth=0, zorder=0))

        im_g = axg.imshow(g_maps[proj], origin="lower", extent=extent,
                          cmap=g_cmap, norm=g_norm,
                          interpolation="nearest", rasterized=True, zorder=1)
        im_mu = axm.imshow(np.log2(mu_maps[proj]), origin="lower", extent=extent,
                           cmap=mu_cmap, norm=mu_norm,
                           interpolation="nearest", rasterized=True, zorder=1)

        for ax in (axg, axm):
            if contours is not None:
                k = 0
                while f"{proj}_{MASK_INDEX}_{k}" in contours:
                    v = contours[f"{proj}_{MASK_INDEX}_{k}"]
                    ax.plot(v[:, 0], v[:, 1], color=INK, lw=0.6, zorder=5)
                    k += 1
            ax.set_xlim(-LIM_DEG, LIM_DEG)
            ax.set_ylim(-LIM_DEG, LIM_DEG)
            ax.set_aspect("equal")
            ax.set_xticks([-40, -20, 0, 20, 40])
            ax.set_yticks([-40, -20, 0, 20, 40])
            ax.tick_params(length=2.0, pad=1.5)

        # The column is named in ink; a rule in the projection's colour sits
        # under the name and is the only place the identity hue appears.
        axg.text(0.5, 1.10, LABELS[proj], transform=axg.transAxes,
                 ha="center", va="bottom", fontsize=8.5, color=INK, clip_on=False)
        axg.add_patch(Rectangle((0.0, 1.03), 1.0, 0.045, transform=axg.transAxes,
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

    axes[0, 0].set_ylabel(r"$\Delta y$ [deg]", labelpad=1)
    axes[1, 0].set_ylabel(r"$\Delta y$ [deg]", labelpad=1)
    axes[1, 2].set_xlabel(r"$\Delta x$ [deg]", labelpad=2)

    cax_g = fig.add_axes([0.895, 0.52, 0.014, 0.38])
    cb_g = fig.colorbar(im_g, cax=cax_g, extend="max")
    cb_g.set_label(r"$g$", fontsize=8, labelpad=3)
    cb_g.set_ticks([0, 0.2, 0.4, 0.6])
    cb_g.ax.tick_params(labelsize=6.5, length=2)

    cax_mu = fig.add_axes([0.895, 0.11, 0.014, 0.38])
    cb_mu = fig.colorbar(im_mu, cax=cax_mu, extend="both")
    cb_mu.set_label(r"$\mu$", fontsize=8, labelpad=3)
    cb_mu.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb_mu.set_ticklabels(["$1/2$", r"$1/\sqrt{2}$", "$1$", r"$\sqrt{2}$", "$2$"])
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
