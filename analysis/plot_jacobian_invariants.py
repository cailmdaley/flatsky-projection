"""The anchor figure for Section 2: local shear g and magnification mu over the
1500d footprint, for all five projections, computed relative to the sky (not
relative to ZEA -- this is the single-projection Jacobian of Eq. scalar_remap,
the invariants introduced right after the itemization in sec:law).

Layout: 2 rows (g, mu) x 5 columns (SFL, SIN, STG, ZEA, BICEP), one shared
colorbar per row. The 10% mask's W=0.5 contour (contoured from the projected
mask image cached by plot_masks_figure.py) is overlaid on every panel so the
reader sees what the analysis footprint actually samples, versus the wider
field shown for context.

Reachable sky. A grid pixel is drawn only if the sky point it names projects
back onto it. That one test covers SIN's horizon, the cylindrical pair's
pole, and SFL's boundary meridian: past |alpha - alpha_0| = 180 deg, near the
pole, SFL's grid wraps around and samples sky it has already drawn, so
neither its Jacobian nor its mask there is new sky. The mask contour is taken
of W on reachable pixels only (zero elsewhere), so it closes on the footprint
as the projection reaches it rather than tracing the wrapped copies of the
disk.

Run inside the project container:
    OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=12G -p CPUQuota=400% bash env.sh python3 analysis/plot_jacobian_invariants.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

import geometry as G
import paper_style as PS
from paper_style import INK, MASK_RAMP, PROJ_COLORS
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


def flat_map(proj):
    """The N x N, RES_DEG FlatSkyMap centred on the field."""
    return maps.FlatSkyMap(
        N, N, RES_DEG * core.G3Units.deg, proj=getattr(maps.MapProjection, proj),
        alpha_center=G.ALPHA_CENTER_DEG * core.G3Units.deg,
        delta_center=G.DELTA_CENTER_DEG * core.G3Units.deg,
    )


def reachable(proj, dx, dy):
    """(alpha_deg, delta_deg, ok) at map-plane offsets (dx, dy) in degrees.

    ok: the inverse projection names a sky point that the forward projection
    sends back to the same place on the grid.
    """
    fsm = flat_map(proj)
    px = np.ravel(dx) / RES_DEG + N // 2
    py = np.ravel(dy) / RES_DEG + N // 2
    a, d = fsm.xy_to_angle(core.DoubleVector(px), core.DoubleVector(py))
    x2, y2 = fsm.angle_to_xy(a, d)
    alpha = np.asarray(a) / core.G3Units.deg
    delta = np.asarray(d) / core.G3Units.deg
    back = np.hypot(np.asarray(x2) - px, np.asarray(y2) - py)
    with np.errstate(invalid="ignore"):
        ok = back < 1e-3
    alpha = np.where(alpha > 180, alpha - 360, alpha)
    shape = np.shape(dx)
    return alpha.reshape(shape), delta.reshape(shape), ok.reshape(shape)


def sky_grid(proj):
    """(alpha_deg, delta_deg, ok) at each pixel of the N x N grid, plus the
    (dx, dy) map-plane offsets in degrees."""
    off = (np.arange(N) - N // 2) * RES_DEG
    dx, dy = np.meshgrid(off, off, indexing="xy")
    alpha, delta, ok = reachable(proj, dx, dy)
    return alpha, delta, ok, dx, dy


def compute():
    g_maps, mu_maps, valid = {}, {}, {}
    for proj in PROJS:
        alpha, delta, ok, dx, dy = sky_grid(proj)
        with np.errstate(invalid="ignore"):
            jac = G.jacobian(proj, alpha.ravel(), delta.ravel())
            s1, s2 = G.singular_values(jac)
        g = ((s1 - s2) / (s1 + s2)).reshape(N, N)
        mu = (s1 * s2).reshape(N, N)
        # The Jacobian in (alpha, delta) is singular at the pole itself.
        ok &= np.isfinite(g) & np.isfinite(mu) & (delta > -89.5)
        g_maps[proj] = np.where(ok, g, np.nan)
        mu_maps[proj] = np.where(ok, mu, np.nan)
        valid[proj] = ok
    return g_maps, mu_maps, dx, dy


def footprint(proj, cache):
    """(x, y, W) of the 10% mask on its cached grid, zeroed off reachable sky."""
    w = cache[f"bg_{proj}"].astype(float)
    ext = cache["extent"]
    x = np.linspace(ext[0], ext[1], w.shape[1])
    y = np.linspace(ext[2], ext[3], w.shape[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    _, _, ok = reachable(proj, xx, yy)
    return x, y, np.where(ok, w, 0.0)


def main():
    plt.rcParams.update(PS.rc())
    plt.rcParams.update({
        "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "hatch.linewidth": 0.4, "hatch.color": HATCH_COLOR,
    })

    g_maps, mu_maps, dx, dy = compute()
    extent = (dx.min(), dx.max(), dy.min(), dy.max())

    cache = np.load(CONTOUR_CACHE) if CONTOUR_CACHE.exists() else None

    g_cmap = G_CMAP.copy()
    g_cmap.set_bad(alpha=0.0)
    mu_cmap = MU_CMAP.copy()
    mu_cmap.set_bad(alpha=0.0)
    g_norm = Normalize(vmin=0.0, vmax=G_VMAX)
    mu_norm = Normalize(vmin=-LOG2MU_MAX, vmax=LOG2MU_MAX)

    # Square panels at text width, laid out in inches: margins for the tick
    # labels and axis labels, a header band for the column names, and a strip
    # on the right for the two colour bars.
    left, right, top, bottom = 0.40, 0.66, 0.30, 0.36
    wgap, hgap, cbar_gap, cbar_w = 0.05, 0.08, 0.08, 0.09
    side = (PS.TEXT_W - left - right - 4 * wgap) / 5
    height = bottom + 2 * side + hgap + top
    fig, axes = plt.subplots(2, 5, figsize=(PS.TEXT_W, height),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(
                                 wspace=wgap / side, hspace=hgap / side,
                                 left=left / PS.TEXT_W,
                                 right=1 - right / PS.TEXT_W,
                                 bottom=bottom / height, top=1 - top / height))

    im_g = im_mu = None
    for j, proj in enumerate(PROJS):
        axg, axm = axes[0, j], axes[1, j]
        fp = footprint(proj, cache) if cache is not None else None

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
            if cache is not None:
                ax.contour(*fp, levels=[0.5], colors=INK, linewidths=0.6,
                           zorder=5)
            ax.set_xlim(-LIM_DEG, LIM_DEG)
            ax.set_ylim(-LIM_DEG, LIM_DEG)
            ax.set_aspect("equal")
            ax.set_xticks([-40, -20, 0, 20, 40])
            ax.set_yticks([-40, -20, 0, 20, 40])

        # The column is named in ink; a rule in the projection's colour sits
        # under the name and is the only place the identity hue appears.
        axg.text(0.5, 1.10, LABELS[proj], transform=axg.transAxes,
                 ha="center", va="bottom", color=INK, clip_on=False)
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

    def cbar_axes(row):
        pos = axes[row, -1].get_position()
        x0 = pos.x1 + cbar_gap / PS.TEXT_W
        return fig.add_axes([x0, pos.y0, cbar_w / PS.TEXT_W, pos.height])

    cb_g = fig.colorbar(im_g, cax=cbar_axes(0), extend="max")
    cb_g.set_label(r"$g$", labelpad=3)
    cb_g.set_ticks([0, 0.2, 0.4, 0.6])

    cb_mu = fig.colorbar(im_mu, cax=cbar_axes(1), extend="both")
    cb_mu.set_label(r"$\mu$", labelpad=3)
    cb_mu.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb_mu.set_ticklabels(["$1/2$", r"$1/\sqrt{2}$", "$1$", r"$\sqrt{2}$", "$2$"])

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
