"""How a projection distorts a spectrum, place by place, and what the footprint sums to.

The paper's first figure, drawn with real numbers throughout: BICEP's
equidistant cylindrical grid (standard parallel -56 deg) over SPT's 1500d
field, and the unlensed EE spectrum of the simulations. BICEP's Jacobian is
diagonal and depends on declination alone -- the identity on the standard
parallel, compressed east-west to the north, stretched to the south -- so one
column of the field carries every deformation the footprint has, and there is
no rotation for flattening to remove.

Left to right:
  map     The field on the grid, with Tissot's indicatrix: the ellipses that
          equal small sky circles become. Three are drawn in ink, one per row.
  rows    One small sky circle and its image on the grid, labelled with that
          point's shear g and magnification mu. The stripes are the crests of
          one E-mode plane wave; the ticks are its polarization, parallel to the
          wavevector on the sky. The grid shears the crests but carries the
          ticks unrotated, so they no longer meet the crests at right angles:
          the grid reads part of the E mode as B.
  local   Each point's EE spectrum on the grid, mu < cos^2(2 dpsi) C(l rho) >
          over mode direction (Eq. forward at one position), against the input.
  field   The same quantity averaged over the footprint with the forward
          model's weight (sky area x |det J|, top-hat mask), against the input.

One colour rule: teal is the sky, ink is the grid.

Run:  bash env.sh python3 analysis/plot_shear_schematic.py
          --theory-spectra results/fiducial/theory_spectra --out <dir>
Out:  <out>/shear_schematic.pdf, which the paper includes, and
      <out>/fig_shear_schematic.png
"""

import argparse
from pathlib import Path

import numpy as np

import geometry as G
import paper_style as PS
import spectra as S
import theory as T

PROJ = "ProjBICEP"
FIELD_RA = (-50.0, 50.0)
FIELD_DEC = (-70.0, -42.0)
ROW_DEC = (-44.0, -56.0, -68.0)       # the three points, north to south, at RA 0
TISSOT_RA = np.linspace(-40, 40, 7)
TISSOT_DEC = np.linspace(-44, -68, 5)
TISSOT_R = 2.3                        # indicatrix radius on the sky, deg

K_ANGLE = np.deg2rad(45.0)            # the plane wave's direction on the sky
CREST_SPACING = 0.62                  # in units of the sky circle's radius
TICK = 0.3

ELL_RANGE = (300.0, 4500.0)
N_CHI = 720

SKY = "#009988"
GRID = PS.SECONDARY[0]
FAINT = PS.SECONDARY[2]


def local_ee(ell, J, ell_th, cl):
    """|det J| < cos^2(2 dpsi) C(|J^T l|) > over the grid mode's direction."""
    phi = (np.arange(N_CHI) + 0.5) / N_CHI * 2 * np.pi
    lhat = np.stack([np.cos(phi), np.sin(phi)])
    q = J.T @ lhat
    rho = np.hypot(*q)
    dpsi = np.arctan2(q[1], q[0]) - phi
    kernel = np.cos(2 * dpsi) ** 2
    c = np.interp(ell[:, None] * rho[None, :], ell_th, cl)
    return abs(np.linalg.det(J)) * (kernel * c).mean(axis=1)


def invariants(J):
    s1, s2 = G.singular_values(J)
    return (s1 - s2) / (s1 + s2), s1 * s2


def circle(n=361):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)])


def plane_wave(J, ax, x0, color):
    """Crests and polarization ticks of one E mode, inside the unit circle,
    mapped by J; the ticks keep their sky orientation."""
    k = np.array([np.cos(K_ANGLE), np.sin(K_ANGLE)])
    t = np.array([-k[1], k[0]])
    for off in (-CREST_SPACING, 0.0, CREST_SPACING):
        half = np.sqrt(1 - off**2)
        ends = off * k[:, None] + np.outer(t, [-half, half])
        x, y = J @ ends
        ax.plot(x + x0, y, color=color, lw=0.6, solid_capstyle="butt")
        for s in np.arange(-1, 2) * 0.48 * half:
            c = J @ (off * k + s * t)
            tx, ty = 0.5 * TICK * k
            ax.plot([c[0] - tx + x0, c[0] + tx + x0], [c[1] - ty, c[1] + ty],
                    color=color, lw=1.1, solid_capstyle="round")


def panel_map(ax):
    """The 1500d field on BICEP's grid, with Tissot's indicatrix."""
    ra_edge = np.linspace(*FIELD_RA, 101)
    dec_edge = np.linspace(*FIELD_DEC, 51)
    ra = np.concatenate([ra_edge, np.full(51, FIELD_RA[1]), ra_edge[::-1],
                         np.full(51, FIELD_RA[0])])
    dec = np.concatenate([np.full(101, FIELD_DEC[0]), dec_edge,
                          np.full(101, FIELD_DEC[1]), dec_edge[::-1]])
    fsm = G._flat_map(PROJ)
    x, y = G._xy(fsm, ra, dec)
    ax.plot(x, y, color=PS.RULE, lw=0.6)

    centres = {}
    for d in TISSOT_DEC:
        for a in TISSOT_RA:
            J = G.jacobian(PROJ, a, d)[0]
            cx, cy = G._xy(fsm, a, d)[:, 0]
            ex, ey = TISSOT_R * (J @ circle())
            hot = a == 0 and d in ROW_DEC
            ax.plot(ex + cx, ey + cy, color=GRID if hot else FAINT,
                    lw=0.9 if hot else 0.5)
            if hot:
                centres[d] = (cx, cy)
    ax.set_aspect("equal")
    ax.axis("off")
    for d in FIELD_DEC + (-56.0,):
        cy = G._xy(fsm, FIELD_RA[0], d)[1, 0]
        ax.text(x.min() - 1.5, cy, rf"${d:.0f}^\circ$", ha="right", va="center")
    return centres


def panel_row(ax, J, cx_grid):
    ax.plot(*circle(), color=SKY, lw=1.2)
    plane_wave(np.eye(2), ax, 0.0, SKY)
    ex, ey = J @ circle()
    ax.plot(ex + cx_grid, ey, color=GRID, lw=1.2)
    plane_wave(J, ax, cx_grid, GRID)
    ax.annotate("", xy=(cx_grid - 1.75, 0), xytext=(1.2, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3",
                                color=GRID, lw=0.8, shrinkA=0, shrinkB=0))
    g, mu = invariants(J)
    ax.text(cx_grid + 1.72, 0, f"$g = {g:.2f}$\n$\\mu = {mu:.2f}$",
            ha="left", va="center", linespacing=1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def spectrum_axes(plt, ax, labels):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*ELL_RANGE)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(False)
    ax.set_xticks([500, 1000, 2000, 4000])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    if labels:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
        ax.set_xlabel(r"$\ell$")
    else:
        ax.xaxis.set_major_formatter(plt.NullFormatter())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "theory_spectra")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PS.rc())
    th = T.load(args.theory_spectra)
    ell_th = th["ell"].astype(float)
    cl = th["unlensed_EE"]
    ell = np.geomspace(*ELL_RANGE, 400)
    dl = ell * (ell + 1) / (2 * np.pi)
    sky = dl * np.interp(ell, ell_th, cl)

    # The footprint: BICEP's J depends on declination alone, so one column of
    # the field, weighted by sky area x |det J| (Eq. forward), is the whole of it.
    decs = np.linspace(*FIELD_DEC, 113)
    jac = G.jacobian(PROJ, np.zeros_like(decs), decs)
    w = np.cos(np.deg2rad(decs)) * np.abs(np.linalg.det(jac))
    field = dl * np.average([local_ee(ell, J, ell_th, cl) for J in jac],
                            axis=0, weights=w)
    rows = [G.jacobian(PROJ, 0.0, d)[0] for d in ROW_DEC]
    local = [dl * local_ee(ell, J, ell_th, cl) for J in rows]
    for d, J in zip(ROW_DEC, rows):
        g, mu = invariants(J)
        print(f"{PROJ} at dec {d:+.0f}: g = {g:.4f}, mu = {mu:.4f}")
    g_field = invariants(jac)[0]
    print(f"field rms g = {np.sqrt(np.average(g_field**2, weights=w)):.4f}")

    W, H = PS.TEXT_W, 2.35
    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    row_h = 0.62
    row_y = [H - 0.12 - row_h - i * (row_h + 0.11) for i in range(3)]
    ylim = (min(sky.min(), field.min()) * 0.7, max(sky.max(), field.max()) * 1.5)

    map_ax = axes(0.36, H / 2 - 0.45, 1.3, 0.9)
    centres = panel_map(map_ax)
    map_xmax = map_ax.dataLim.x1

    cx_grid, x_hi = 3.2, 6.7
    row_axes = []
    for i, J in enumerate(rows):
        ax = axes(1.86, row_y[i], row_h / 2.3 * (x_hi + 1.15), row_h)
        ax.set_xlim(-1.15, x_hi)
        ax.set_ylim(-1.15, 1.15)
        panel_row(ax, J, cx_grid)
        row_axes.append(ax)
    row_axes[0].text(0, 1.2, "sky", ha="center", va="bottom")
    row_axes[0].text(cx_grid, 1.2, "grid", ha="center", va="bottom")

    # Leader lines from the map's marked indicatrices to the rows' sky circles.
    for d, ax in zip(ROW_DEC, row_axes):
        p = fig.transFigure.inverted().transform(map_ax.transData.transform(
            (map_xmax + 1.5, centres[d][1])))
        q = fig.transFigure.inverted().transform(ax.transData.transform((-1.3, 0)))
        fig.add_artist(plt.Line2D([p[0], q[0]], [p[1], q[1]], color=FAINT,
                                  lw=0.5, transform=fig.transFigure))

    mini = []
    for i, spec in enumerate(local):
        ax = axes(4.08, row_y[i] + 0.04, 1.0, row_h - 0.08)
        ax.plot(ell, sky, color=SKY, lw=0.7)
        ax.plot(ell, spec, color=GRID, lw=1.0)
        spectrum_axes(plt, ax, labels=(i == 2))
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        mini.append(ax)

    big = axes(5.38, 0.36, 1.42, H - 0.48)
    s, = big.plot(ell, sky, color=SKY, lw=0.8)
    f, = big.plot(ell, field, color=GRID, lw=1.5)
    big.set_ylim(*ylim)
    spectrum_axes(plt, big, labels=True)
    big.yaxis.tick_right()
    big.yaxis.set_label_position("right")
    big.set_ylabel(r"$D_\ell^{EE}$ [$\mu{\rm K}^2$]")
    big.legend([s, f], ["sky", "grid, footprint average"], loc="lower left",
               handlelength=1.4)

    # The three local spectra converge on the footprint average.
    tip = fig.transFigure.inverted().transform(
        big.transAxes.transform((-0.03, 0.5)))
    for ax in mini:
        p = fig.transFigure.inverted().transform(ax.transAxes.transform((1.03, 0.5)))
        fig.add_artist(plt.Line2D([p[0], tip[0]], [p[1], tip[1]], color=FAINT,
                                  lw=0.5, transform=fig.transFigure))

    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "shear_schematic.pdf")
    fig.savefig(args.out / "fig_shear_schematic.png", dpi=200)
    print(f"wrote {args.out / 'shear_schematic.pdf'}")


if __name__ == "__main__":
    main()
