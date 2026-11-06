"""How a map projection becomes a shear, and what the shear does to a spectrum.

The paper's first figure, drawn with real numbers throughout: the
SFL Jacobian at one sky position in SPT's 1500d field, chosen so that its g
equals the rms shear of SFL on that field, and the unlensed EE spectrum of the
simulations.

  (a) The grid's square pixels are parallelograms on the sky: the same lattice
      drawn both ways, with a sky circle and the ellipse J makes of it.
  (b) The grid ring |l| = const reads the sky's Fourier plane on the ellipse
      of radius l rho(chi), drawn with R undone, as after flattening. One mode
      shows the direction offset delta psi.
  (c) The grid value at l is the direction average of C(l rho(chi)) over the
      band l s2 to l s1, marked for one multipole. Averaged at every l, the
      spectrum loses its acoustic peaks. Drawn as D_l on logarithmic axes, as
      in the spectra figure.

One colour rule across the panels: teal is the sky, ink is the grid.

Run in the SPT environment:
    ~/code/python/spt3g_software/build/env-shell.sh python analysis/plot_shear_schematic.py

Output: figures/shear_schematic.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as G
import paper_style as PS
import theory as T

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "figures" / "shear_schematic.pdf"

PROJ = "ProjSFL"
POINT = (26.26, G.DELTA_CENTER_DEG)  # (alpha, delta) in deg; g is the field's rms
ELL_BAND = 2000.0                    # the multipole whose band panel (c) marks
ELL_RANGE = (300.0, 4800.0)
LATTICE_N = 6                        # pixels on a side in panel (a)
LATTICE_PITCH = 1 / 3                # pixel side, in units of the circle's radius
PIXEL_LW = 0.5
CHI_MODE = np.deg2rad(-45.0)         # the mode drawn in panel (b)

SKY = "#009988"
GRID = PS.SECONDARY[0]
AXIS = PS.SECONDARY[2]
BAND = "#E4E4E4"


def decompose(J):
    """The polar decomposition J = R Sigma, as the numbers the figure draws."""
    U, s, Vt = np.linalg.svd(J)
    R = U @ Vt
    return dict(
        J=J, U=U, V=Vt.T, s1=s[0], s2=s[1], mu=s[0] * s[1],
        g=(s[0] - s[1]) / (s[0] + s[1]),
        rot=np.arctan2(R[1, 0], R[0, 0]),
        u1=np.arctan2(U[1, 0], U[0, 0]) % np.pi,
    )


def ellipse(a, b, tilt, n=361):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = a * np.cos(t), b * np.sin(t)
    return (x * np.cos(tilt) - y * np.sin(tilt),
            x * np.sin(tilt) + y * np.cos(tilt))


def rho(chi, s1, s2):
    return np.sqrt(s1**2 * np.cos(chi)**2 + s2**2 * np.sin(chi)**2)


def delta_psi(chi, s1, s2):
    return np.arctan2(s2 * np.sin(chi), s1 * np.cos(chi)) - chi


def direction_mean(ell, s1, s2, ell_th, cl, n_chi=720):
    """< C(l rho(chi)) >_chi, the bracket of the scalar ratio equation."""
    chi = (np.arange(n_chi) + 0.5) / n_chi * 2 * np.pi
    return np.interp(ell * rho(chi, s1, s2), ell_th, cl).mean()


def arrow(ax, p, q, color, lw=PS.LW, ms=7):
    ax.annotate("", xy=q, xytext=p, zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.32",
                                color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=ms))


def arc(ax, r, a, b, color, lw=PS.LW_REF):
    t = np.linspace(a, b, 40)
    ax.plot(r * np.cos(t), r * np.sin(t), color=color, lw=lw, solid_capstyle="butt")


def lattice(n, pitch, T):
    """Segments of an n x n square lattice centred on 0, each mapped by T."""
    t = np.linspace(-n / 2, n / 2, 2) * pitch
    edges = (np.arange(n + 1) - n / 2) * pitch
    for e in edges:
        for pts in (np.array([t, [e, e]]), np.array([[e, e], t])):
            yield T @ pts


def panel_sky_to_grid(ax, d):
    """(a): the grid's pixels and one sky circle, on the sky and on the grid."""
    ax.set_xlim(-1.45, 4.05)
    ax.set_ylim(-2.45, 1.87)
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 2.75
    J, Jinv = d["J"], np.linalg.inv(d["J"])

    for x, y in lattice(LATTICE_N, LATTICE_PITCH, Jinv):
        ax.plot(x, y, color=GRID, lw=PIXEL_LW, solid_capstyle="round")
    for x, y in lattice(LATTICE_N, LATTICE_PITCH, np.eye(2)):
        ax.plot(x + cx, y, color=GRID, lw=PIXEL_LW, solid_capstyle="round")

    ax.plot(*ellipse(1, 1, 0), color=SKY, lw=1.5 * PS.LW)
    gx, gy = ellipse(d["s1"], d["s2"], d["u1"])
    ax.plot(gx + cx, gy, color=SKY, lw=1.5 * PS.LW)

    arrow(ax, (1.32, 0), (cx - 1.12, 0), GRID, ms=8)
    ax.text(0.5 * (1.32 + cx - 1.12), 0.1, r"$J$", ha="center", va="bottom")
    bottom = -1.1
    ax.text(0, bottom, "sky", ha="center", va="top")
    ax.text(cx, bottom, "grid", ha="center", va="top")

    rows = [rf"$s_1 = {d['s1']:.2f}$, $s_2 = {d['s2']:.2f}$",
            rf"$g = (s_1 - s_2)/(s_1 + s_2) = {d['g']:.2f}$"]
    for i, row in enumerate(rows):
        ax.text(-1.3, -1.82 - 0.38 * i, row, ha="left", va="center")


def panel_fourier(ax, d):
    """(b): the grid ring and the sky ellipse it reads, R undone."""
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")

    s1, s2, tilt = d["s1"], d["s2"], d["u1"]
    ax.plot(*ellipse(1, 1, 0), color=GRID, lw=1.5 * PS.LW)
    ax.plot(*ellipse(s1, s2, tilt), color=SKY, lw=1.5 * PS.LW)

    a2 = tilt + np.pi / 2
    for s, a in ((s1, tilt), (s2, a2)):
        ax.plot([0, s * np.cos(a)], [0, s * np.sin(a)], color=AXIS,
                lw=PS.LW_REF, ls=(0, (2, 2)), zorder=1)
    ax.text(1.03 * s1 * np.cos(tilt), 1.03 * s1 * np.sin(tilt) + 0.02,
            r"$\ell s_1$", ha="left", va="bottom")
    ax.text(0.62 * s2 * np.cos(a2) + 0.16 * np.cos(tilt),
            0.62 * s2 * np.sin(a2) + 0.16 * np.sin(tilt) + 0.08,
            r"$\ell s_2$", ha="center", va="center")

    # The mode and its sky partner, at azimuth chi from the major axis.
    a_ell = tilt + CHI_MODE
    a_sky = a_ell + delta_psi(CHI_MODE, s1, s2)
    r_sky = rho(CHI_MODE, s1, s2)
    arrow(ax, (0, 0), (np.cos(a_ell), np.sin(a_ell)), GRID, ms=8)
    arrow(ax, (0, 0), (r_sky * np.cos(a_sky), r_sky * np.sin(a_sky)), SKY, ms=8)

    r_chi = 0.34
    arc(ax, r_chi, a_ell, tilt, PS.INK_SOFT)
    mid = 0.5 * (a_ell + tilt)
    ax.text(1.28 * r_chi * np.cos(mid), 1.28 * r_chi * np.sin(mid), r"$\chi$",
            ha="left", va="center")

    r_dp = 0.72
    arc(ax, r_dp, a_ell, a_sky, PS.INK_SOFT)
    mid = 0.5 * (a_ell + a_sky)
    ax.annotate(r"$\delta\psi$", xy=(r_dp * np.cos(mid), r_dp * np.sin(mid)),
                xytext=(0.47, -0.36), ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=PS.INK_SOFT, lw=0.5,
                                shrinkA=1, shrinkB=1))

    ax.text(-1.4, 1.38, "grid ring\n" r"$|\ell| = {\rm const}$",
            ha="left", va="top", linespacing=1.15)
    ax.text(-1.4, -1.38, "sky ellipse\n" r"$\ell\,\rho(\chi)$",
            ha="left", va="bottom", linespacing=1.15)


def panel_spectrum(plt, ax, d, ell_th, cl):
    """(c): the sky spectrum, one band, and the spectrum averaged over its band."""
    s1, s2 = d["s1"], d["s2"]
    m = (ell_th >= ELL_RANGE[0]) & (ell_th <= ELL_RANGE[1])
    ell = ell_th[m]
    smooth = d["mu"] * np.array([direction_mean(l, s1, s2, ell_th, cl) for l in ell])
    weight = ell * (ell + 1) / (2 * np.pi)

    lo, hi = ELL_BAND * s2, ELL_BAND * s1
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    sky, = ax.plot(ell, weight * cl[m], color=SKY, lw=PS.LW)
    grid, = ax.plot(ell, weight * smooth, color=GRID, lw=1.5 * PS.LW)
    ax.plot(ELL_BAND, np.interp(ELL_BAND, ell, weight * smooth), "o", mfc="white",
            mec=GRID, ms=4, mew=0.9, zorder=5)
    ax.set_xlim(*ELL_RANGE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(False)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{EE}$ [$\mu{\rm K}^2$]")
    ax.set_xticks([500, 1000, 2000, 4000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    for x, ha, name, pad in ((lo, "right", r"$\ell s_2$", 0.98), (hi, "left", r"$\ell s_1$", 1.02)):
        ax.text(pad * x, 0.015, name, ha=ha, va="bottom",
                transform=ax.get_xaxis_transform())
    ax.legend([sky, grid], [r"$C(\ell)$", r"$\langle C(\ell\rho)\rangle_\chi$"],
              loc="upper right", handlelength=1.4)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = decompose(G.jacobian(PROJ, *POINT)[0])
    print(f"{PROJ} at (alpha, delta) = {POINT}: g = {d['g']:.4f}, s1 = {d['s1']:.4f}, "
          f"s2 = {d['s2']:.4f}, mu = {d['mu']:.4f}, "
          f"rotation = {np.degrees(d['rot']):.2f} deg, "
          f"delta psi at chi = {np.degrees(CHI_MODE):.0f} deg: "
          f"{np.degrees(delta_psi(CHI_MODE, d['s1'], d['s2'])):.2f} deg")

    plt.rcParams.update(PS.rc())
    th = T.load()
    ell_th = th["ell"].astype(float)
    cl = th["unlensed_EE"]

    W, H = PS.TEXT_W, 2.3
    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    panel_sky_to_grid(axes(0.0, 0.1, 2.3, 2.1), d)
    panel_fourier(axes(2.18, 0.1, 2.0, 2.0), d)
    panel_spectrum(plt, axes(4.72, 0.34, 2.46, 1.88), d, ell_th, cl)

    for x, s in ((0.0, "(a)"), (2.2, "(b)"), (4.2, "(c)")):
        fig.text(x / W, 0.995, s, ha="left", va="top")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
