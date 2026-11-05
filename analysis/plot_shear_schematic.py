"""How a map projection becomes a shear, and what the shear does to a spectrum.

Schematic for Sect. 2 of the paper, drawn with real numbers throughout: the
SFL Jacobian at one sky position, chosen so that its g equals the rms shear of
SFL on the 5% reference disk, and the unlensed EE spectrum of the simulations.

  (a) J sends a circle on the sky to an ellipse on the grid; its principal
      axes, marked on both, show the rotation R and the stretches s1, s2.
  (b) The grid ring |l| = const reads the sky's Fourier plane on the ellipse
      of radius l rho(chi), drawn with R undone, as after flattening. One mode
      shows the direction offset delta psi.
  (c) The grid value at l is the direction average of C(l rho(chi)) over the
      band l s2 to l s1, marked for one multipole. Averaged at every l, the
      spectrum loses its acoustic peaks. Drawn at l^6 C_l, as in the spectra
      figure, so that the peaks read on a linear axis.

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
POINT = (22.5, G.DELTA_CENTER_DEG)   # (alpha, delta) in deg; g is the 5% disk's rms
ELL_BAND = 2000.0                    # the multipole whose band panel (c) marks
ELL_RANGE = (300.0, 4800.0)
POWER = 6
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
        U=U, V=Vt.T, s1=s[0], s2=s[1], mu=s[0] * s[1],
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


def panel_sky_to_grid(ax, d):
    """(a): the circle, the ellipse, and the principal axes on each."""
    ax.set_xlim(-1.25, 4.05)
    ax.set_ylim(-3.0, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 2.75
    ax.plot(*ellipse(1, 1, 0), color=SKY, lw=1.5 * PS.LW)
    gx, gy = ellipse(d["s1"], d["s2"], d["u1"])
    ax.plot(gx + cx, gy, color=GRID, lw=1.5 * PS.LW)

    # Each principal axis on the sky lands on the matching one on the grid.
    v1, v2 = d["V"][:, 0], d["V"][:, 1]
    u1, u2 = d["U"][:, 0], d["U"][:, 1]
    for v in (v1, v2):
        ax.plot([-v[0], v[0]], [-v[1], v[1]], color=SKY, lw=PS.LW_REF)
    if u1[0] < 0:
        u1 = -u1
    if u2[1] < 0:
        u2 = -u2
    for u, s in ((u1, d["s1"]), (u2, d["s2"])):
        ax.plot([cx - s * u[0], cx + s * u[0]], [-s * u[1], s * u[1]],
                color=GRID, lw=PS.LW_REF)
    ax.text(cx + 1.02 * d["s1"] * u1[0] + 0.03, 1.02 * d["s1"] * u1[1] + 0.05,
            r"$s_1$", ha="left", va="bottom")
    ax.text(cx + 1.02 * d["s2"] * u2[0] - 0.05, 1.02 * d["s2"] * u2[1] + 0.03,
            r"$s_2$", ha="right", va="bottom")

    arrow(ax, (1.14, 0), (cx - 1.12, 0), GRID, ms=8)
    ax.text(0.5 * (1.14 + cx - 1.12), 0.1, r"$J$", ha="center", va="bottom")
    ax.text(0, -1.12, "sky", ha="center", va="top")
    ax.text(cx, -1.12, "grid", ha="center", va="top")

    rows = [rf"$J = R\,\Sigma$: rotation $R$ by ${np.degrees(d['rot']):+.1f}^\circ$",
            rf"magnification $\mu = s_1 s_2 = {d['mu']:.2f}$",
            rf"shear $g = (s_1 - s_2)/(s_1 + s_2) = {d['g']:.2f}$",
            rf"with $s_1 = {d['s1']:.2f}$, $s_2 = {d['s2']:.2f}$"]
    for i, row in enumerate(rows):
        ax.text(-1.1, -1.72 - 0.36 * i, row, ha="left", va="center")


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


def panel_spectrum(ax, d, ell_th, cl):
    """(c): the sky spectrum, one band, and the spectrum averaged over its band."""
    s1, s2 = d["s1"], d["s2"]
    m = (ell_th >= ELL_RANGE[0]) & (ell_th <= ELL_RANGE[1])
    ell = ell_th[m]
    smooth = d["mu"] * np.array([direction_mean(l, s1, s2, ell_th, cl) for l in ell])
    unit = 1e15
    weight = ell**POWER / unit

    lo, hi = ELL_BAND * s2, ELL_BAND * s1
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    sky, = ax.plot(ell, weight * cl[m], color=SKY, lw=PS.LW)
    grid, = ax.plot(ell, weight * smooth, color=GRID, lw=1.5 * PS.LW)
    ax.plot(ELL_BAND, np.interp(ELL_BAND, ell, weight * smooth), "o", mfc="white",
            mec=GRID, ms=4, mew=0.9, zorder=5)
    ax.set_xlim(*ELL_RANGE)
    ax.set_ylim(0, None)
    ax.grid(False)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(rf"$\ell^{{{POWER}}}\,C_\ell^{{EE}}$ [$10^{{15}}\ \mu{{\rm K}}^2$]")
    ax.set_xticks([1000, 2000, 3000, 4000])
    top = ax.get_ylim()[1]
    for x, ha, name in ((lo, "right", r"$\ell s_2$"), (hi, "left", r"$\ell s_1$")):
        ax.text(x + (40 if ha == "left" else -40), 0.97 * top, name, ha=ha, va="top")
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
    print(f"l = {ELL_BAND:.0f}: band {ELL_BAND * d['s2']:.0f}-{ELL_BAND * d['s1']:.0f}")

    th = T.load()
    ell_th = th["ell"].astype(float)
    cl = th["unlensed_EE"]

    plt.rcParams.update(PS.rc())
    W, H = PS.TEXT_W, 2.3
    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    panel_sky_to_grid(axes(0.0, 0.1, 2.3, 2.3 * 4.7 / 5.3), d)
    panel_fourier(axes(2.18, 0.1, 2.0, 2.0), d)
    panel_spectrum(axes(4.72, 0.34, 2.46, 1.88), d, ell_th, cl)

    for x, s in ((0.0, "(a)"), (2.2, "(b)"), (4.2, "(c)")):
        fig.text(x / W, 0.995, s, ha="left", va="top")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
