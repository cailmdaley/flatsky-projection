"""The simulated spectra on SPT's 1500d field, projection by projection.

This is the distortion where it lives, in the bandpowers, on a footprint with a
survey's edges (RA +-50 deg, dec -70 to -42 deg, `field_bandpowers`). The field
is wide and short, which also splits SFL from BICEP. Data only: the forward
model is compared with these spectra in `plot_model_residuals`.

  top     D_l = l(l+1) C_l / 2 pi for TT, EE, BB on logarithmic axes, the
          standard plot of a CMB spectrum, with the sims' input, the binned
          unlensed CAMB spectrum, as a thin line over them. The sims are
          unlensed, so the input BB is zero, and the BB panel carries the
          lensed BB for scale. A dotted line marks white noise of 5 uK-arcmin
          in polarization and 5/sqrt(2) in temperature, with no beam, so a
          reader sees which scales an experiment of that depth can reach.
  bottom  TT / input, EE / input, BB / EE. The two scalar ratios share a y
          axis, because they share a shape: both track the local slope of C(l)
          and, for the sheared projections, rise above l = 3000. BB/EE is pure
          leakage, since the true B is zero, and the lensed BB/EE of the real
          sky is drawn for scale, with the noise over EE dotted.

With zero shear STG's BB is the pipeline's own E-to-B floor, not a leakage. It
is drawn in the BB spectrum panel, where the log axis reaches it, and left out
of BB/EE so that axis stays on the leakage.

Every projection is one line through every dl = 50 bandpower, the mean over
the 20 realizations. There are no error bars: `print_error_reach` reports the
standard error of the mean against the line width.

Measured TT comes from `spectra.load_scalar`. `measured_tt` returns None until
the `scalar_tt_field_bandpowers` output exists, and until then the TT panels
show the input alone.

Run:  ~/code/python/spt3g_software/build/env-shell.sh python analysis/plot_projection_spectra.py
Out:  figures/projection_spectra.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as PS
import shear_leakage as SL
import spectra as S

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "figures" / "projection_spectra.pdf"
SCALAR_TT_DIR = ROOT / "results" / "highell" / "scalar_tt_field_bandpowers"

ELL_MIN, ELL_MAX = 300.0, 4800.0

SPECTRA = ("TT", "EE", "BB")
DATA_LW = 1.2
REF_LW = 0.6            # the input and unity
REF_COLOR = PS.SECONDARY[2]
INPUT_COLOR = PS.SECONDARY[0]
REF_Z = 1.0
INPUT_Z = 3.0           # over the data, which would otherwise hide it
DATA_Z = {"ProjSFL": 2.0, "ProjBICEP": 2.1, "ProjSTG": 2.2, "ProjZEA": 2.3}
FIG_H = 3.5
NOISE_P_UK_ARCMIN = 5.0
NOISE_LS = (0, (1, 1.5))


def noise_cl(spec):
    """White noise power for a spectrum, in uK^2 sr."""
    depth = NOISE_P_UK_ARCMIN / (np.sqrt(2.0) if spec == "TT" else 1.0)
    return np.radians(depth / 60.0) ** 2


def measured_tt():
    """Measured TT on the field, (n_projs, NSIMS, NBINS), or None."""
    if not SCALAR_TT_DIR.is_dir():
        return None
    return S.load_scalar("TT", field=True)


def load_data():
    """Bandpowers and inputs, per spectrum, in absolute C_l."""
    ell = S.ELL
    return dict(
        ell=ell,
        measured={"TT": measured_tt(), "EE": S.load_field("EE"),
                  "BB": S.load_field("BB")},
        input={"TT": S.theory("unlensed_TT"), "EE": S.theory("unlensed_EE"),
               "BB": np.zeros_like(ell)},
        lensed_bb=S.theory("lensed_BB"),
    )


def plot(plt, d):
    """TT, EE, BB side by side: the spectrum above, its ratio below."""
    from matplotlib.lines import Line2D

    ell = d["ell"]
    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    x = ell[keep]
    dl = ell * (ell + 1) / (2 * np.pi)

    fig, axes = plt.subplots(2, len(SPECTRA), figsize=(PS.TEXT_W, FIG_H),
                             sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    axes[1, 1].sharey(axes[1, 0])

    series = []                 # (ax, seed-by-band array), for the error check

    def data_line(ax, proj, seeds, scale=1.0, **style):
        y = scale * seeds[:, keep]
        ax.plot(x, y.mean(0), color=PS.PROJ_COLORS[proj], zorder=DATA_Z[proj],
                **{"lw": DATA_LW, **style})
        series.append((ax, y))

    for col, spec in enumerate(SPECTRA):
        top, bottom = axes[:, col]
        measured = d["measured"][spec]
        cl_in = d["input"][spec]

        if spec == "BB":
            top.plot(x, (dl * d["lensed_bb"])[keep], "--",
                     color=PS.SECONDARY[1], lw=PS.LW_REF, zorder=REF_Z)
        else:
            top.plot(x, (dl * cl_in)[keep], color=INPUT_COLOR, lw=REF_LW,
                     zorder=INPUT_Z)
        top.plot(x, (dl * noise_cl(spec))[keep], color=PS.SECONDARY[1],
                 lw=PS.LW_REF, ls=NOISE_LS, zorder=REF_Z)
        top.set_ylabel(rf"$D_\ell^{{{spec}}}$ [$\mu{{\rm K}}^2$]")
        top.set_yscale("log")
        if measured is not None:
            for j, proj in enumerate(S.PROJS):
                data_line(top, proj, dl * measured[j])
        top.margins(y=0.04)

        if spec == "BB":
            bottom.plot(x, (d["lensed_bb"] / d["input"]["EE"])[keep], "--",
                        color=PS.SECONDARY[1], lw=PS.LW_REF, zorder=REF_Z)
            for j, proj in enumerate(S.PROJS):
                if proj != SL.FLOOR_PROJ:
                    data_line(bottom, proj,
                              d["measured"]["BB"][j] / d["measured"]["EE"][j])
            bottom.plot(x, (noise_cl("BB") / d["input"]["EE"])[keep],
                        color=PS.SECONDARY[1], lw=PS.LW_REF, ls=NOISE_LS,
                        zorder=REF_Z)
            bottom.set_ylim(1e-4, 1.0)
            bottom.set_ylabel(r"$C_\ell^{BB} / C_\ell^{EE}$")
        else:
            bottom.axhline(1.0, color=REF_COLOR, lw=REF_LW, zorder=REF_Z)
            if measured is not None:
                for j, proj in enumerate(S.PROJS):
                    data_line(bottom, proj, measured[j] / cl_in)
            bottom.set_ylabel(rf"$C_\ell^{{{spec}}} / "
                              rf"C_\ell^{{{spec},\,\mathrm{{in}}}}$")
        bottom.set_yscale("log")
        bottom.set_xlabel(r"$\ell$")

    # The TT and EE ratios share this axis, so it is set once, after both.
    ratio = axes[1, 0].yaxis
    ratio.set_major_locator(plt.FixedLocator([0.5, 0.7, 1, 1.5, 2, 3, 4]))
    ratio.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
    ratio.set_minor_formatter(plt.NullFormatter())

    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlim(ELL_MIN, ELL_MAX)
    axes[0, 0].set_xticks([500, 1000, 2000, 4000])
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
    axes[0, 0].xaxis.set_minor_formatter(plt.NullFormatter())
    fig.align_ylabels(axes)

    handles = [Line2D([], [], color=PS.PROJ_COLORS[p], lw=DATA_LW)
               for p in S.PROJS]
    labels = [S.PROJ_LABEL[p] for p in S.PROJS]
    handles += [Line2D([], [], color=INPUT_COLOR, lw=REF_LW),
                Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW_REF, ls="--")]
    handles.append(Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW_REF, ls=NOISE_LS))
    labels += ["input", "lensed", "noise"]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")
    print_error_reach(fig, series)


def print_error_reach(fig, series):
    """The largest standard error of the mean, in points, against the line."""
    fig.canvas.draw()
    reach = []
    for ax, seeds in series:
        x = np.linspace(*ax.get_xlim(), seeds.shape[1] + 2)[1:-1]
        mean = seeds.mean(0)
        err = seeds.std(0, ddof=1) / np.sqrt(seeds.shape[0])
        lo = ax.transData.transform(np.c_[x, mean - err])[:, 1]
        hi = ax.transData.transform(np.c_[x, mean + err])[:, 1]
        reach.append(0.5 * (hi - lo).max() * 72 / fig.dpi)
    print(f"largest standard error {max(reach):.2f} pt, "
          f"line width {DATA_LW:.2f} pt")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PS.rc())
    plot(plt, load_data())


if __name__ == "__main__":
    main()
