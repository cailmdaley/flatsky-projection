"""The projection effect on SPT's 1500d field, measured and forward-modelled.

Three panels over 500 < l < 4500, every dl = 50 bandpower of the mean over the
20 realizations as a point and the parameter-free forward model as a line in
the same colour:

  TT / input, EE / input   the sims over the binned unlensed CAMB input. The
                           sheared projections (SFL, BICEP) trace the acoustic
                           oscillation of the spectrum's local slope and lift
                           the damping tail; ZEA stays at 1 and STG loses
                           power to magnification.
  BB / EE                  pure leakage, since the true B is zero: projected
                           BB over projected EE, the model likewise, with the
                           lensed BB over the input EE dashed for scale. STG
                           is left out: its BB is the pipeline's numerical
                           floor.

TT and EE are drawn against the estimator reference
(`residual_estimator_response`: the coupling-matrix model of the field's
shear moments, read at k = l + 1/2 through NaMaster's own coupling), BB and
its EE denominator against the integer-multipole `shear_leakage.coupling_matrix`
model of `pipeline_windows`, which also carries the measured bandpowers. The
measured arrays and the TT/EE reference carry the simulation's own pixel
windows; both are divided by the same per-projection window (`windowed /
base` in `report_pipeline_windows.py`), so points and lines sit on the input's
scale.

Error bars are drawn only where the standard error of the mean reaches past
the marker; `print_numbers` reports the largest reach and the model offsets.

Run:  bash env.sh python3 analysis/plot_spectra_model.py
          --pipeline-windows <dir> --residual-estimator-response <dir>
          --theory-spectra results/fiducial/theory_spectra --out <dir>
Out:  <out>/spectra_model.pdf, which the paper includes, and
      <out>/fig_spectra_model.png
"""

import argparse
from pathlib import Path

import numpy as np

import paper_style as PS
import shear_leakage as SL
import spectra as S
import theory as T

ELL_MIN, ELL_MAX = 500.0, 4500.0
SPECTRA = ("TT", "EE", "BB")
FIG_H = 2.35
MS = 2.0
MODEL_LW = 0.7
DATA_Z = {"ProjSFL": 2.0, "ProjBICEP": 2.1, "ProjSTG": 2.2, "ProjZEA": 2.3}
YLABEL = {"TT": r"$C_\ell^{TT} / C_\ell^{TT,\,\rm in}$",
          "EE": r"$C_\ell^{EE} / C_\ell^{EE,\,\rm in}$",
          "BB": r"$C_\ell^{BB} / C_\ell^{EE}$"}


def load_data(pipeline_windows, residual_estimator_response, theory_spectra):
    """Per spectrum and projection: (seeds x bands measured, bands model)."""
    win = np.load(pipeline_windows / "pipeline_windows.npz")
    ref = np.load(residual_estimator_response / "residual_estimator_response.npz")
    ell = win["ell"]
    th = T.load(theory_spectra)
    half = (ell[1] - ell[0]) / 2

    def theory(key):
        """The CAMB spectrum averaged over each bandpower's integer multipoles."""
        return np.array([th[key][int(l - half):int(l + half)].mean() for l in ell])

    def corrected(spec, proj):
        """Measured C_l with the simulation window removed, and its window."""
        k = f"field_{spec}_{proj}"
        window = win[f"{k}_windowed"] / win[f"{k}_base"]
        return win[f"{k}_raw"] / window, window

    out = {s: {} for s in SPECTRA}
    for proj in S.PROJS:
        for spec in ("TT", "EE"):
            meas, window = corrected(spec, proj)
            cl_in = theory(f"unlensed_{spec}")
            out[spec][proj] = (meas / cl_in,
                               ref[f"{proj}_{spec}_spherical"] / window / cl_in)
        if proj != SL.FLOOR_PROJ:
            out["BB"][proj] = (
                corrected("BB", proj)[0] / corrected("EE", proj)[0],
                win[f"field_BB_{proj}_base"] / win[f"field_EE_{proj}_base"])
    return dict(ell=ell, ratios=out,
                lensed=theory("lensed_BB") / theory("unlensed_EE"))


def plot(plt, d, out):
    from matplotlib.lines import Line2D

    ell = d["ell"]
    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    x = ell[keep]

    fig, axes = plt.subplots(1, 3, figsize=(PS.TEXT_W, FIG_H), sharex=True,
                             layout="constrained")
    axes[1].sharey(axes[0])
    points = []
    for ax, spec in zip(axes, SPECTRA):
        if spec == "BB":
            ax.plot(x, d["lensed"][keep], "--", color=PS.SECONDARY[1],
                    lw=PS.LW_REF, zorder=1)
        else:
            ax.axhline(1.0, color=PS.SECONDARY[2], lw=0.6, zorder=1)
        for proj, (meas, model) in d["ratios"][spec].items():
            c, z = PS.PROJ_COLORS[proj], DATA_Z[proj]
            y = meas[:, keep]
            mean = y.mean(0)
            err = y.std(0, ddof=1) / np.sqrt(len(y))
            ax.plot(x, mean, ls="none", marker=PS.PROJ_MARKERS[proj], ms=MS,
                    mfc="white", mec=c, mew=0.5, zorder=z)
            ax.plot(x, model[keep], color=c, lw=MODEL_LW, zorder=z + 0.05)
            points.append((ax, x, mean, err, c, z))
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(YLABEL[spec])

    ratio = axes[0].yaxis
    ratio.set_major_locator(plt.FixedLocator([0.5, 0.7, 1, 1.5, 2, 3]))
    ratio.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}$"))
    ratio.set_minor_formatter(plt.NullFormatter())
    axes[1].tick_params(labelleft=True)
    axes[2].set_ylim(2e-4, 0.6)
    axes[0].set_xlim(ELL_MIN, ELL_MAX)
    axes[0].set_xticks([1000, 2000, 3000, 4000])

    handles = [Line2D([], [], color=PS.PROJ_COLORS[p], lw=MODEL_LW,
                      marker=PS.PROJ_MARKERS[p], ms=MS + 0.6, mfc="white",
                      mew=0.6) for p in S.PROJS]
    handles.append(Line2D([], [], color=PS.SECONDARY[1], lw=PS.LW_REF, ls="--"))
    labels = [S.PROJ_LABEL[p] for p in S.PROJS] + ["lensed"]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")

    # Error bars only where they reach past the marker.
    fig.canvas.draw()
    radius, worst = 0.5 * MS, 0.0
    for ax, xx, mean, err, c, z in points:
        lo = ax.transData.transform(np.c_[xx, mean - err])[:, 1]
        hi = ax.transData.transform(np.c_[xx, mean + err])[:, 1]
        half = 0.5 * (hi - lo) * 72 / fig.dpi
        worst = max(worst, half.max())
        big = half > radius
        if big.any():
            ax.errorbar(xx[big], mean[big], err[big], ls="none", lw=0.5,
                        color=c, zorder=z)
    print(f"largest standard error {worst:.2f} pt, marker radius {radius:.2f} pt")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "spectra_model.pdf")
    fig.savefig(out / "fig_spectra_model.png", dpi=200)
    print(f"wrote {out / 'spectra_model.pdf'}")


def print_numbers(d):
    """Measured / input and measured / model, for the caption and prose."""
    ell = d["ell"]
    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    for spec in SPECTRA:
        print(f"\n{spec}: measured ratio at l (measured/model)")
        for proj, (meas, model) in d["ratios"][spec].items():
            m = meas.mean(0)
            row = []
            for L in (1000, 2000, 3000, 4000, 4475):
                k = np.argmin(abs(ell - L))
                row.append(f"{L}: {m[k]:.4g} ({m[k] / model[k]:.3f})")
            r = (m / model)[keep]
            sem = (meas.std(0, ddof=1) / np.sqrt(len(meas)) / model)[keep]
            i = np.argmax(abs(r - 1))
            print(f"  {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row)
                  + f"\n         worst {r[i] - 1:+.4f} +/- {sem[i]:.4f} at l={ell[keep][i]:.0f}"
                  + f"   rms(r-1) {np.sqrt(np.mean((r - 1) ** 2)):.4f}"
                  + f"   median sem {np.median(sem):.4f}"
                  + f"   chi2/n {np.mean(((r - 1) / sem) ** 2):.2f}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "pipeline_windows", "residual_estimator_response", "theory_spectra")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update(PS.rc())
    d = load_data(args.pipeline_windows, args.residual_estimator_response, args.theory_spectra)
    plot(plt, d, args.out)
    print_numbers(d)


if __name__ == "__main__":
    main()
