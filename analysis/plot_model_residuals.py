"""Simulations over the forward model on SPT's 1500d field: TT, EE and BB.

One panel per spectrum, each a line through every dl = 50 bandpower of the
mean over the 20 realizations, divided by the model, and a band of the
standard error of that mean.

The model is `shear_leakage.coupling_matrix` on the shear moments of the
pipeline's own field mask (the `shear_moments` output): the sky spectrum
read at the sheared wavenumber |J^T l|, with the exact
sin^2/cos^2(2 dpsi + 2 Delta) kernel for E and B and kernel 1 for T, Delta the
residual of the pipeline's own flattening angle, averaged over mode azimuth
and over the footprint with weight W^2 |det J|^2 per unit sky area, and
band-limited at the sims' harmonic band limit. No free parameters, no fit. TT and EE are
then compared as the estimator sees them: the spectrum reads the sky at
k = l + 1/2, and NaMaster's own radial coupling and decoupling act on it
(`residual_estimator_response.py`), since a model averaged over integer
multipoles draws acoustic structure of its own at the 0.2% level. That
reference carries the simulation's windows, so those measured arrays have no
window removed. BB stays against the integer-multipole model with the windows
removed: the estimator reference is gated on TT and EE only, and it worsens
ZEA's BB.

Normalisation. The pseudo-C_l denominator is the plane-area mask integral,
sum(W^2 |det J|) per unit sky area, one power of |det J| lower than the
numerator's. Getting this right matters only for a projection whose
magnification varies across the footprint, and not at all for the two
equal-area projections.

Range. The 2 arcmin grid has an axis Nyquist multipole of 5400. The input
sky cutoff at 6000 is mapped to different directional cutoffs by each
projection. Aliasing can therefore arrive below 4800 on sheared projections,
and matters particularly for the small geometric BB of ZEA. This diagnostic
retains 4800 to expose that rise; it is not a universal clean-data cutoff.

Pixel windows. Both comparisons treat the coherent ring-interpolation and
finite 8x8-rebin windows separately for each projection and spin block,
inside the sky-frame forward integral. No parameter is fitted. Aliased power
and the STG estimator floor are not in the model. Bands show pointwise standard errors,
not independent errors across bins or projections; use paired-seed contrasts
(`report_pipeline_windows.py`) to compare projections.

STG is left out of BB: its model is the rounding of zero shear, and its
measured BB is the pipeline's floor.

`load_data` assembles the field's measured spectra beside the integer-bin
forward model for `paper_numbers`; the figure itself reads only the two
reference outputs.

Run:  bash env.sh python3 analysis/plot_model_residuals.py
          --pipeline-windows <dir> --residual-estimator-response <dir> --out <dir>
Out:  <out>/model_residuals.pdf, the file paper/main.tex includes, and
      <out>/fig_model_residuals.png
"""

import argparse
from pathlib import Path

import numpy as np

import paper_style as PS
import plot_projection_spectra as PPS
import shear_leakage as SL
import spectra as S
import theory as T

SPECTRA = PPS.SPECTRA
ELL_MIN, ELL_MAX = PPS.ELL_MIN, PPS.ELL_MAX
Y_LIM = {'TT': (0.97, 1.03), 'EE': (0.97, 1.03), 'BB': (0.97, 1.24)}
BAND_ALPHA = 0.25
FIG_H = 2.2


def load_ratios(windows, estimator):
    """Measured over model, per spectrum: (projection, seed, band)."""
    measured = np.load(windows / 'pipeline_windows.npz')
    reference = np.load(estimator / 'residual_estimator_response.npz')
    ratios = {spec: {proj: measured[f'field_{spec}_{proj}_raw'] / reference[f'{proj}_{spec}_spherical']
                     for proj in S.PROJS} for spec in ('TT', 'EE')}
    ratios['BB'] = {proj: measured[f'field_BB_{proj}_ratio']
                    for proj in S.PROJS if proj != SL.FLOOR_PROJ}
    return ratios


def load_data(run, windows, estimator):
    """The field's bandpowers and forward model, per spectrum, in absolute C_l.

    `run` carries bandpowers, scalar_bandpowers, simulation_windows,
    theory_spectra and shear_moments. The model is band-limited at the run's
    own harmonic band limit.
    """
    mom = run.field_moments()
    th = T.load(run.theory_spectra)
    ell_th = th["ell"].astype(float)
    fine = np.arange(run.lmax, dtype=float)

    def binned(values):
        return values.reshape(run.nbins, run.delta_l).mean(-1)

    model = {s: np.empty((len(S.PROJS), run.nbins)) for s in SPECTRA}
    for j in range(len(S.PROJS)):
        R = SL.coupling_matrix(mom, 0, j, fine, ell_th, band_limit=run.lmax)
        model["BB"][j] = binned(R["BB<-EE"] @ th["unlensed_EE"])
        model["EE"][j] = binned(R["EE<-EE"] @ th["unlensed_EE"])
        model["TT"][j] = binned(R["TT<-TT"] @ th["unlensed_TT"])

    return dict(
        mom=mom, ell=run.ell, model=model, cl_ee=run.theory("unlensed_EE"),
        ratios=load_ratios(windows, estimator),
        measured={"TT": run.load_scalar("TT", field=True), "EE": run.load_field("EE"),
                  "BB": run.load_field("BB")},
    )


def plot(plt, ell, ratios, out):
    """Measured over model, TT, EE, BB, on one y axis."""
    from matplotlib.lines import Line2D

    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    x = ell[keep]

    fig, axes = plt.subplots(1, len(SPECTRA), figsize=(PS.TEXT_W, FIG_H),
                             sharex=True, sharey=False, layout="constrained")
    for ax, spec in zip(axes, SPECTRA):
        ax.axhline(1.0, color=PPS.REF_COLOR, lw=PPS.REF_LW, zorder=PPS.REF_Z)
        for proj, ratio in ratios[spec].items():
            r = ratio[:, keep]
            mean = r.mean(0)
            err = r.std(0, ddof=1) / np.sqrt(r.shape[0])
            color = PS.PROJ_COLORS[proj]
            ax.fill_between(x, mean - err, mean + err, color=color, lw=0,
                            alpha=BAND_ALPHA, zorder=PPS.DATA_Z[proj] - 1)
            ax.plot(x, mean, color=color, lw=PS.LW, zorder=PPS.DATA_Z[proj])
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(rf"$C_\ell^{{{spec}}} / C_\ell^{{{spec},\,\mathrm{{model}}}}$")
        ax.tick_params(labelleft=True)
        ax.set_ylim(*Y_LIM[spec])

    axes[0].set_xlim(ELL_MIN, ELL_MAX)
    axes[0].set_xticks([1000, 2000, 3000, 4000])

    fig.legend([Line2D([], [], color=PS.PROJ_COLORS[p], lw=PS.LW)
                for p in S.PROJS], [S.PROJ_LABEL[p] for p in S.PROJS],
               ncol=len(S.PROJS), loc="outside upper center")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "model_residuals.pdf")
    fig.savefig(out / "fig_model_residuals.png", dpi=200)
    print(f"wrote {out / 'model_residuals.pdf'}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "pipeline_windows", "residual_estimator_response")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update(PS.rc())
    with np.load(args.pipeline_windows / "pipeline_windows.npz") as z:
        ell = z["ell"]
    plot(plt, ell, load_ratios(args.pipeline_windows, args.residual_estimator_response), args.out)


if __name__ == "__main__":
    main()
