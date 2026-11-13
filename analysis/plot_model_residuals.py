"""Simulations over the forward model on SPT's 1500d field: TT, EE and BB.

One panel per spectrum, each a line through every dl = 50 bandpower of the
mean over the 20 realizations, divided by the model, and a band of the
standard error of that mean.

The model is `shear_leakage.coupling_matrix` on the shear moments of the
pipeline's own field mask (`shear_leakage.field_moments`): the sky spectrum
read at the sheared wavenumber |J^T l|, with the exact
sin^2/cos^2(2 dpsi + 2 Delta) kernel for E and B and kernel 1 for T, Delta the
residual of the pipeline's own flattening angle, averaged over mode azimuth
and over the footprint with weight W^2 |det J|^2 per unit sky area, and
band-limited at the sims' l = 6000. No free parameters, no fit. TT and EE are
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
measured BB is the pipeline's floor. `plot_projection_spectra.measured_tt`
supplies TT; until it exists the TT panel holds only its unity line.

Run:  ~/code/python/spt3g_software/build/env-shell.sh python analysis/plot_model_residuals.py
Out:  figures/model_residuals.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as PS
import plot_projection_spectra as PPS
import shear_leakage as SL
import spectra as S
import theory as T

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "figures" / "model_residuals.pdf"

SPECTRA = PPS.SPECTRA
ELL_MIN, ELL_MAX = PPS.ELL_MIN, PPS.ELL_MAX
BAND_LIMIT = S.LMAX     # 6000: the run's own harmonic band limit
Y_LIM = {'TT': (0.97, 1.03), 'EE': (0.97, 1.03), 'BB': (0.97, 1.24)}
BAND_ALPHA = 0.25
FIG_H = 2.2


def binned(values):
    """Bandpowers of an array whose last axis is every multipole below LMAX."""
    return values.reshape(*values.shape[:-1], S.NBINS, S.DELTA_L).mean(-1)


WINDOWS = Path('/sptlocal/user/cailmdaley/pol_project/pipeline_windows_production.npz')
REFERENCE = Path('/sptlocal/user/cailmdaley/pol_project/residual_estimator_response.npz')


def load_ratios():
    """Measured over model, per spectrum: (projection, seed, band)."""
    measured, reference = np.load(WINDOWS), np.load(REFERENCE)
    ratios = {spec: {proj: measured[f'field_{spec}_{proj}_raw'] / reference[f'{proj}_{spec}_spherical']
                     for proj in S.PROJS} for spec in ('TT', 'EE')}
    ratios['BB'] = {proj: measured[f'field_BB_{proj}_ratio']
                    for proj in S.PROJS if proj != SL.FLOOR_PROJ}
    return ratios


def load_data():
    """The field's bandpowers and forward model, per spectrum, in absolute C_l."""
    mom = SL.field_moments()
    th = T.load()
    ell_th = th["ell"].astype(float)
    fine = np.arange(S.LMAX, dtype=float)

    model = {s: np.empty((len(S.PROJS), S.NBINS)) for s in SPECTRA}
    for j in range(len(S.PROJS)):
        R = SL.coupling_matrix(mom, 0, j, fine, ell_th, band_limit=BAND_LIMIT)
        model["BB"][j] = binned(R["BB<-EE"] @ th["unlensed_EE"])
        model["EE"][j] = binned(R["EE<-EE"] @ th["unlensed_EE"])
        model["TT"][j] = binned(R["TT<-TT"] @ th["unlensed_TT"])

    return dict(
        mom=mom, ell=S.ELL, model=model, cl_ee=S.theory("unlensed_EE"),
        ratios=load_ratios(),
        measured={"TT": PPS.measured_tt(), "EE": S.load_field("EE"),
                  "BB": S.load_field("BB")},
    )


def plot(plt, d):
    """Measured over model, TT, EE, BB, on one y axis."""
    from matplotlib.lines import Line2D

    ell = d["ell"]
    keep = (ell > ELL_MIN) & (ell < ELL_MAX)
    x = ell[keep]

    fig, axes = plt.subplots(1, len(SPECTRA), figsize=(PS.TEXT_W, FIG_H),
                             sharex=True, sharey=False, layout="constrained")
    for ax, spec in zip(axes, SPECTRA):
        ax.axhline(1.0, color=PPS.REF_COLOR, lw=PPS.REF_LW, zorder=PPS.REF_Z)
        for proj, ratio in d["ratios"][spec].items():
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

    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


def print_numbers(d):
    """A few numbers for the caption and the prose, printed rather than typed."""
    ell = d["ell"]
    ee, bb = d["measured"]["EE"], d["measured"]["BB"]
    m_ee, m_bb = d["model"]["EE"], d["model"]["BB"]
    cl_ee = d["cl_ee"]

    print("\n1500d field, measured / input EE (measured / model in parentheses):")
    for j, proj in enumerate(S.PROJS):
        r = ee[j].mean(0) / cl_ee
        row = []
        for L in (1000, 2000, 3000, 4000, 4500):
            k = np.argmin(abs(ell - L))
            row.append(f"{L}: {r[k]:.3f} ({r[k] / (m_ee[j][k] / cl_ee[k]):.3f})")
        print(f"  {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row))
    sel = (ell > 500) & (ell < 3000)
    print("\nBB on the field, 500 < l < 3000 mean [uK^2]:")
    for j, proj in enumerate(S.PROJS):
        print(f"  {S.PROJ_LABEL[proj]:<6s} measured {bb[j].mean(0)[sel].mean():9.3e}   "
              f"model {m_bb[j][sel].mean():9.3e}   "
              f"<g^2> {d['mom']['g2'][0, j]:9.3e}")
    print("\nMeasured / model, mean over l range [min, max of the band means]:")
    for spec in SPECTRA:
        m = d["measured"][spec]
        if m is None:
            continue
        for j, proj in enumerate(S.PROJS):
            if spec == "BB" and proj == SL.FLOOR_PROJ:
                continue
            r = m[j].mean(0) / d["model"][spec][j]
            row = []
            for lo, hi in ((300, 1000), (1000, 3000), (3000, 4000), (4000, 4800)):
                k = (ell > lo) & (ell < hi)
                row.append(f"{lo}-{hi}: {r[k].mean():.3f} "
                           f"[{r[k].min():.3f}, {r[k].max():.3f}]")
            print(f"  {spec} {S.PROJ_LABEL[proj]:<6s} " + "  ".join(row))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PS.rc())
    d = load_data()
    plot(plt, d)
    print_numbers(d)


if __name__ == "__main__":
    main()
