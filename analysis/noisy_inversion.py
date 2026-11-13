"""Invert the projection response on noisy SPT-3G-like 1500d bandpowers.

Reads `noisy_inversion_sims`: lensed skies, a 1.2' Gaussian beam, 5 uK-arcmin
polarization noise (5/sqrt(2) in T) drawn white on the grid in two splits.
Spectra are split cross-spectra, so noise adds scatter and no bias; the same
skies' noise-free spectra separate what the noise adds.

Two estimators of one amplitude per 500-wide sky band over 500 < l < 4000,
through the scalar block (TT) and the joint EE/BB system of Eq. (coupling),
with the exterior held at its model:
  equal     every output bin weighted equally in fractional error;
  weighted  bins weighted by their measured scatter.
E amplitudes multiply the beamed lensed EE, B amplitudes the beamed lensed BB.
Amplitudes are deviations from the truth, so their mean is the residual bias
and their variance the cost. BB is reported in units of the input lensed BB.

Detectability of SFL's leaked BB: the forward model's BB<-EE term over the
measured scatter of SFL's noisy BB bandpowers, per band of width 50 and summed
in quadrature over bands.

Run:  bash env.sh python3 analysis/noisy_inversion.py
Out:  analysis/figures/noisy_inversion_summary.json, and plots under --plots
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shear_leakage as SL
import spectra as S
import theory as T
from validate_fine_spectra import read_matrix, decouple_many

DIR = S.ROOT/'results/highell/noisy_inversion_sims'
OUT = Path(__file__).resolve().parent/'figures/noisy_inversion_summary.json'
PROJS = ('ProjZEA', 'ProjSTG', 'ProjSFL')
EDGES = np.arange(500, 4001, 500)
NB = len(S.ELL)


def load(proj, directory=DIR):
    """Decoupled width-50 cross and signal spectra, windows divided out."""
    paths = sorted(directory.rglob(f'noisy_{proj}_seed*.npz'))
    with np.load(paths[0]) as z:
        counts, beam = z['counts'].copy(), float(z['beam_fwhm_arcmin'])
    raw = {k: [] for k in ('tt_cross', 'pol_cross', 'tt_signal', 'pol_signal')}
    seeds = []
    for path in paths:
        with np.load(path) as z:
            for k in raw:
                raw[k].append(z[k])
            seeds.append(int(z['seed']))
    j = S.PROJS.index(proj)
    pix = S.flat_pixwin(S.ELL)
    out = {}
    edges = np.arange(0, 6001, 50)
    for spin, ncls, specs in ((0, 1, [('TT', 0)]), (2, 4, [('EE', 0), ('BB', 3)])):
        matrix, old = read_matrix(S.ROOT/'results/highell/fine_field_coupling', proj, 0, spin)
        response = S.rebin_flat_response(matrix, old, edges, counts, ncls)
        for kind in ('cross', 'signal'):
            coupled = S.rebin_pseudo(np.asarray(raw[('tt_' if spin == 0 else 'pol_')+kind]), counts, edges)
            cl, _ = decouple_many(response, coupled)
            for spec, i in specs:
                window = S.simulation_window(spec, field=True)[j]
                out[spec, kind] = cl[:, i, :NB]/(pix*window)
    return out, seeds, beam


def design(r, lp, shapes, spin):
    """Columns: sky-band amplitudes of each input shape, through R."""
    basis = lambda shape: np.array([shape*((lp >= a) & (lp < b))
                                    for a, b in zip(EDGES[:-1], EDGES[1:])]).T
    if not spin:
        return r['TT<-TT'] @ basis(shapes['TT']), r['TT<-TT'] @ shapes['TT']
    be, bb = basis(shapes['EE']), basis(shapes['BB'])
    a = np.block([[r['EE<-EE'] @ be, r['EE<-BB'] @ bb],
                  [r['BB<-EE'] @ be, r['BB<-BB'] @ bb]])
    expected = np.r_[r['EE<-EE'] @ shapes['EE'] + r['EE<-BB'] @ shapes['BB'],
                     r['BB<-EE'] @ shapes['EE'] + r['BB<-BB'] @ shapes['BB']]
    return a, expected


def fit(a, expected, data, weight, keep):
    """Weighted least-squares deviations from the true amplitudes, per sky."""
    w = a[keep]/weight[keep, None]
    return ((data-expected)[:, keep]/weight[keep]) @ np.linalg.pinv(w).T


def band_mean(a, b):
    """Ratio of band-averaged variances, as in the noise-free comparison."""
    return a.var(0, ddof=1).mean()/b.var(0, ddof=1).mean()


def combined(a, b):
    """Ratio of variances of the inverse-variance-combined amplitude."""
    return np.sum(1/b.var(0, ddof=1))/np.sum(1/a.var(0, ddof=1))


def jackknife_ratio(x, y, stat):
    d = np.array([stat(np.delete(x, i, 0), np.delete(y, i, 0)) for i in range(len(x))])
    return float(stat(x, y)), float(np.sqrt((len(x)-1)/len(x)*np.sum((d-d.mean())**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', type=Path, default=DIR)
    p.add_argument('--plots', type=Path, default=Path('/sptlocal/user/cailmdaley/pol_project/noisy_inversion'))
    args = p.parse_args()
    th = T.load()
    lp = th['ell'].astype(float)
    mom = SL.field_moments()
    select = (S.ELL >= 500) & (S.ELL < 4000)
    amps, data, seeds, extra = {}, {}, {}, {}
    for proj in PROJS:
        d, seeds[proj], fwhm = load(proj, args.dir)
        data[proj] = d
        b2 = np.exp(-lp*(lp+1)*(np.deg2rad(fwhm/60)/np.sqrt(8*np.log(2)))**2)
        shapes = {s: th[f'lensed_{s}']*b2 for s in ('TT', 'EE', 'BB')}
        r = SL.coupling_matrix(mom, 0, S.PROJS.index(proj), S.ELL, lp, band_limit=S.LMAX)
        extra[proj] = dict(leak=r['BB<-EE'] @ shapes['EE'], lensed=r['BB<-BB'] @ shapes['BB'],
                           ee=r['EE<-EE'] @ shapes['EE'])
        for spec, spin in (('TT', False), ('POL', True)):
            a, expected = design(r, lp, shapes, spin)
            keep = np.tile(select, 2) if spin else select
            ref = np.r_[S.bin_theory(shapes['EE']), S.bin_theory(shapes['BB'])] if spin \
                else S.bin_theory(shapes['TT'])
            for kind in ('cross', 'signal'):
                x = np.c_[d['EE', kind], d['BB', kind]] if spin else d['TT', kind]
                for est, weight in (('equal', ref), ('weighted', x.std(0, ddof=1))):
                    v = fit(a, expected, x, weight, keep)
                    n = len(EDGES)-1
                    parts = [('EE', v[:, :n]), ('BB', v[:, n:])] if spin else [('TT', v)]
                    for label, val in parts:
                        amps[proj, label, kind, est] = val
        print(proj, len(seeds[proj]), 'skies', flush=True)
    assert seeds['ProjZEA'] == seeds['ProjSTG'] == seeds['ProjSFL']
    nsky = len(seeds['ProjZEA'])

    rows = []
    for (proj, spec, kind, est), v in amps.items():
        per_sky = v.mean(1)
        w = 1/v.var(0, ddof=1)
        per_sky_combined = v @ w/w.sum()
        ratio, err = jackknife_ratio(v, amps['ProjZEA', spec, kind, est], band_mean)
        cratio, cerr = jackknife_ratio(v, amps['ProjZEA', spec, kind, est], combined)
        rows.append(dict(projection=proj, spectrum=spec, spectra=kind, estimator=est,
                         mean_bias=float(per_sky.mean()),
                         mean_bias_se=float(per_sky.std(ddof=1)/np.sqrt(nsky)),
                         combined_bias=float(per_sky_combined.mean()),
                         combined_bias_se=float(per_sky_combined.std(ddof=1)/np.sqrt(nsky)),
                         bin_bias=v.mean(0).tolist(),
                         bin_sd=v.std(0, ddof=1).tolist(),
                         variance_over_zea=ratio, variance_over_zea_se=err,
                         combined_over_zea=cratio, combined_over_zea_se=cerr))

    # SFL's leaked BB against the scatter of its noisy BB bandpowers.
    sfl = data['ProjSFL']
    sd = sfl['BB', 'cross'].std(0, ddof=1)
    snr = extra['ProjSFL']['leak']/sd
    measured_leak = sfl['BB', 'cross'].mean(0) - extra['ProjSFL']['lensed']
    detect = dict(ell=S.ELL.tolist(), leak_model=extra['ProjSFL']['leak'].tolist(),
                  lensed_model=extra['ProjSFL']['lensed'].tolist(),
                  measured_leak=measured_leak.tolist(), sd=sd.tolist(), snr=snr.tolist())
    leak, lensed = extra['ProjSFL']['leak'], extra['ProjSFL']['lensed']
    above = S.ELL > 1000
    detect['ell_leak_below_lensed'] = float(S.ELL[above][np.argmax(leak[above] < lensed[above])])
    for lo, hi in ([(a, a+500) for a in range(500, 5000, 500)] + [(500, 4000)]):
        s = (S.ELL >= lo) & (S.ELL < hi)
        detect[f'snr_{lo}_{hi}'] = float(np.sqrt(np.sum(snr[s]**2)))
        detect[f'leak_over_lensed_{lo}_{hi}'] = float(np.mean(extra['ProjSFL']['leak'][s]/extra['ProjSFL']['lensed'][s]))
    result = dict(nsky=nsky, bands=EDGES.tolist(), beam_fwhm_arcmin=fwhm,
                  noise_pol_uk_arcmin=5., rows=rows, sfl_detectability=detect,
                  caveat='Known-shape band amplitudes, exterior at its model; lensed sky drawn Gaussian; '
                         'white grid noise at fixed depth per sky area; BB in units of input lensed BB.')
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    args.plots.mkdir(parents=True, exist_ok=True)
    np.savez(args.plots/'amplitudes.npz', **{'_'.join(k): v for k, v in amps.items()})
    plot(result, data, extra, args.plots)
    for r in rows:
        if r['estimator'] == 'weighted':
            print(f"{r['projection']} {r['spectrum']} {r['spectra']:6s} bias {r['combined_bias']:+.2e} ± {r['combined_bias_se']:.1e} band-mean {r['mean_bias']:+.2e}"
                  f" ± {r['mean_bias_se']:.1e}  var/ZEA {r['variance_over_zea']:.3f} ± {r['variance_over_zea_se']:.3f}"
                  f"  combined {r['combined_over_zea']:.3f} ± {r['combined_over_zea_se']:.3f}")
    print({k: v for k, v in detect.items() if k.startswith(('snr_', 'leak_over'))})


def plot(result, data, extra, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.rc()
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, layout='constrained')
    x = 0.5*(EDGES[1:]+EDGES[:-1])
    for col, spec in enumerate(('TT', 'EE', 'BB')):
        for proj in PROJS:
            for kind, ls in (('cross', '-'), ('signal', ':')):
                row = next(r for r in result['rows'] if r['projection'] == proj and r['spectrum'] == spec
                           and r['spectra'] == kind and r['estimator'] == 'weighted')
                if kind == 'cross':
                    axes[0, col].errorbar(x, row['bin_bias'], yerr=np.array(row['bin_sd'])/np.sqrt(result['nsky']),
                                          color=ps.PROJ_COLORS[proj], label=proj[4:], marker='o', ms=3)
                axes[1, col].plot(x, row['bin_sd'], ls, color=ps.PROJ_COLORS[proj],
                                  label=f'{proj[4:]} {"noisy" if kind == "cross" else "noise-free"}')
        axes[0, col].axhline(0, color='.6', lw=.6)
        unit = ' / input lensed BB' if spec == 'BB' else ' / input'
        axes[0, col].set_ylabel(spec+' residual'+unit)
        axes[1, col].set_ylabel(spec+' scatter'+unit)
        axes[1, col].set_yscale('log')
        axes[1, col].set_xlabel(r'$\ell$')
    axes[0, 0].legend(frameon=False)
    axes[1, 2].legend(frameon=False, fontsize=6)
    fig.savefig(out/'inversion.png', dpi=140)
    plt.close(fig)

    d = result['sfl_detectability']
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ell = np.array(d['ell'])
    f = ell*(ell+1)/(2*np.pi)
    ax.plot(ell, f*np.array(d['leak_model']), label='SFL leaked BB (model)')
    ax.plot(ell, f*np.array(d['measured_leak']), '.', ms=3, label='SFL BB minus lensed (measured)')
    ax.plot(ell, f*np.array(d['lensed_model']), label='lensed BB (through R)')
    ax.plot(ell, f*np.array(d['sd']), label=r'scatter per band, $\Delta\ell=50$')
    ax.set_yscale('log')
    ax.set_xlim(0, 5000)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$], beamed')
    ax.legend(frameon=False)
    fig.savefig(out/'sfl_detectability.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
