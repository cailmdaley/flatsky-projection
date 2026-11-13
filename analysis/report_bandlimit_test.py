"""Paired spherical band-limit intervention against the local-affine alias prediction.

`window_bandlimit_test` holds one field realization per projection at two sky
cutoffs: the production band limit, whose spectra must reproduce the
production bandpowers before anything is compared, and a lower cutoff that
differs only in the harmonic band limit. Per projection, spectrum and band,
this records the fraction of the production-cutoff power the lower cutoff
retains, the measured change, and the change `pipeline_aliases` predicts from
the windowed direct power plus the folded reciprocal-grid images. The cutoffs
are read from the test's own keys.

Run:  bash env.sh python3 analysis/report_bandlimit_test.py
          --window-bandlimit-test results/fiducial/window_bandlimit_test
          --bandpowers results/fiducial/bandpowers
          --theory-spectra data/theory_spectra.npz
          --pipeline-aliases <pipeline_aliases dir> --out <dir>
Out:  <out>/pipeline_bandlimit_summary.json, `{proj}_{spec}_{lo}_{hi}` ->
      {retained, delta_over_upper, predicted_delta_over_upper}; and a
      diagnostic figure, pipeline_bandlimit_test.{pdf,png}.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import paper_style as PS
import spectra as S

BANDS = ((4000, 4500), (4500, 4800), (4800, 5400))


def records(directory, run):
    """Each projection's test spectra, after checking the production cutoff reproduces production."""
    data = {}
    for j, proj in enumerate(S.PROJS):
        with np.load(directory/f'bandlimit_{proj}_seed0000.npz') as z:
            data[proj] = {k: z[k] for k in z.files}
        for spec in ('EE', 'BB'):
            production = run.load_field(spec, correct_windows=False)[j, 0]
            np.testing.assert_allclose(data[proj][f'{spec.lower()}_{run.lmax}'], production,
                                       rtol=2e-12, atol=1e-25)
        print(proj, f'cutoff {run.lmax} reproduces production', flush=True)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'window_bandlimit_test', 'bandpowers', 'theory_spectra', 'pipeline_aliases')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    run = S.Run.from_args(args)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(PS.rc())

    data = records(args.window_bandlimit_test, run)
    (test,) = {int(k[3:]) for d in data.values() for k in d if k.startswith('ee_')} - {run.lmax}
    upper_cut, lower_cut = run.lmax, test
    alias = np.load(args.pipeline_aliases/'pipeline_aliases.npz')
    cl_ee = run.theory('unlensed_EE')
    summary = {}
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, layout='constrained')
    for j, proj in enumerate(S.PROJS):
        d = data[proj]
        ell = d['ell']
        for row, spec in enumerate(('EE', 'BB')):
            ax = axes[row, j]
            for cutoff, color in ((upper_cut, '#b13c4e'), (lower_cut, '#276f9c')):
                key = f'{cutoff}_{proj}_{spec}'
                pred = alias[key+'_windowed']+alias[key+'_alias']
                ax.plot(ell, d[f'{spec.lower()}_{cutoff}']/cl_ee, color=color, label=str(cutoff))
                ax.plot(alias['ell'], pred/np.interp(alias['ell'], run.ell, cl_ee),
                        color=color, ls='--', lw=1)
                for lo, hi in BANDS+((5400, 5800),):
                    sel = (ell >= lo) & (ell < hi)
                    print(proj, spec, cutoff, lo, hi, 'mean/input_EE',
                          float(np.mean(d[f'{spec.lower()}_{cutoff}'][sel]/cl_ee[sel])), flush=True)
            for lo, hi in BANDS:
                sel = (ell >= lo) & (ell < hi)
                upper = d[f'{spec.lower()}_{upper_cut}'][sel]
                lower = d[f'{spec.lower()}_{lower_cut}'][sel]
                model_delta = np.interp(
                    ell[sel], alias['ell'],
                    alias[f'{upper_cut}_{proj}_{spec}_windowed']+alias[f'{upper_cut}_{proj}_{spec}_alias']
                    - alias[f'{lower_cut}_{proj}_{spec}_windowed']-alias[f'{lower_cut}_{proj}_{spec}_alias'])
                rec = dict(retained=float(lower.mean()/upper.mean()),
                           delta_over_upper=float((upper-lower).mean()/upper.mean()),
                           predicted_delta_over_upper=float(model_delta.mean()/upper.mean()))
                summary[f'{proj}_{spec}_{lo}_{hi}'] = rec
                print('paired cutoff change', proj, spec, lo, hi, rec, flush=True)
            ax.axvspan(5400, upper_cut, color='.5', alpha=.08, lw=0)
            ax.set_xlim(3500, upper_cut)
            ax.set_yscale('log')
            ax.set_ylabel(rf'$C^{{{spec}}}/C^{{EE,\rm sky}}$ ({S.PROJ_LABEL[proj]})')
            if row == 1:
                ax.set_xlabel(r'$\ell$')
                ax.set_ylim(1e-6, 2)
            else:
                ax.set_ylim(.1, 20)
    fig.legend([Line2D([], [], color='#b13c4e'), Line2D([], [], color='#276f9c'),
                Line2D([], [], color='.3', ls='--')],
               [f'Sky cutoff {upper_cut}', f'Sky cutoff {lower_cut}',
                r'Local-affine alias model ($\ell<5400$)'], ncol=3, loc='outside upper center')
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(args.out/f'pipeline_bandlimit_test.{ext}', dpi=160)
    plt.close(fig)
    (args.out/'pipeline_bandlimit_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
