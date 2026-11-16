"""The 1500d field's residuals against the forward model, on independent skies.

The 260 skies of `bin_covariance_sims` pass through the production field
pipeline but share no realization with the 20 production skies, so they test
the model without the skies it was looked at on. Their unit-width pseudo-
spectra are rebinned to the bandpower width and decoupled with the production
field workspaces (`coupling_matrices`), then divided by the flat pixel window,
as production does. Each is compared with two references from upstream:

  bin        the window-corrected coupling-matrix model on the production
             bins (`pipeline_windows`, `_windowed`);
  spherical  that model read at k = l + 1/2 through NaMaster's own radial
             response (`residual_estimator_response`, `_spherical`).

Over 1000 <= l < 4000 it records, per projection and spectrum:

  independent_clean  the band mean of each sky's fractional residual and its
                     standard error, on both references (TT and EE);
  ensemble_tests     a Hotelling T^2 test of the independent mean against each
                     reference, with the band mean, its error, and the largest
                     single-bin residual (TT, EE, and the sheared BB);

and, on the 20 production skies, `paired`: BICEP minus ZEA on the spherical
reference in EE, the same skies through both projections (BICEP is not among
the independent skies' projections). STG's BB is the pipeline's floor, not a
model prediction, and is left out.

Run:  bash env.sh python3 analysis/residual_summary.py
          --pipeline-windows <dir> --residual-estimator-response <dir>
          --bin-covariance-sims results/fiducial/bin_covariance_sims
          --coupling-matrices results/fiducial/coupling_matrices
          --flatsky-masks results/fiducial/flatsky_masks
          --delta-ell d50 --out <dir>
Out:  <out>/residual_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import f

import spectra as S
from _decisions import parse_delta_ell
from compare_bin_covariance import load_ensemble

PROJS = ('ProjSFL', 'ProjSTG', 'ProjZEA')
CLEAN = (1000, 4000)


def estimate(x):
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))), nseed=len(x))


def hotelling(residual):
    """T^2 of an ensemble's mean residual against zero, with its F p-value."""
    n, dim = residual.shape
    mean = residual.mean(0)
    t2 = float(n * mean @ np.linalg.solve(np.cov(residual, rowvar=False), mean))
    return t2, float(f.sf((n - dim) * t2 / ((n - 1) * dim), dim, n - dim))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'pipeline_windows', 'residual_estimator_response', 'bin_covariance_sims',
                 'coupling_matrices', 'flatsky_masks')
    ap.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = np.load(args.pipeline_windows / 'pipeline_windows.npz')
    response = np.load(args.residual_estimator_response / 'residual_estimator_response.npz')
    (res_arcmin,) = {json.loads(q.read_text())['res_arcmin']
                     for q in args.flatsky_masks.glob('index_*.json')}
    ell = report['ell']
    np.testing.assert_array_equal(ell, (np.arange(len(ell)) + .5) * args.delta_ell)
    pix = S.flat_pixwin(ell, res_arcmin)
    sel = (ell >= CLEAN[0]) & (ell < CLEAN[1])

    sims, seeds = load_ensemble(args.bin_covariance_sims, args.coupling_matrices,
                                [args.delta_ell], PROJS)
    result = dict(lo=CLEAN[0], hi=CLEAN[1], nseed={p: len(seeds[p]) for p in PROJS},
                  independent_clean=[], ensemble_tests=[], paired=[])
    for proj in PROJS:
        for spec in ('TT', 'EE', 'BB'):
            if spec == 'BB' and proj == 'ProjSTG':
                continue
            independent = sims[proj, args.delta_ell, spec.lower(), 'decoupled'] / pix
            references = dict(bin=report[f'field_{spec}_{proj}_windowed'],
                              spherical=response[f'{proj}_{spec}_spherical'])
            row = dict(projection=proj, spectrum=spec)
            for label, model in references.items():
                residual = independent[:, sel] / model[sel] - 1
                band = estimate(residual.mean(-1))
                row[label + '_residual'], row[label + '_se'] = band['mean'], band['se']
                t2, p = hotelling(residual)
                result['ensemble_tests'].append(dict(
                    projection=proj, spectrum=spec, null='new-model-' + label,
                    t2=t2, dimension=int(sel.sum()), p=p, mean=band['mean'], se=band['se'],
                    max_bin_residual=float(np.abs(residual.mean(0)).max())))
                print(f'{proj} {spec} {label}: {100 * band["mean"]:+.4f} +/- '
                      f'{100 * band["se"]:.4f} %, T2 {t2:.1f}, p {p:.3g}', flush=True)
            if spec != 'BB':
                result['independent_clean'].append(row)

    residual = {proj: report[f'field_EE_{proj}_raw'][:, sel] / response[f'{proj}_EE_spherical'][sel] - 1
                for proj in ('ProjBICEP', 'ProjZEA')}
    contrast = (residual['ProjBICEP'] - residual['ProjZEA']).mean(-1)
    result['paired'].append(dict(spectrum='EE', projections=['ProjBICEP', 'ProjZEA'],
                                 **estimate(contrast)))
    print('paired BICEP - ZEA, EE:', result['paired'][-1])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'residual_summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
