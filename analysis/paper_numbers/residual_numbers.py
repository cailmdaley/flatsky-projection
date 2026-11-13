"""Bounded residual claims, paper/residual_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/residual_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def production_upper():
    """STG's TT over 4500-4800 on the 20 skies, against the integer-bin model."""
    d = np.load('/sptlocal/user/cailmdaley/pol_project/pipeline_windows_production.npz')
    sel = (d['ell'] >= 4500) & (d['ell'] < 4800)
    per_seed = (d['field_TT_ProjSTG_ratio'][:, sel] - 1).mean(-1)
    return {'mean': float(per_seed.mean()),
            'se': float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))}


def production_hann_worst():
    """Largest TT or EE residual of the 20 skies on the estimator reference.

    Residuals are smoothed with the Hann window of full width 600 that the
    paper uses for measured power, away from the ends of 1000 < l < 4000.
    """
    directory = Path('/sptlocal/user/cailmdaley/pol_project')
    d = np.load(directory / 'pipeline_windows_production.npz')
    reference = np.load(directory / 'residual_estimator_response.npz')
    ell = d['ell']
    centres = np.flatnonzero((ell >= 1300) & (ell < 3700))
    worst = 0.
    for spec in ('TT', 'EE'):
        for proj in S.PROJS:
            residual = (d[f'field_{spec}_{proj}_raw'] / reference[f'{proj}_{spec}_spherical']).mean(0) - 1
            for i in centres:
                near = np.abs(ell - ell[i]) < 300
                weight = np.cos(np.pi * (ell[near] - ell[i]) / 600)**2
                worst = max(worst, abs((residual[near] * weight).sum() / weight.sum()))
    return worst


def residual_numbers():
    """Generate only the bounded residual claims, never a closure calibration.

    Run summarize_residual_anomalies.py after all canonical controls and gates.
    The radial response remains an approximate diagnostic; the summary retains
    covariant per-bin tests alongside band means so those cannot be confused.
    """
    import json
    d = json.loads((ROOT/'analysis/figures/residual_anomalies/summary.json').read_text())
    M = Macros()

    def find(rows, **keys):
        matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
        assert len(matches) == 1, keys
        return matches[0]

    def uncertainty(row, mean='mean', se='se', nd=3):
        return rf"$({100*row[mean]:+.{nd}f}\pm{100*row[se]:.{nd}f})\%$"

    stg = find(d['production_conditioned'], projection='ProjSTG', spectrum='TT', lo=4500, hi=4800)
    M.add('residSTGTTUpper', uncertainty(stg))
    for spec in ('TT', 'EE'):
        row = find(d['paired'], spectrum=spec)
        M.add('residBICEPZEA'+spec, uncertainty(row))
    zea = find(d['map_interventions'], projection='ProjZEA', spectrum='BB', lo=4500, hi=4800)
    assert zea['nseed'] == 4
    M.add('residZEABBRefineDeltaPP', rf"$({100*zea['delta']:+.2f}\pm{100*zea['delta_se']:.2f})$")
    M.add('residZEABBRefineRemaining', uncertainty(zea, 'fine', 'fine_se', nd=2))
    M.add('residMapSeeds', str(zea['nseed']))
    for proj in ('ProjSFL', 'ProjSTG', 'ProjZEA'):
        for spec in ('TT', 'EE'):
            row = find(d['independent_clean'], projection=proj, spectrum=spec)
            M.add('resid'+proj[4:]+spec+'Clean', uncertainty(row, 'spherical_residual', 'spherical_se'))
            M.add('resid'+proj[4:]+spec+'Integer', uncertainty(row, 'bin_residual', 'bin_se'))
    for spec in ('TT', 'EE'):
        worst = max(abs(r['spherical_residual']) for r in d['independent_clean'] if r['spectrum'] == spec)
        M.add('resid'+spec+'CleanWorst', pct(worst, 2))
    low = find(d['independent_low'], projection='ProjZEA', spectrum='TT')
    M.add('residZEATTLowInteger', uncertainty(low, 'bin_residual', 'bin_se', nd=2))
    M.add('residZEATTLow', uncertainty(low, 'spherical_residual', 'spherical_se', nd=2))
    fits = {(r['projection'], r['spectrum']): r for r in d['ensemble_tests']
            if r['null'] == 'new-model-spherical' and r['lo'] == 1000 and r['hi'] == 4000}
    tests = fits['ProjSTG', 'EE']
    M.add('residSTGEEMaxBin', pct(tests['max_bin_residual'], 2))
    M.add('residSTGEEFitP', sci(tests['p'], 0))
    M.add('residEEFitPRange', rng(*sorted(fits[p, 'EE']['p'] for p in ('ProjZEA', 'ProjSFL')), 3))
    integer = {r['projection']: r for r in d['ensemble_tests']
               if r['null'] == 'new-model-bin' and r['spectrum'] == 'BB'
               and r['lo'] == 1000 and r['hi'] == 4000}
    for proj in ('ProjZEA', 'ProjSFL'):
        M.add('resid'+proj[4:]+'BBClean', uncertainty(fits[proj, 'BB']))
        M.add('resid'+proj[4:]+'BBInteger', uncertainty(integer[proj]))
        M.add('resid'+proj[4:]+'BBFitP', sci(integer[proj]['p'], 0))
    M.add('residZEABBWorst', pct(fits['ProjZEA', 'BB']['max_bin_residual'], 2))
    M.add('residZEABBIntegerWorst', pct(integer['ProjZEA']['max_bin_residual'], 2))
    M.add('residSTGTTUpperInteger', uncertainty(production_upper()))
    for name, lo in (('Upper', 4000), ('Wall', 4500)):
        removed = np.mean([find(d['map_interventions'], projection=proj, spectrum='BB',
                                lo=lo, hi=4800 if lo == 4500 else 4500)['removed_fraction']
                           for proj in ('ProjSTG', 'ProjZEA')])
        M.add('residRefine'+name+'Removed', pct(removed, 0))
    ring = find(d['ring_controls'], projection='ProjSTG', intervention='n4096_r16', lo=4500, hi=4800)
    coarse = find(d['ring_controls'], projection='ProjSTG', intervention='n4096_r8', lo=4500, hi=4800)
    M.add('residRingBDrop', pct(1-ring['bb_ee_vs_4096_r8'], 0))
    M.add('residRingEEChange', pct(abs(1-ring['ee_native_ratio']/coarse['ee_native_ratio']), 1))
    M.add('residHannWorst', pct(production_hann_worst(), 2))
    assert all(name.isalpha() for name in M)
    path = ROOT/'paper/residual_numbers.tex'
    path.write_text(M.tex('residual_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    residual_numbers()
