"""Noisy inversion macros, paper/noisy_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/noisy_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def noisy_numbers():
    """Inverting R on noisy SPT-3G-like lensed skies (analysis/noisy_inversion.py)."""
    data = json.loads((ROOT/'analysis/figures/noisy_inversion_summary.json').read_text())
    M = Macros()
    row = {(r['projection'][4:], r['spectrum'], r['spectra']): r
           for r in data['rows'] if r['estimator'] == 'weighted'}
    pm = lambda v, e, n=2: rf"${v:.{n}f} \pm {e:.{n}f}$"
    M.add('noisySkies', str(data['nsky']))
    for proj in ('STG', 'SFL'):
        for spec in ('TT', 'EE', 'BB'):
            r = row[proj, spec, 'cross']
            n = 2 if proj == 'STG' else 1 if r['combined_over_zea'] < 10 else 0
            M.add(f'noisyCost{proj}{spec}', pm(r['combined_over_zea'], r['combined_over_zea_se'], n))
        for spec in ('TT', 'EE'):
            r = row[proj, spec, 'signal']
            n = 2 if proj == 'STG' else 1
            M.add(f'noisyFreeCost{proj}{spec}', pm(r['combined_over_zea'], r['combined_over_zea_se'], n))
    # Biases of the inverse-variance-combined amplitude, as for the variances.
    offsets = [row[p, s, 'cross']['combined_bias'] for p in ('ZEA', 'STG', 'SFL') for s in ('TT', 'EE')]
    M.add('noisyOffsetLow', pct(min(offsets), 2))
    M.add('noisyOffsetHigh', pct(max(offsets), 2))
    for proj in ('ZEA', 'STG', 'SFL'):
        r = row[proj, 'BB', 'cross']
        M.add(f'noisyBBBias{proj}', f(r['combined_bias'], 3))
        M.add(f'noisyBBBias{proj}Error', f(r['combined_bias_se'], 3))
    low = row['SFL', 'BB', 'cross']
    low_se = low['bin_sd'][0]/np.sqrt(data['nsky'])
    M.add('noisySFLBBLowBias', f(low['bin_bias'][0], 3))
    M.add('noisySFLBBLowBiasError', f(low_se, 3))
    ratio = data['sfl_detectability']['leak_over_lensed_500_1000']
    M.add('noisySFLLeakLensedLow', f"{ratio:.0f}")
    M.add('noisySFLLeakModelError', pct(low['bin_bias'][0]/ratio, 1))
    sfl, zea = row['SFL', 'BB', 'cross'], row['ZEA', 'BB', 'cross']
    M.add('noisySFLBBScatterLow', f"{sfl['bin_sd'][0]/zea['bin_sd'][0]:.0f}")
    M.add('noisySFLBBLeakedShare', pct(row['SFL', 'BB', 'signal']['bin_sd'][0]**2/sfl['bin_sd'][0]**2, 0))
    d = data['sfl_detectability']
    M.add('noisySFLSnr', f"{d['snr_500_4000']:.0f}")
    M.add('noisySFLSnrMid', f"{d['snr_2500_3000']:.0f}")
    M.add('noisySFLSnrHigh', f"{d['snr_3000_3500']:.1f}")
    M.add('noisySFLLensedEll', f"{50*round(d['ell_leak_below_lensed']/50):.0f}")
    path = ROOT/'paper/noisy_numbers.tex'
    path.write_text(M.tex('noisy_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    noisy_numbers()
