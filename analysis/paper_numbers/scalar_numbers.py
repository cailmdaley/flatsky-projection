"""Scalar-leg macros: closure of the coupling matrix and diagonal-transfer errors.

Formats the `scalar_summary` output over its summary range (1000 <= l < 4000)
on the reference disk ("Disk") and the 1500d field ("Field"), per projection:

  Scalar<fp><proj><TT|White>Ratio{Min,Max,Mean}   measured over forward model
  Scalar<fp><proj>WhiteDiagonal{Min,Max,Mean}Error  white after a TT-calibrated
                                                    diagonal transfer function
  Scalar<fp><proj>WhiteFull{Min,Max,Mean}Error      white over R @ white
  ...MeanErrorSEM                                   their standard errors
  tf<proj>{mean,worst}, tfclosure{worst,zrms}       the rounded field values
                                                    the text quotes

Run:  bash env.sh python3 analysis/paper_numbers/scalar_numbers.py
          --scalar-summary <scalar_summary dir> --out <dir>
Out:  <out>/scalar_numbers.tex and <out>/scalar_numbers.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, f, pct, rng  # noqa: E402


def macros(d):
    lo, hi = d['summary_range']
    clean = (d['ell'] >= lo) & (d['ell'] < hi)
    closure, diagonal_white, full_white = d['ratio'], d['diagonal_white'], d['full_white']
    summary_sems = {'diagonal': d['diagonal_mean_sem'], 'full': d['full_mean_sem']}
    M = Macros()
    for fi, footprint in enumerate(('Disk', 'Field')):
        for j, proj in enumerate(d['projections']):
            label = S.PROJ_LABEL[str(proj)]
            for k, sky in enumerate(('TT', 'White')):
                prefix = f'Scalar{footprint}{label}{sky}'
                M.add(prefix + 'RatioMin', f(np.min(closure[fi, k, j, clean])))
                M.add(prefix + 'RatioMax', f(np.max(closure[fi, k, j, clean])))
                M.add(prefix + 'RatioMean', f(np.mean(closure[fi, k, j, clean])))
            M.add(f'Scalar{footprint}{label}WhiteDiagonalMeanError',
                  pct(np.mean(diagonal_white[fi, j, clean] - 1)))
            for name, curve in (('Diagonal', diagonal_white), ('Full', full_white)):
                for stat, operation in (('Min', np.min), ('Max', np.max), ('Mean', np.mean)):
                    # The diagonal mean already has its established macro name.
                    if name == 'Diagonal' and stat == 'Mean':
                        continue
                    M.add(f'Scalar{footprint}{label}White{name}{stat}Error',
                          pct(operation(curve[fi, j, clean] - 1)))
            for name in ('Diagonal', 'Full'):
                M.add(f'Scalar{footprint}{label}White{name}MeanErrorSEM',
                      pct(summary_sems[name.lower()][fi, j]))
            print(footprint, label, 'TT/forward', closure[fi, 0, j, clean].mean(),
                  'white/forward', closure[fi, 1, j, clean].mean(),
                  'white TT-diagonal error', (diagonal_white[fi, j, clean] - 1).mean())
    # Rounded values quoted in the text, 1500d field over the clean range.
    error = diagonal_white[1][:, clean] - 1
    closure_error = np.abs(full_white[1][:, clean] - 1)
    for j, proj in enumerate(d['projections']):
        label = S.PROJ_LABEL[str(proj)]
        M.add(f'tf{label}mean', pct(abs(error[j].mean()), 0))
        M.add(f'tf{label}worst', pct(np.abs(error[j]).max(), 0))
    M.add('tfclosureworst', pct(closure_error.max(), 1))
    per_band = closure_error / (d['full_white_sem'][1][:, clean])
    zrms = np.sqrt((per_band**2).mean(1))
    M.add('tfclosurezrms', rng(zrms.min(), zrms.max(), 1))
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'scalar_summary')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.scalar_summary / 'scalar_summary.npz') as z:
        macros(dict(z)).write(args.out, 'scalar_numbers')


if __name__ == '__main__':
    main()
