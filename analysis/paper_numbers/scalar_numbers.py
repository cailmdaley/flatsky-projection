"""Scalar-leg macros: the diagonal-transfer error and the coupling matrix's closure.

Formats the `scalar_summary` output over its summary range (1000 <= l < 4000)
on the 1500d field:

  tf<SFL|STG|ZEA>worst   the largest |error| of the white bandpower after a
                         TT-calibrated diagonal transfer function
  tfSFLmean              SFL's mean |error| of the same
  tfclosureworst         the largest |measured white / R applied to the white
                         input - 1| over the projections

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
from macros import Macros, pct  # noqa: E402


def macros(d):
    lo, hi = d['summary_range']
    clean = (d['ell'] >= lo) & (d['ell'] < hi)
    error = d['diagonal_white'][:, clean] - 1
    label = [S.PROJ_LABEL[str(p)] for p in d['projections']]
    M = Macros()
    M.add('tfSFLmean', pct(abs(error[label.index('SFL')].mean()), 0))
    for proj in ('SFL', 'STG', 'ZEA'):
        M.add(f'tf{proj}worst', pct(np.abs(error[label.index(proj)]).max(), 0))
    M.add('tfclosureworst', pct(np.abs(d['full_white'][:, clean] - 1).max(), 1))
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
