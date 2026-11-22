"""Residual macros of the 1500d field, residual_numbers.tex.

Formats `residual_summary` over its band, 1000 <= l < 4000:

  resid<TT|EE>CleanWorst    the largest |band-mean residual| of the independent
                            skies on the spherical reference (the model read at
                            k = l + 1/2), over SFL, STG, ZEA
  residTTEEIntegerWorst     the same over TT and EE on the integer-bin reference
  residSTGEESpherical       STG's EE band mean on the spherical reference
  residEEMaxBin             the largest single-bin EE residual on the spherical
                            reference, over SFL, STG, ZEA
  residEEFitPMax            the largest Hotelling p-value of EE on that
                            reference, over SFL, STG, ZEA
  resid<ZEA|SFL>BBInteger   the BB band-mean residual on the integer-bin model
  residZEABBSpherical       ZEA's BB band-mean residual on the spherical one
  residBICEPZEAEE           BICEP minus ZEA in EE on the 20 production skies

Run:  bash env.sh python3 analysis/paper_numbers/residual_numbers.py
          --residual-summary <dir> --out <dir>
Out:  <out>/residual_numbers.tex and <out>/residual_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, pct  # noqa: E402


def uncertainty(row, mean='mean', se='se'):
    return rf"$({100 * row[mean]:+.3f}\pm{100 * row[se]:.3f})\%$"


def macros(d):
    M = Macros()

    def test(proj, spec, null):
        (row,) = [r for r in d['ensemble_tests'] if r['projection'] == proj
                  and r['spectrum'] == spec and r['null'] == 'new-model-' + null]
        return row

    for spec in ('TT', 'EE'):
        worst = max(abs(r['spherical_residual']) for r in d['independent_clean']
                    if r['spectrum'] == spec)
        M.add('resid' + spec + 'CleanWorst', pct(worst, 2))
    M.add('residTTEEIntegerWorst', pct(max(abs(r['bin_residual'])
                                           for r in d['independent_clean']), 2))
    M.add('residSTGEESpherical', uncertainty(test('ProjSTG', 'EE', 'spherical')))
    projs = ('ProjSFL', 'ProjSTG', 'ProjZEA')
    M.add('residEEMaxBin', pct(max(test(p, 'EE', 'spherical')['max_bin_residual']
                                   for p in projs), 2))
    M.add('residEEFitPMax', f"{max(test(p, 'EE', 'spherical')['p'] for p in projs):.3f}")
    for proj in ('ProjZEA', 'ProjSFL'):
        M.add('resid' + proj[4:] + 'BBInteger', uncertainty(test(proj, 'BB', 'bin')))
    M.add('residZEABBSpherical', uncertainty(test('ProjZEA', 'BB', 'spherical')))
    (paired,) = [r for r in d['paired'] if r['spectrum'] == 'EE']
    M.add('residBICEPZEAEE', uncertainty(paired))
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'residual_summary')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    d = json.loads((args.residual_summary / 'residual_summary.json').read_text())
    macros(d).write(args.out, 'residual_numbers')


if __name__ == '__main__':
    main()
