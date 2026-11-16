"""Noisy-inversion macros, noisy_numbers.tex.

Formats `noisy_inversion_summary`: inverting the response on noisy
SPT-3G-like lensed 1500d skies, with the scatter-weighted estimator on the
split-cross spectra.

  noisySkies                    the number of skies
  noisyCost<STG|SFL><spec>      variance of the inverse-variance-combined
                                amplitude over ZEA's (STG in TT and EE, SFL
                                in TT, EE and BB)
  noisyBBBias<ZEA|SFL>[Error]   the combined BB bias, in input lensed BB

Run:  bash env.sh python3 analysis/paper_numbers/noisy_numbers.py
          --noisy-inversion-summary <dir> --out <dir>
Out:  <out>/noisy_numbers.tex and <out>/noisy_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, f  # noqa: E402


def macros(data):
    M = Macros()
    row = {(r['projection'][4:], r['spectrum']): r for r in data['rows']
           if r['estimator'] == 'weighted' and r['spectra'] == 'cross'}
    M.add('noisySkies', str(data['nsky']))
    for proj, specs in (('STG', ('TT', 'EE')), ('SFL', ('TT', 'EE', 'BB'))):
        for spec in specs:
            r = row[proj, spec]
            n = 2 if proj == 'STG' else 1 if r['combined_over_zea'] < 10 else 0
            M.add(f'noisyCost{proj}{spec}',
                  rf"${r['combined_over_zea']:.{n}f} \pm {r['combined_over_zea_se']:.{n}f}$")
    for proj in ('ZEA', 'SFL'):
        r = row[proj, 'BB']
        M.add(f'noisyBBBias{proj}', f(r['combined_bias'], 3))
        M.add(f'noisyBBBias{proj}Error', f(r['combined_bias_se'], 3))
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'noisy_inversion_summary')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.noisy_inversion_summary/'noisy_inversion_summary.json').read_text())
    macros(data).write(args.out, 'noisy_numbers')


if __name__ == "__main__":
    main()
