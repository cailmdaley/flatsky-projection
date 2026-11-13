"""Beyond-two-point lognormal control macros, paper/beyond_two_point_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/beyond_two_point_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def beyond_two_point_numbers():
    """Signal-only lognormal controls, never survey-level parameter biases."""
    data = json.loads((ROOT/'analysis/beyond_two_point_results/summary.json').read_text())
    M = Macros()
    for chart in ('TAN', 'STG', 'ZEA'):
        for diameter, size in ((14.6, 'Small'), (30., 'Medium'), (60., 'Large')):
            row = next(r for r in data['records']
                       if r['chart'] == chart and r['diameter'] == diameter)
            for key, label in (('smooth_variance', 'Variance'),
                               ('smooth_third', 'Third'),
                               ('reduced_skewness', 'ReducedSkewness'),
                               ('wavelet_l1', 'Wavelet'),
                               ('peaks_gt_2sigma', 'Peaks')):
                stat = row['statistics'][key]
                prefix = 'ng'+chart+label+size
                digits = 3 if label == 'Wavelet' else 2 if label == 'ReducedSkewness' else 1
                M.add(prefix, pct(stat['bias'], digits))
                M.add(prefix+'Error', pct(stat['se'], digits))
                cal = next(r for r in data['calibrated_records']
                           if r['chart'] == chart and r['diameter'] == diameter)['statistics'][key]
                M.add('ngCal'+chart+label+size, pct(cal['bias'], digits))
                M.add('ngCal'+chart+label+size+'Error', pct(cal['se'], digits))
    path = ROOT/'paper/beyond_two_point_numbers.tex'
    path.write_text(M.tex('beyond_two_point_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    beyond_two_point_numbers()
