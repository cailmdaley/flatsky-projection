"""Beyond-two-point lognormal control macros: signal-only controls, never survey-level biases.

Formats the `beyond_two_point_summary` output: per chart (TAN, STG, ZEA) and
sky-disk diameter (14.6, 30 and 60 deg, `Small`, `Medium`, `Large`), the
paired fractional shift of the smoothed variance, third moment, reduced
skewness, wavelet L1 and peak count at fixed central filter scale (`ng...`)
and after mean angular-scale calibration (`ngCal...`), each with its error.

Run:  bash env.sh python3 analysis/paper_numbers/beyond_two_point_numbers.py
          --beyond-two-point-summary <beyond_two_point_summary dir> --out <dir>
Out:  <out>/beyond_two_point_numbers.tex and <out>/beyond_two_point_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, pct  # noqa: E402


def macros(data):
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
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "beyond_two_point_summary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.beyond_two_point_summary / "beyond_two_point_summary.json").read_text())
    macros(data).write(args.out, "beyond_two_point_numbers")


if __name__ == "__main__":
    main()
