"""Compact-source response and S/N macros: spherical responses and finite-patch forecasts.

Formats the `local_source_summary` output. Amplitude macros (`...Loss`, the
BICEP range, the metric-template error, the position bias) are noiseless
spherical template-mismatch responses over the 15 positions; `...SnrLoss`
macros are local-affine finite-patch covariance forecasts; `...Transfer` the
corner S/N relative to the centre from the 64 coarse CMB skies calibrated by
their paired repeats at the finest Nside. None measures an SPT catalogue bias or a
maximum over the whole footprint.

Run:  bash env.sh python3 analysis/paper_numbers/local_source_numbers.py
          --local-source-summary <local_source_summary dir> --out <dir>
Out:  <out>/local_source_numbers.tex and <out>/local_source_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, f, pct, sci  # noqa: E402


def macros(d):
    geo = d['geometry']
    M = Macros()
    for proj in ('SFL', 'ZEA'):
        for kind in ('point', 'cluster'):
            g = geo['Proj' + proj][kind + '_cmb_white']
            M.add('source' + proj + kind.title() + 'Loss', pct(1 - g['amplitude_range'][0], 2))
            M.add('source' + proj + kind.title() + 'SnrLoss',
                  pct(1 - g['finite_patch_snr_relative_center_range'][0], 1))
    lo, hi = geo['ProjBICEP']['point_cmb_white']['amplitude_range']
    M.add('sourceBICEPPointLow', pct(lo - 1, 1))
    M.add('sourceBICEPPointHigh', pct(hi - 1, 1))
    cells = [c for proj in geo.values() for c in proj.values()]
    M.add('sourceMetricError', sci(max(c['metric_amplitude_max_error'] for c in cells), 1))
    M.add('sourcePositionArcsec', f(max(c['position_bias_max_arcsec'] for c in cells), 3))
    finest = max(d['sampling_resolution'], key=int)
    transferred = d['sampling_resolution'][finest]['transferred_noise_validation']['ProjSFL']
    for kind in ('point', 'cluster'):
        row = transferred[kind]
        M.add('sourceSFL' + kind.title() + 'Transfer',
              rf"${row['corner_snr']:.2f} \pm {row['corner_snr_jackknife_error']:.2f}$")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "local_source_summary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = json.loads((args.local_source_summary / "local_source_summary.json").read_text())
    macros(d).write(args.out, "local_source_numbers")


if __name__ == "__main__":
    main()
