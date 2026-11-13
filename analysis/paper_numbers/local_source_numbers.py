"""Compact-source response and S/N macros, paper/local_source_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/local_source_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def local_source_numbers(M=None):
    """Spherical source responses and finite-patch S/N forecasts at 15 points.

    Amplitude macros are noiseless spherical template-mismatch responses;
    SnrLoss macros are local-affine covariance forecasts. Neither measures
    an SPT catalogue bias or a maximum over its whole footprint.
    """
    import json
    path=ROOT/'results/highell/local_source_geometry/geometry.json'
    rows=json.loads(path.read_text())['rows']
    from local_source_window_covariance import models
    finite=models(rows)
    own=M is None
    if own:
        M=Macros()
    for proj in ('SFL','ZEA'):
        for kind in ('point','cluster'):
            rs=[r for r in rows if r['projection']=='Proj'+proj and r['profile']==kind and r['noise']=='cmb_white']
            M.add('source'+proj+kind.title()+'Loss',pct(max(1-r['amplitude'] for r in rs),2))
            relative_snr=[finite[r['projection'],r['ra'],r['dec'],r['profile']]['snr_relative_center_finite_patch'] for r in rs]
            M.add('source'+proj+kind.title()+'SnrLoss',pct(1-min(relative_snr),1))
    rs=[r for r in rows if r['projection']=='ProjBICEP' and r['profile']=='point' and r['noise']=='cmb_white']
    M.add('sourceBICEPPointLow',pct(min(r['amplitude']-1 for r in rs),1))
    M.add('sourceBICEPPointHigh',pct(max(r['amplitude']-1 for r in rs),1))
    M.add('sourceMetricError',sci(max(abs(r['metric_template_amplitude']-1) for r in rows),1))
    M.add('sourcePositionArcsec',f(60*max(np.linalg.norm(r['position_arcmin']) for r in rows),3))
    transferred=json.loads((ROOT/'analysis/results_disk_trend/local_sources/local_source_summary.json').read_text())
    transferred=transferred['sampling_resolution']['8192']['transferred_noise_validation']['ProjSFL']
    for kind in ('point','cluster'):
        row=transferred[kind]
        M.add('sourceSFL'+kind.title()+'Transfer',rf"${row['corner_snr']:.2f} \pm {row['corner_snr_jackknife_error']:.2f}$")
    if own:
        output=ROOT/'paper/local_source_numbers.tex'
        output.write_text(M.tex('local_source_numbers'))
        print(f'Wrote {output}')


if __name__ == "__main__":
    local_source_numbers()
