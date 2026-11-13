"""Measured beam macros, paper/beam_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/beam_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def beam_numbers():
    """Write measured beam macros without rerunning the geometry calculation."""
    data = json.loads((ROOT / 'paper' / 'beam_summary.json').read_text())
    M = Macros()
    M.add('beamMeasuredEll', f(data['ell_response'], 0))
    for footprint, name in (('disk5', 'Disk'), ('field1500d', 'Field')):
        response = data['response'][footprint]
        prefix = 'beamMeasured' + name
        M.add(prefix + 'Difference', pct(response['fractional_difference'], 3))
        M.add(prefix + 'Prediction', pct(response['prediction'], 3))
        M.add(prefix + 'SFL', f(response['sfl_mean'], 3))
        M.add(prefix + 'Sphere', f(response['sphere_mean'], 3))
        M.add(prefix + 'SectorMin', f(response['sfl_sector_min'], 3))
        M.add(prefix + 'SectorMax', f(response['sfl_sector_max'], 3))
        # Absolute response residual, in percentage points (not percent of response).
        M.add(prefix + 'ModelResidual', f(100*response['sector_model_max_abs_1000_3000'], 3))
        M.add(prefix + 'QuadrupoleSlope', f(response['quadrupole_amplitude_slope_1000_3000'], 3))
        for band, label in (('1000_2000', 'Low'), ('2000_3000', 'Mid'), ('3000_4000', 'High')):
            M.add(prefix + label + 'Difference', pct(response['bands'][band]['measured'], 3))
            M.add(prefix + label + 'Prediction', pct(response['bands'][band]['predicted'], 3))
        for projection in ('ProjZEA', 'ProjSTG'):
            diagnostic = data['b_diagnostic'][footprint + '_' + projection]
            for key, label in (('mismatch', 'Mismatch'), ('differential', 'Differential'),
                               ('cross_mismatch', 'CrossMismatch'),
                               ('mismatch_power_change', 'MismatchChange')):
                M.add(prefix + projection[4:] + label, sci(diagnostic[key], 3))
    gaps = [abs(r['bands'][b]['measured'] - r['bands'][b]['predicted'])
            for r in data['response'].values() for b in ('1000_2000', '2000_3000')]
    M.add('beamBandAgreement', f(100 * max(gaps), 3))
    assert all(name.isalpha() for name in M)
    path = ROOT / 'paper' / 'beam_numbers.tex'
    path.write_text(M.tex('beam_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    beam_numbers()
