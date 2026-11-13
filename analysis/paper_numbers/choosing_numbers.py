"""Choosing-a-projection macros, paper/choosing_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/choosing_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def choosing_numbers():
    """Conditional known-shape inverse test; never a noisy BB forecast."""
    data = json.loads((ROOT/'analysis/figures/choosing_projection_summary.json').read_text())
    M = Macros()
    charts = {r['projection']: r for r in data['geometry']['charts']}
    z, stg = charts['ProjZEA'], charts['ProjSTG']
    M.add('choiceZEACentreDec', f(-z['centre'][1], 2))
    M.add('choiceZEATuneGain', pct(1-z['optimum_g2']/z['default_g2'], 1))
    M.add('choiceZEAAffineGain', pct(1-data['geometry']['affine_zea']['g2']/z['default_g2'], 1))
    M.add('choiceSTGPlaneArea', pct(stg['area_magnification']-1, 2))
    M.add('choiceSTGWhitePlane', pct(stg['sky_white_noise_plane_factor']-1, 2))
    for row in data['measurements']:
        if row['width'] != 500:
            continue
        prefix = 'choice'+row['projection'][4:]+row['spectrum']
        M.add(prefix+'VarianceRatio', f"{float(f'{row['variance_over_zea']:.2g}'):g}"
              if row['spectrum'] != 'BB' else sci(row['variance_over_zea'], 1))
        M.add(prefix+'MeanBias', sci(row['mean_bias'], 1))
        M.add(prefix+'MeanBiasError', sci(row['mean_bias_se'], 1))
    cost = json.loads((ROOT/'analysis/figures/stg_correction_variance.json').read_text())
    M.add('choiceSkies', str(cost['nsky']))
    for spec in ('TT', 'EE'):
        for name, label in (('equal', 'Equal'), ('weighted', 'Weighted'), ('direct', 'Direct')):
            row = cost[f'{spec}_{name}']
            M.add('choiceCost'+spec+label, rf"${row['ratio']:.2f} \pm {row['error']:.2f}$")
    path = ROOT/'paper/choosing_numbers.tex'
    path.write_text(M.tex('choosing_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    choosing_numbers()
