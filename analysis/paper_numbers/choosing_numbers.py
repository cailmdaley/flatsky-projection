"""Choosing-a-projection macros, choosing_numbers.tex.

Formats `choosing_projection` and `stg_correction_variance`: a conditional
known-shape inverse test, never a noisy BB forecast.

  choiceZEACentreDec            declination of the centre minimizing ZEA's <g^2>
  choiceZEA{Tune,Affine}Gain    <g^2> saved by that centre, and by a further
                                determinant-one stretch
  choiceSTG{PlaneArea,WhitePlane}  STG's area magnification and white-noise
                                   plane factor, over one
  choice<proj><spec>VarianceRatio, MeanBias[Error]
                                the 500-wide-band inverse against ZEA's
  choiceSkies, choiceCost<TT|EE><Equal|Weighted|Direct>
                                the cost of correcting STG's magnification
                                on the covariance skies

Run:  bash env.sh python3 analysis/paper_numbers/choosing_numbers.py
          --choosing-projection <dir> --stg-correction-variance <dir> --out <dir>
Out:  <out>/choosing_numbers.tex and <out>/choosing_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, f, pct, sci  # noqa: E402


def macros(data, cost):
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
    M.add('choiceSkies', str(cost['nsky']))
    for spec in ('TT', 'EE'):
        for name, label in (('equal', 'Equal'), ('weighted', 'Weighted'), ('direct', 'Direct')):
            row = cost[f'{spec}_{name}']
            M.add('choiceCost'+spec+label, rf"${row['ratio']:.2f} \pm {row['error']:.2f}$")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'choosing_projection', 'stg_correction_variance')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.choosing_projection/'choosing_projection.json').read_text())
    cost = json.loads((args.stg_correction_variance/'stg_correction_variance.json').read_text())
    macros(data, cost).write(args.out, 'choosing_numbers')


if __name__ == "__main__":
    main()
