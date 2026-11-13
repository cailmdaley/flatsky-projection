"""The compact residual summary the residual macros are written from.

Selects from `residual_controls` the records the paper quotes (the
independent skies over 1000 <= l < 4000 and below 500, the 128-pixel-padded
map refinement and the band-limit-6000 pure-E control above 4000, every
ensemble test, the FFT gate and the ring controls), and adds three
computations of its own on the 20 production skies:

  production_conditioned  TT and EE over the realized-input reference, per
                          projection, over 1000-4000 and 4500-4800.
  paired                  BICEP minus ZEA on the spherical reference, the
                          same skies, over 1000-4000.
  partial_stg_subtraction ZEA's refined-map BB minus STG's, over the
                          windowed-plus-alias model, at native and doubled
                          sampling: a partial diagnostic, not a floor
                          calibration.

Run:  bash env.sh python3 analysis/summarize_residual_anomalies.py
          --residual-controls <dir> --residual-estimator-response <dir>
          --pipeline-windows <dir>
          --residual-map-upsample results/fiducial/residual_map_upsample
          --flatsky-masks results/fiducial/flatsky_masks --out <dir>
Out:  <out>/residual_summary.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import spectra as S


def estimate(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1)/np.sqrt(len(x))), nseed=len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'residual_controls', 'residual_estimator_response', 'pipeline_windows',
                 'residual_map_upsample', 'flatsky_masks')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    response = np.load(args.residual_estimator_response/'residual_estimator_response.npz')
    production = np.load(args.pipeline_windows/'pipeline_windows.npz')
    controls = json.loads((args.residual_controls/'residual_controls.json').read_text())
    (res_arcmin,) = {json.loads(q.read_text())['res_arcmin'] for q in args.flatsky_masks.glob('index_*.json')}
    ELL = production['ell']
    pix = S.flat_pixwin(ELL, res_arcmin)
    means = controls['independent_summary']
    result = dict(
        independent_clean=[r for r in means if r['ensemble'] == 'independent260'
                           and r['lo'] == 1000 and r['hi'] == 4000 and r['spectrum'] in ('TT', 'EE')],
        independent_low=[r for r in means if r['ensemble'] == 'independent260'
                         and r['hi'] == 500 and r['spectrum'] in ('TT', 'EE')],
        production_conditioned=[], paired=[],
        map_interventions=[r for r in controls['map_upsample'] if r['pad'] == 128 and r['lo'] >= 4000],
        pure_e=[r for r in controls['pure_e'] if r['cutoff'] == 6000 and r['lo'] >= 4000],
        ensemble_tests=controls['independent_tests'],
        fft_gate=controls['fft_gate'],
        ring_controls=controls['rings'],
        ring_input_b=controls['ring_input_b'])
    for spec in ('TT', 'EE'):
        ratios = {}
        for proj in S.PROJS:
            value = production[f'field_{spec}_{proj}_raw']
            ratios[proj] = value/response[f'{proj}_{spec}_spherical']-1
            realized = value/response[f'{proj}_{spec}_realized']-1
            for lo, hi in ((1000, 4000), (4500, 4800)):
                sel = (ELL >= lo) & (ELL < hi)
                result['production_conditioned'].append(dict(projection=proj, spectrum=spec,
                    lo=lo, hi=hi, **estimate(realized[:, sel].mean(-1))))
        sel = (ELL >= 1000) & (ELL < 4000)
        contrast = (ratios['ProjBICEP'][:, sel]-ratios['ProjZEA'][:, sel]).mean(-1)
        result['paired'].append(dict(spectrum=spec, projections=['ProjBICEP', 'ProjZEA'],
                                     lo=1000, hi=4000, **estimate(contrast)))
    paired = []
    proj = 'ProjZEA'
    data = {p: [np.load(path) for path in sorted(args.residual_map_upsample.glob(f'upsample_{p}_seed*.npz'))]
            for p in (proj, 'ProjSTG')}
    model = (production[f'field_BB_{proj}_windowed']+production[f'field_BB_{proj}_alias'])*pix
    for factor in (1, 2):
        a = np.array([d[f'pad128_x{factor}_decoupled'][3] for d in data[proj]])
        b = np.array([d[f'pad128_x{factor}_decoupled'][3] for d in data['ProjSTG']])
        for lo, hi in ((4000, 4500), (4500, 4800)):
            sel = (ELL >= lo) & (ELL < hi)
            r = ((a-b)[:, sel]/model[sel]-1).mean(-1)
            paired.append(dict(projection=proj, factor=factor, lo=lo, hi=hi, **estimate(r)))
    for ds in data.values():
        for d in ds:
            d.close()
    result['partial_stg_subtraction'] = paired
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'residual_summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
