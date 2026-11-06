"""Write the compact, auditable summary used by the paper macro writer."""
from pathlib import Path
import json
import numpy as np
import spectra as S

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')
OUT = S.ROOT/'analysis/figures/residual_anomalies/summary.json'


def estimate(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1)/np.sqrt(len(x))), nseed=len(x))


def main():
    response = np.load(ROOT/'residual_estimator_response.npz')
    production = np.load(ROOT/'pipeline_windows_production.npz')
    means = json.loads((ROOT/'residual_independent_summary.json').read_text())
    maps = json.loads((ROOT/'residual_map_upsample_summary.json').read_text())
    pure = json.loads((ROOT/'residual_pure_e_summary.json').read_text())
    result = dict(
        independent_clean=[r for r in means if r['ensemble'] == 'independent260'
                           and r['lo'] == 1000 and r['hi'] == 4000 and r['spectrum'] in ('TT', 'EE')],
        independent_low=[r for r in means if r['ensemble'] == 'independent260'
                         and r['hi'] == 500 and r['spectrum'] in ('TT', 'EE')],
        production_conditioned=[], paired=[],
        map_interventions=[r for r in maps if r['pad'] == 128 and r['lo'] >= 4000],
        pure_e=[r for r in pure if r['cutoff'] == 6000 and r['lo'] >= 4000],
        ensemble_tests=json.loads((ROOT/'residual_independent_tests.json').read_text()),
        fft_gate=json.loads((ROOT/'residual_fft_gate.json').read_text()),
        ring_controls=json.loads((ROOT/'residual_ring_summary.json').read_text()),
        ring_input_b=json.loads((ROOT/'residual_ring_input_b.json').read_text()))
    for spec in ('TT', 'EE'):
        ratios = {}
        for proj in S.PROJS:
            value = production[f'field_{spec}_{proj}_raw']
            ratios[proj] = value/response[f'{proj}_{spec}_spherical']-1
            realized = value/response[f'{proj}_{spec}_realized']-1
            for lo, hi in ((1000, 4000), (4500, 4800)):
                sel = (S.ELL >= lo) & (S.ELL < hi)
                result['production_conditioned'].append(dict(projection=proj, spectrum=spec,
                    lo=lo, hi=hi, **estimate(realized[:, sel].mean(-1))))
        sel = (S.ELL >= 1000) & (S.ELL < 4000)
        contrast = (ratios['ProjBICEP'][:, sel]-ratios['ProjZEA'][:, sel]).mean(-1)
        result['paired'].append(dict(spectrum=spec, projections=['ProjBICEP', 'ProjZEA'],
                                     lo=1000, hi=4000, **estimate(contrast)))
    # Subtracting STG remains a partial diagnostic, not a floor calibration.
    paired = []
    for proj in ('ProjZEA',):
        data = {p: [np.load(S.ROOT/'results/highell/residual_map_upsample'/f'upsample_{p}_seed{s:04d}.npz')
                    for s in range(4)] for p in (proj, 'ProjSTG')}
        model = (production[f'field_BB_{proj}_windowed']+production[f'field_BB_{proj}_alias'])*S.flat_pixwin()
        for factor in (1, 2):
            a = np.array([d[f'pad128_x{factor}_decoupled'][3] for d in data[proj]])
            b = np.array([d[f'pad128_x{factor}_decoupled'][3] for d in data['ProjSTG']])
            for lo, hi in ((4000, 4500), (4500, 4800)):
                sel = (S.ELL >= lo) & (S.ELL < hi)
                r = ((a-b)[:, sel]/model[sel]-1).mean(-1)
                paired.append(dict(projection=proj, factor=factor, lo=lo, hi=hi, **estimate(r)))
        for ds in data.values():
            for d in ds:
                d.close()
    result['partial_stg_subtraction'] = paired
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
