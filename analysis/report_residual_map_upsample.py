"""Production-pair estimator intervention, with mandatory raw roundtrip."""
from pathlib import Path
import json
import numpy as np
import spectra as S

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')


def main():
    report = np.load(ROOT/'pipeline_windows_production.npz')
    rows, saved = [], {'ell': S.ELL}
    for proj in S.PROJS:
        paths = sorted((S.ROOT/'results/highell/residual_map_upsample').glob(f'upsample_{proj}_seed*.npz'))
        if not paths:
            paths = sorted((S.ROOT/'results/highell/residual_map_upsample__units').glob(f'*-{proj}/upsample_{proj}_seed*.npz'))
        if not paths:
            continue
        data = [np.load(path) for path in paths]
        seeds = [int(d['seed']) for d in data]
        for spec, index in [('EE', 0), ('BB', 3)]:
            floor = spec == 'BB' and proj == 'ProjSTG'
            prefix = f'field_{spec}_{proj}'
            raw = report[prefix+'_raw'][seeds]*S.flat_pixwin()
            native = np.array([d['pad0_x1_decoupled'][index] for d in data])
            np.testing.assert_allclose(native, raw, rtol=2e-9, atol=1e-25)
            print(proj, spec, 'native production roundtrip passed', flush=True)
            model = (report[f'field_EE_{proj}_windowed'] if floor else
                     report[prefix+'_windowed']+report[prefix+'_alias'])*S.flat_pixwin()
            saved[f'{proj}_{spec}_model'] = model
            for pad in (0, 64, 128):
                a, b = [np.array([d[f'pad{pad}_x{factor}_decoupled'][index] for d in data])
                        for factor in (1, 2)]
                saved[f'{proj}_{spec}_pad{pad}_native'] = a
                saved[f'{proj}_{spec}_pad{pad}_fine'] = b
                for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
                    sel = (S.ELL >= lo) & (S.ELL < hi)
                    x, y = [(v[:, sel]/model[sel] - (0 if floor else 1)).mean(-1) for v in (a, b)]
                    change = y-x
                    row = dict(projection=proj, spectrum=spec, nseed=len(data), pad=pad,
                               lo=lo, hi=hi, native=float(x.mean()), fine=float(y.mean()),
                               delta=float(change.mean()),
                               native_se=float(x.std(ddof=1)/np.sqrt(len(x))),
                               fine_se=float(y.std(ddof=1)/np.sqrt(len(y))),
                               delta_se=float(change.std(ddof=1)/np.sqrt(len(change))))
                    rows.append(row)
                    print(f'{proj} {spec} pad{pad} {lo}-{hi}: native {100*x.mean():+.5f}, fine {100*y.mean():+.5f} +/- {100*row["fine_se"]:.5f}, delta {100*change.mean():+.5f} +/- {100*row["delta_se"]:.5f} %', flush=True)
        for d in data:
            d.close()
    (ROOT/'residual_map_upsample_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    np.savez(ROOT/'residual_map_upsample_summary.npz', **saved)


if __name__ == '__main__':
    main()
