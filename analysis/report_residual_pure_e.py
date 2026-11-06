"""Report paired native/fine pure-E mask-product interventions."""
from pathlib import Path
import json
import numpy as np
import spectra as S

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')


def main():
    rows = []
    production = np.load(ROOT/'pipeline_windows_production.npz')
    for proj in S.PROJS:
        paths = sorted((S.ROOT/'results/highell/residual_pure_e_test').glob(f'pure_e_{proj}_seed*.npz'))
        if not paths:
            paths = sorted((S.ROOT/'results/highell/residual_pure_e_test__units').glob(f'*-{proj}/pure_e_{proj}_seed*.npz'))
        if not paths:
            continue
        data = [np.load(p) for p in paths]
        ell = data[0]['ell']
        print(proj, 'seeds', len(data), flush=True)
        for cutoff in (4000, 6000):
            coarse = np.array([z[f'cut{cutoff}_x1_decoupled'][3] for z in data])
            fine = np.array([z[f'cut{cutoff}_x2_decoupled'][3] for z in data])
            # Dimensionless leakage normalized to the same periodic native E
            # power, keeping coarse/fine cancellation exact in numerator.
            ee = np.array([z[f'cut6000_x1_periodic'][0] for z in data])
            floor = production[f'field_BB_ProjSTG_raw'].mean(0)*S.flat_pixwin()
            for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
                sel = (ell >= lo) & (ell < hi)
                c = (coarse[:, sel]/ee[:, sel]).mean(-1)
                f = (fine[:, sel]/ee[:, sel]).mean(-1)
                removed = ((coarse-fine)[:, sel]/ee[:, sel]).mean(-1)
                row = dict(projection=proj, nseed=len(data), cutoff=cutoff, lo=lo, hi=hi,
                           native=float(c.mean()), fine=float(f.mean()), removed=float(removed.mean()),
                           native_se=float(c.std(ddof=1)/np.sqrt(len(c))),
                           fine_se=float(f.std(ddof=1)/np.sqrt(len(f))))
                rows.append(row)
                print(f' cut{cutoff} {lo}-{hi}: native {c.mean():.6g} +/- {row["native_se"]:.2g}, fine {f.mean():.6g} +/- {row["fine_se"]:.2g}, removed/native {removed.mean()/c.mean():.6g}', flush=True)
        for d in data:
            d.close()
    (ROOT/'residual_pure_e_summary.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
