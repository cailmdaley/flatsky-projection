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
        assert len(paths) == 4, (proj, 'requires four canonical pure-E controls')
        data = [np.load(p) for p in paths]
        assert sorted(int(d['seed']) for d in data) == list(range(4))
        ell = data[0]['ell']
        print(proj, 'seeds', len(data), flush=True)
        for cutoff in (4000, 6000):
            coarse = np.array([z[f'cut{cutoff}_x1_decoupled'][3] for z in data])
            fine = np.array([z[f'cut{cutoff}_x2_decoupled'][3] for z in data])
            # Dimensionless leakage normalized to the same periodic native E
            # power, keeping coarse/fine cancellation exact in numerator.
            ee = np.array([z[f'cut6000_x1_periodic'][0] for z in data])
            # Compare dimensionful native spectra before any window division.
            prod_bb = production[f'field_BB_{proj}_raw'].mean(0)*S.flat_pixwin()
            prod_model = (np.zeros_like(S.ELL) if proj == 'ProjSTG' else
                          production[f'field_BB_{proj}_windowed']+production[f'field_BB_{proj}_alias'])*S.flat_pixwin()
            clean = (ell >= 1000) & (ell < 4800)
            for d in data:
                for factor in (1, 2):
                    periodic = d[f'cut{cutoff}_x{factor}_periodic']
                    assert abs(periodic[3, clean]).max()/periodic[0, clean].mean() < 1e-12
            assert abs(fine[:, clean]).mean() < 1e-4*abs(coarse[:, clean]).mean()
            for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
                sel = (ell >= lo) & (ell < hi)
                c = (coarse[:, sel]/ee[:, sel]).mean(-1)
                f = (fine[:, sel]/ee[:, sel]).mean(-1)
                removed = ((coarse-fine)[:, sel]/ee[:, sel]).mean(-1)
                row = dict(projection=proj, nseed=len(data), cutoff=cutoff, lo=lo, hi=hi,
                           native=float(c.mean()), fine=float(f.mean()), removed=float(removed.mean()),
                           native_se=float(c.std(ddof=1)/np.sqrt(len(c))),
                           fine_se=float(f.std(ddof=1)/np.sqrt(len(f))),
                           planar_native=float(coarse[:, sel].mean()),
                           planar_removed_native=float((coarse-fine)[:, sel].mean()),
                           production_bb_native=float(prod_bb[sel].mean()),
                           production_excess_native=float((prod_bb-prod_model)[sel].mean()),
                           planar_over_production_excess=float(coarse[:, sel].mean()/(prod_bb-prod_model)[sel].mean()))
                rows.append(row)
                print(f' cut{cutoff} {lo}-{hi}: native {c.mean():.6g} +/- {row["native_se"]:.2g}, fine {f.mean():.6g} +/- {row["fine_se"]:.2g}, removed/native {removed.mean()/c.mean():.6g}; planar/native-production-excess {row["planar_over_production_excess"]:.6g}', flush=True)
        for d in data:
            d.close()
    (ROOT/'residual_pure_e_summary.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
