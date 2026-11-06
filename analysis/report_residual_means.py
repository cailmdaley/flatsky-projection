"""Independent 260-sky check of production20 residuals, never pooling skies."""
from pathlib import Path
import numpy as np
from scipy.stats import chi2, f
import spectra as S
from check_bin_covariance_means import load_native

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')


def main():
    report = np.load(ROOT/'pipeline_windows_production.npz')
    estimator = np.load(ROOT/'residual_estimator_response.npz')
    new = load_native([S.ROOT/'results/highell/bin_covariance_pilot',
                       S.ROOT/'results/highell/bin_covariance_sims'])
    result = {'ell': S.ELL}
    for (proj, spec), native in new.items():
        spec = spec.upper()
        prefix = f'field_{spec}_{proj}'
        old = report[prefix+'_raw']
        arr = native/S.flat_pixwin()
        assert len(arr) == 260
        result[f'{proj}_{spec}_production'] = old
        result[f'{proj}_{spec}_independent'] = arr
        # BB STG has no geometric signal, normalize to EE for floor displays.
        floor = spec == 'BB' and proj == 'ProjSTG'
        model = report[f'field_EE_{proj}_windowed'] if floor else report[prefix+'_windowed']
        exact = estimator[f'{proj}_{spec}_estimator'] if not floor else model
        spherical = estimator[f'{proj}_{spec}_spherical'] if not floor else model
        print(proj, spec, 'floor/EE' if floor else 'fractional residual', flush=True)
        for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
            sel = (S.ELL >= lo) & (S.ELL < hi)
            for name, value in [('production20', old), ('independent260', arr)]:
                ratio = (value[:, sel]/model[sel] - (0 if floor else 1)).mean(-1)
                radial = (value[:, sel]/exact[sel] - (0 if floor else 1)).mean(-1)
                curved = (value[:, sel]/spherical[sel] - (0 if floor else 1)).mean(-1)
                print(f'{lo}-{hi} {name}: bin {100*ratio.mean():+.5f} +/- {100*ratio.std(ddof=1)/np.sqrt(len(ratio)):.5f} %, radial-response {100*radial.mean():+.5f} +/- {100*radial.std(ddof=1)/np.sqrt(len(radial)):.5f} %, spherical {100*curved.mean():+.5f} +/- {100*curved.std(ddof=1)/np.sqrt(len(curved)):.5f} %', flush=True)
        if not floor:
            for lo, hi in ((100, 1000), (500, 1000), (1000, 4000)):
                sel = (S.ELL >= lo) & (S.ELL < hi)
                x, y = old[:, sel]/model[sel], arr[:, sel]/model[sel]
                cov = np.cov(y, rowvar=False)
                d = x.mean(0)-y.mean(0)
                # Test against the independently estimated covariance, not a
                # diagonal chi-square. The difference includes old-alm vs CAMB
                # mismatch, so this is not a pure sample-variance p-value.
                stat = float(d @ np.linalg.solve(cov, d)/(1/len(x)+1/len(y)))
                dim, n = len(d), len(y)
                p = f.sf((n-dim)*stat/((n-1)*dim), dim, n-dim)
                print(f'old-new {lo}-{hi}: T2 {stat:.3f}, dim {dim}, covariance-corrected F p {p:.5g}', flush=True)
    np.savez(ROOT/'residual_independent_means.npz', **result)


if __name__ == '__main__':
    main()
