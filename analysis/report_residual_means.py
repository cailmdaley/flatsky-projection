"""Independent 260-sky check of production20 residuals, never pooling skies."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import f, t
import spectra as S
from check_bin_covariance_means import load_native

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')


def main():
    report = np.load(ROOT/'pipeline_windows_production.npz')
    estimator = np.load(ROOT/'residual_estimator_response.npz')
    new = load_native([S.ROOT/'results/highell/bin_covariance_pilot',
                       S.ROOT/'results/highell/bin_covariance_sims'])
    result = {'ell': S.ELL}
    rows, tests = [], []
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
        radial_model = estimator[f'{proj}_{spec}_estimator'] if not floor else model
        spherical = estimator[f'{proj}_{spec}_spherical'] if not floor else model
        print(proj, spec, 'floor/EE' if floor else 'fractional residual', flush=True)
        for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
            sel = (S.ELL >= lo) & (S.ELL < hi)
            for name, value in [('production20', old), ('independent260', arr)]:
                ratio = (value[:, sel]/model[sel] - (0 if floor else 1)).mean(-1)
                radial = (value[:, sel]/radial_model[sel] - (0 if floor else 1)).mean(-1)
                curved = (value[:, sel]/spherical[sel] - (0 if floor else 1)).mean(-1)
                rows.append(dict(projection=proj, spectrum=spec, ensemble=name, lo=lo, hi=hi,
                                 bin_residual=float(ratio.mean()), bin_se=float(ratio.std(ddof=1)/np.sqrt(len(ratio))),
                                 spherical_residual=float(curved.mean()), spherical_se=float(curved.std(ddof=1)/np.sqrt(len(curved)))))
                print(f'{lo}-{hi} {name}: bin {100*ratio.mean():+.5f} +/- {100*ratio.std(ddof=1)/np.sqrt(len(ratio)):.5f} %, radial-response {100*radial.mean():+.5f} +/- {100*radial.std(ddof=1)/np.sqrt(len(radial)):.5f} %, spherical {100*curved.mean():+.5f} +/- {100*curved.std(ddof=1)/np.sqrt(len(curved)):.5f} %', flush=True)
        if not floor:
            for lo, hi in ((100, 1000), (500, 1000), (1000, 4000), (4000, 4800), (4500, 4800)):
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
                tests.append(dict(projection=proj, spectrum=spec, null='old-new', lo=lo, hi=hi,
                                  t2=stat, dimension=dim, p=float(p)))
                print(f'old-new {lo}-{hi}: T2 {stat:.3f}, dim {dim}, covariance-corrected F p {p:.5g}', flush=True)
                # Remove only the predicted all-sky input-mean difference, not
                # the finite-footprint realization scatter.
                if spec in ('TT', 'EE'):
                    offset = (estimator[f'{proj}_{spec}_realized']-spherical)/model
                    dc = d-offset[sel]
                    tc = float(dc @ np.linalg.solve(cov, dc)/(1/len(x)+1/len(y)))
                    pc = float(f.sf((n-dim)*tc/((n-1)*dim), dim, n-dim))
                    tests.append(dict(projection=proj, spectrum=spec, null='old-new-input-conditioned',
                                      lo=lo, hi=hi, t2=tc, dimension=dim, p=pc))
                    print(f'old-new input-conditioned {lo}-{hi}: T2 {tc:.3f}, F p {pc:.5g}', flush=True)
                for label, theory in [('bin', model), ('radial', radial_model), ('spherical', spherical)]:
                    residual = arr[:, sel]/theory[sel]-1
                    mean = residual.mean(0)
                    covr = np.cov(residual, rowvar=False)
                    t2 = float(n*mean @ np.linalg.solve(covr, mean))
                    prob = f.sf((n-dim)*t2/((n-1)*dim), dim, n-dim)
                    # A 1-D test of the per-sky band mean complements the
                    # full-covariance test of structured bin residuals.
                    band = residual.mean(-1)
                    se = band.std(ddof=1)/np.sqrt(n)
                    p1 = float(2*t.sf(abs(band.mean()/se), n-1))
                    tests.append(dict(projection=proj, spectrum=spec, null='new-model-'+label,
                                      lo=lo, hi=hi, t2=t2, dimension=dim, p=float(prob),
                                      mean=float(band.mean()), se=float(se), band_mean_p=p1,
                                      max_bin_residual=float(max(abs(mean)))))
                    print(f'new-model {label} {lo}-{hi}: T2 {t2:.3f}, dim {dim}, F p {prob:.5g}, band-mean p {p1:.5g}, max residual {100*max(abs(mean)):.4f} %', flush=True)
    for spec in ('TT', 'EE'):
        select = (S.ELL >= 1000) & (S.ELL < 4000)
        ratios = {}
        for proj in S.PROJS:
            value = report[f'field_{spec}_{proj}_raw']
            ratios[proj] = value/estimator[f'{proj}_{spec}_spherical']-1
            conditioned = value/estimator[f'{proj}_{spec}_realized']-1
            for lo, hi in ((1000, 4000), (4500, 4800)):
                sel = (S.ELL >= lo) & (S.ELL < hi)
                r = conditioned[:, sel].mean(-1)
                print(f'realized-input {proj} {spec} {lo}-{hi}: {100*r.mean():+.5f} +/- {100*r.std(ddof=1)/np.sqrt(len(r)):.5f} %', flush=True)
        delta = (ratios['ProjBICEP'][:, select]-ratios['ProjZEA'][:, select]).mean(-1)
        print(f'paired spherical BICEP-ZEA {spec} 1000-4000: {100*delta.mean():+.5f} +/- {100*delta.std(ddof=1)/np.sqrt(len(delta)):.5f} %', flush=True)
    np.savez(ROOT/'residual_independent_means.npz', **result)
    (ROOT/'residual_independent_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    (ROOT/'residual_independent_tests.json').write_text(json.dumps(tests, indent=2)+'\n')


if __name__ == '__main__':
    main()
