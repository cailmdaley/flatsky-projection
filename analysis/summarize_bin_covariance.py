"""Compare scoped local predictions with final coupled Monte Carlo statistics.

The reported maxima cover only these broad-band summaries, not every matrix
entry or an exact covariance model. No simulation-tuned correction is applied.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def local_summary(comparison, prediction):
    rows = []
    for lo, hi, width in ((2000, 3000, 100), (3500, 4800, 50)):
        for proj in ('ProjZEA', 'ProjSFL', 'ProjSTG'):
            for spec in ('tt', 'ee', 'bb'):
                if proj == 'ProjSTG' and spec == 'bb':
                    continue
                select = dict(projection=proj, spectrum=spec, width=width,
                              estimator='coupled', ell_lo=lo, ell_hi=hi)
                keep = lambda r: all(r[k] == v for k, v in select.items())
                v = next(r for r in comparison['variances'] if keep(r))
                c = next(r for r in comparison['correlations'] if keep(r) and r['separation'] == width)
                assert v['n_projected'] == 260 and v['n_control'] == 256
                p = [r for r in prediction['rows'] if r['projection'] == proj
                     and r['spectrum'] == spec and r['width'] == width and lo <= r['ell'] < hi]
                assert len(p) == (hi-lo)//50
                average = lambda key: sum(r[key] for r in p)/len(p)
                rows.append(dict(**select,
                    predicted_variance_ratio=average('fractional_variance_ratio'),
                    wide_bin_variance_ratio=average('wide_bin_ratio'),
                    measured_variance_ratio=v['fractional_variance_ratio'],
                    variance_ratio_se=v['jackknife_se'],
                    predicted_adjacent_rho=sum(r['projected_rho'][0] for r in p)/len(p),
                    measured_adjacent_rho=c['projected_rho'], adjacent_rho_se=c['projected_se']))
    return dict(rows=rows,
        worst_fractional_variance_difference=max(abs(
            r['predicted_variance_ratio']/r['measured_variance_ratio']-1) for r in rows),
        worst_adjacent_rho_difference=max(abs(
            r['predicted_adjacent_rho']-r['measured_adjacent_rho']) for r in rows),
        scope='Two broad bands, coupled auto-spectra; STG BB excluded. Not a matrix-accuracy bound.')


def main():
    root = ROOT/'results/highell'
    comparison = json.loads((root/'bin_covariance_comparison/comparison.json').read_text())
    prediction = json.loads((root/'bin_covariance_local_prediction/prediction.json').read_text())
    summary = local_summary(comparison, prediction)
    out = root/'bin_covariance_comparison'
    (out/'local_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    text = ['# Leading local prediction versus coupled Monte Carlo', '', summary['scope'], '',
            '| Range / width | Chart | Spectrum | Variance ratio, predicted | Measured | Adjacent rho, predicted | Measured |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in summary['rows']:
        text.append(f"| {r['ell_lo']}–{r['ell_hi']} / {r['width']} | {r['projection'][4:]} | {r['spectrum'].upper()} "
                    f"| {r['predicted_variance_ratio']:.3f} | {r['measured_variance_ratio']:.3f} ± {r['variance_ratio_se']:.3f} "
                    f"| {r['predicted_adjacent_rho']:.3f} | {r['measured_adjacent_rho']:.3f} ± {r['adjacent_rho_se']:.3f} |")
    (out/'local_summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
