"""Compare projected and matched homogeneous Gaussian covariance ensembles.

Raw unit-width pseudo-spectra are rebinned before decoupling, using the same
width-five response in both ensembles. Each error deletes whole skies, not
individual bin pairs. The ensembles must be independent; projection charts
within the projected ensemble need not be independent of one another.

Run through bash env.sh python3 analysis/compare_bin_covariance.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import spectra as sp
from projection_covariance import PROJS, SPECS, correlation
from validate_fine_spectra import read_matrix, decouple_many

RANGES = ((500, 2000), (2000, 3500), (3500, 4800), (500, 4800))


def load_ensemble(directories, widths):
    """Return actual coupled and decoupled bandpowers, without window division.

    A positive deterministic per-bin window cancels from correlations and
    fractional variances. Omitting it keeps the means comparable to the native
    control spectra, which already contain the projection sampling window.
    """
    arrays, seeds = {}, {}
    for proj in PROJS:
        paths = sorted(path for directory in directories
                       for path in directory.glob(f'pseudo_{proj}_seed*_mask0.npz'))
        ids = [int(path.stem.split('_seed')[1].split('_')[0]) for path in paths]
        if len(ids) != len(set(ids)):
            raise ValueError(f'{proj}: duplicate realization IDs across directories')
        if len(ids) < 4:
            raise ValueError(f'{proj}: need at least four independent skies')
        seeds[proj] = ids
        with np.load(paths[0]) as z:
            counts = z['counts'].copy()
        raw = {key: [] for key in ('tt', 'pol')}
        for path in paths:
            with np.load(path) as z:
                np.testing.assert_array_equal(z['counts'], counts)
                for key in raw:
                    raw[key].append(z[key])
        for spin, key in ((0, 'tt'), (2, 'pol')):
            matrix, old_edges = read_matrix(sp.ROOT/'results/highell/fine_field_coupling', proj, 0, spin)
            ncls = 1 if spin == 0 else 4
            for width in widths:
                edges = np.arange(0, 6001, width)
                response = sp.rebin_flat_response(matrix, old_edges, edges, counts, ncls)
                coupled = sp.rebin_pseudo(np.asarray(raw[key]), counts, edges)
                decoupled, _ = decouple_many(response, coupled)
                for spec, i in ([('tt', 0)] if spin == 0 else [('ee', 0), ('bb', 3)]):
                    for estimator, values in (('coupled', coupled), ('decoupled', decoupled)):
                        arrays[proj, width, spec, estimator] = values[:, i]
        print(f'Loaded {len(ids)} skies for {proj}', flush=True)
    return arrays, seeds


def jackknife(values, statistic):
    estimate = statistic(values)
    deleted = np.asarray([statistic(np.delete(values, i, axis=0))
                          for i in range(len(values))])
    se = np.sqrt((len(values)-1)/len(values)*np.sum((deleted-deleted.mean())**2))
    return float(estimate), float(se)


def lag_statistic(values, lag):
    # Work on just the requested range, retaining the sky as the resampling unit.
    centered = values-values.mean(0)
    cross = np.sum(centered[:, :-lag]*centered[:, lag:], axis=0)
    ss = np.sum(centered**2, axis=0)
    return np.mean(cross/np.sqrt(ss[:-lag]*ss[lag:]))


def relative_variance(a, b):
    """Ratio of per-bin fractional variances, then averaged over the range."""
    def fractional(x):
        return x.var(0, ddof=1)/x.mean(0)**2
    return np.mean(fractional(a)/fractional(b))


def independent_ratio_error(a, b):
    value, ea = jackknife(a, lambda x: relative_variance(x, b))
    _, eb = jackknife(b, lambda x: relative_variance(a, x))
    return value, float(np.hypot(ea, eb))


def compare(projected, control, widths):
    correlations, variances, means = [], [], []
    for key, a in projected.items():
        proj, width, spec, estimator = key
        b = control[key]
        ell = np.arange(width/2, 6000, width)
        for lo, hi in RANGES:
            select = (ell >= lo) & (ell < hi)
            x, y = a[:, select], b[:, select]
            identity = dict(projection=proj, spectrum=spec, width=width,
                            estimator=estimator, ell_lo=lo, ell_hi=hi,
                            n_projected=len(a), n_control=len(b))
            mean_b = y.mean(0)
            ratio, ea = jackknife(x, lambda z: np.mean(z.mean(0)/mean_b))
            _, eb = jackknife(y, lambda z: np.mean(x.mean(0)/z.mean(0)))
            means.append(dict(**identity, projected_over_control=ratio,
                              jackknife_se=float(np.hypot(ea, eb))))
            ratio, se = independent_ratio_error(x, y)
            variances.append(dict(**identity, fractional_variance_ratio=ratio,
                                  jackknife_se=se))
            for lag in (1, 2, 4):
                ra, ea = jackknife(x, lambda z: lag_statistic(z, lag))
                rb, eb = jackknife(y, lambda z: lag_statistic(z, lag))
                row = dict(**identity, separation=width*lag, npair=x.shape[1]-lag,
                           projected_rho=ra, projected_se=ea,
                           control_rho=rb, control_se=eb,
                           excess_rho=ra-rb, excess_se=float(np.hypot(ea, eb)))
                correlations.append(row)
                if width == 50 and lo == 3500 and estimator == 'decoupled':
                    print(row, flush=True)
    return dict(correlations=correlations, variances=variances, means=means)


def plot_comparison(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.rc()
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), sharex=True, sharey=True,
                             layout='constrained')
    for i, proj in enumerate(PROJS):
        for j, spec in enumerate(SPECS):
            ax = axes[i, j]
            sel = [r for r in rows['correlations'] if r['projection'] == proj
                   and r['spectrum'] == spec and r['width'] == 50
                   and r['estimator'] == 'decoupled' and r['ell_lo'] == 3500
                   and r['ell_hi'] == 4800]
            ax.axhline(0, color='.8', lw=.7)
            for source, color, marker in (('projected', ps.PROJ_COLORS[proj], 'o'),
                                           ('control', '.3', 's')):
                ax.errorbar([r['separation'] for r in sel],
                            [r[f'{source}_rho'] for r in sel],
                            yerr=[r[f'{source}_se'] for r in sel],
                            color=color, marker=marker, ms=3, lw=1,
                            label=ps.PROJ_LABEL[proj] if source == 'projected' else 'Homogeneous')
            ax.text(.95, .95, spec.upper(), transform=ax.transAxes, ha='right', va='top')
            if j == 0:
                ax.legend(frameon=False, fontsize=8)
                ax.set_ylabel(r'$\mathrm{Corr}(\widehat C_b,\widehat C_{b+\Delta b})$')
            if i == 2:
                ax.set_xlabel(r'$\Delta\ell$')
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = sp.ROOT/'results/highell'
    p.add_argument('--projected', type=Path, nargs='+',
                   default=[root/'bin_covariance_pilot', root/'bin_covariance_sims'])
    p.add_argument('--control', type=Path, nargs='+', default=[root/'bin_covariance_flat_controls'])
    p.add_argument('--widths', type=int, nargs='+', default=[30, 50, 100])
    p.add_argument('--out', type=Path, default=root/'bin_covariance_comparison')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, a_seeds = load_ensemble(args.projected, args.widths)
    b, b_seeds = load_ensemble(args.control, args.widths)
    result = compare(a, b, args.widths)
    result['inputs'] = dict(projected=[str(d) for d in args.projected],
                            control=[str(d) for d in args.control],
                            projected_seeds=a_seeds, control_seeds=b_seeds,
                            errors='delete-one-whole-sky jackknife; independent ensembles')
    (args.out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    moments = {}
    for label, ensemble in (('projected', a), ('control', b)):
        for (proj, width, spec, estimator), values in ensemble.items():
            prefix = f'{label}_{proj}_d{width}_{spec}_{estimator}'
            moments[prefix+'_mean'] = values.mean(0)
            moments[prefix+'_covariance'] = np.cov(values, rowvar=False)
            moments[prefix+'_correlation'] = correlation(values)
    np.savez_compressed(args.out/'moments.npz', **moments)
    plot_comparison(result, args.out)


if __name__ == '__main__':
    main()
