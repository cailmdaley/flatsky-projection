"""Compare projected and matched homogeneous Gaussian covariance ensembles.

The 260 projected 1500d skies (`bin_covariance_sims`) against the 256
homogeneous controls (`bin_covariance_flat_controls`), per chart, spectrum and
bin width. Raw unit-width pseudo-spectra are rebinned before decoupling, using
the same width-five response (`fine_coupling_matrices`) in both ensembles.
Each error deletes whole skies, not individual bin pairs. The ensembles must
be independent; projection charts within the projected ensemble need not be
independent of one another.

`--widths` are the bin widths compared over the ell ranges `RANGES`;
`--binwise-widths` the ones also compared bin pair by bin pair, the first of
which the two diagnostic plots show. Each is a bare width or a
`bandpower_width` option id (`d50`).

Run:  bash env.sh python3 analysis/compare_bin_covariance.py
          --bin-covariance-sims results/fiducial/bin_covariance_sims
          --bin-covariance-flat-controls results/fiducial/bin_covariance_flat_controls
          --fine-coupling-matrices results/fiducial/fine_coupling_matrices
          --widths d50 100 --binwise-widths d50 100 --out <dir>
Out:  <out>/bin_covariance_comparison.json, <out>/moments.npz (per-ensemble
      means, covariances and correlations), comparison.png and binwise.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.linalg import get_lapack_funcs, lu_factor, lu_solve

import spectra as sp
from _decisions import parse_delta_ell

PROJS = ('ProjZEA', 'ProjSFL', 'ProjSTG')
SPECS = ('tt', 'ee', 'bb')
# The 2018 SPT spectra stop at 3000; keep that comparison separate from our
# extension to 4800 rather than calling the latter a reproduction of SPT.
RANGES = ((500, 2000), (2000, 3000), (2000, 3500), (3500, 4800), (500, 4800))


def read_matrix(directory, proj, footprint, spin):
    """Binned NaMaster response and its bin edges, from a workspace FITS file."""
    path = Path(directory) / f'w{spin}{spin}_{proj}_{footprint}.fits'
    with fits.open(path) as h:
        edges = np.r_[h['BINS_SUMMARY'].data.field(0), h['BINS_SUMMARY'].data.field(1)[-1]]
        return h['MCM_BINNED'].data.copy(), edges


def decouple_many(matrix, coupled):
    """One factorization for all seeds, NaMaster's bin-major FITS ordering."""
    lu, piv = lu_factor(matrix)
    rcond, info = get_lapack_funcs('gecon', (lu,))(lu, np.linalg.norm(matrix, 1))
    if info or rcond == 0:
        raise np.linalg.LinAlgError('singular response')
    nseed, ncls, nband = coupled.shape
    rhs = coupled.transpose(2, 1, 0).reshape(nband*ncls, nseed)
    cl = lu_solve((lu, piv), rhs).reshape(nband, ncls, nseed).transpose(2, 1, 0)
    return cl, float(1/rcond)


def correlation(data):
    c = np.cov(data, rowvar=False)
    sd = np.sqrt(np.diag(c))
    return c / sd[:, None] / sd[None, :]


def load_ensemble(directory, workspaces, widths, projs=PROJS):
    """Return actual coupled and decoupled bandpowers, without window division.

    A positive deterministic per-bin window cancels from correlations and
    fractional variances. Omitting it keeps the means comparable to the native
    control spectra, which already contain the projection sampling window.
    Bins run from zero to the pseudo-spectra's stored lmax.
    """
    arrays, seeds = {}, {}
    for proj in projs:
        paths = sorted(Path(directory).glob(f'pseudo_{proj}_seed*_mask0.npz'))
        ids = [int(path.stem.split('_seed')[1].split('_')[0]) for path in paths]
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
            matrix, old_edges = read_matrix(workspaces, proj, sp.FIELD, spin)
            ncls = 1 if spin == 0 else 4
            for width in widths:
                edges = np.arange(0, len(counts)+1, width)
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


def binwise_lag(values, lag):
    centered = values-values.mean(0)
    cross = np.sum(centered[:, :-lag]*centered[:, lag:], axis=0)
    ss = np.sum(centered**2, axis=0)
    return cross/np.sqrt(ss[:-lag]*ss[lag:])


def lag_statistic(values, lag):
    # Work on just the requested range, retaining the sky as the resampling unit.
    return np.mean(binwise_lag(values, lag))


def binwise_comparison(projected, control, widths):
    """Individual pairs, so a range average cannot conceal localized structure.

    Errors are pointwise, not simultaneous confidence limits for the curve.
    """
    rows = []
    for (proj, width, spec, estimator), a in projected.items():
        if width not in widths:
            continue
        b = control[proj, width, spec, estimator]
        ell = np.arange(a.shape[1])*width+width/2
        select = (ell >= 500) & (ell < 4800)
        for lag in (1, 2, 4):
            result = dict(projection=proj, spectrum=spec, estimator=estimator,
                          width=width, separation=lag*width,
                          ell_left=ell[select][:-lag].tolist(),
                          ell_right=ell[select][lag:].tolist())
            for label, x in (('projected', a[:, select]), ('control', b[:, select])):
                rho = binwise_lag(x, lag)
                deleted = np.array([binwise_lag(np.delete(x, i, axis=0), lag)
                                    for i in range(len(x))])
                se = np.sqrt((len(x)-1)/len(x)*np.sum(
                    (deleted-deleted.mean(0))**2, axis=0))
                result[label+'_rho'] = rho.tolist()
                result[label+'_se'] = se.tolist()
            rows.append(result)
    return rows


def relative_variance(a, b):
    """Ratio of per-bin fractional variances, then averaged over the range."""
    def fractional(x):
        return x.var(0, ddof=1)/x.mean(0)**2
    return np.mean(fractional(a)/fractional(b))


def independent_ratio_error(a, b):
    value, ea = jackknife(a, lambda x: relative_variance(x, b))
    _, eb = jackknife(b, lambda x: relative_variance(a, x))
    return value, float(np.hypot(ea, eb))


def compare(projected, control):
    correlations, variances, means = [], [], []
    for key, a in projected.items():
        proj, width, spec, estimator = key
        b = control[key]
        ell = np.arange(a.shape[1])*width+width/2
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
    return dict(correlations=correlations, variances=variances, means=means)


def plot_comparison(rows, width, out):
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
                   and r['spectrum'] == spec and r['width'] == width
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
                ax.legend(frameon=False, fontsize=8, loc='upper left')
                ax.set_ylabel(r'$\mathrm{Corr}(\widehat C_b,\widehat C_{b+\Delta b})$')
            if i == 2:
                ax.set_xlabel(r'$\Delta\ell$')
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)


def plot_binwise(rows, width, out):
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.rc()
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.7), sharex=True, sharey=True,
                             layout='constrained')
    for i, proj in enumerate(('ProjZEA', 'ProjSFL')):
        for j, spec in enumerate(SPECS):
            ax = axes[i, j]
            row = next(r for r in rows if r['projection'] == proj and r['spectrum'] == spec
                       and r['estimator'] == 'decoupled' and r['width'] == width
                       and r['separation'] == width)
            ell = np.asarray(row['ell_left'])+width/2
            ax.axhline(0, color='.8', lw=.7)
            for source, color in (('projected', ps.PROJ_COLORS[proj]), ('control', '.3')):
                rho, se = np.array(row[source+'_rho']), np.array(row[source+'_se'])
                ax.plot(ell, rho, color=color, lw=.8,
                        label=ps.PROJ_LABEL[proj] if source == 'projected' else 'Homogeneous')
                ax.fill_between(ell, rho-se, rho+se, color=color, alpha=.15, linewidth=0)
            ax.text(.95, .95, spec.upper(), transform=ax.transAxes, ha='right', va='top')
            if j == 0:
                ax.legend(frameon=False, fontsize=8, loc='upper left')
                ax.set_ylabel(r'$\mathrm{Corr}(\widehat C_b,\widehat C_{b+1})$')
            if i == 1:
                ax.set_xlabel(r'$\ell$')
    fig.savefig(out/'binwise.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp.add_inputs(p, 'bin_covariance_sims', 'bin_covariance_flat_controls',
                  'fine_coupling_matrices')
    p.add_argument('--widths', type=parse_delta_ell, nargs='+', required=True)
    p.add_argument('--binwise-widths', type=parse_delta_ell, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, a_seeds = load_ensemble(args.bin_covariance_sims, args.fine_coupling_matrices, args.widths)
    b, b_seeds = load_ensemble(args.bin_covariance_flat_controls, args.fine_coupling_matrices,
                               args.widths)
    result = compare(a, b)
    result['binwise'] = binwise_comparison(a, b, args.binwise_widths)
    result['inputs'] = dict(projected=[str(args.bin_covariance_sims)],
                            control=[str(args.bin_covariance_flat_controls)],
                            projected_seeds=a_seeds, control_seeds=b_seeds,
                            errors='delete-one-whole-sky jackknife; independent ensembles')
    (args.out/'bin_covariance_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    moments = {}
    for label, ensemble in (('projected', a), ('control', b)):
        for (proj, width, spec, estimator), values in ensemble.items():
            prefix = f'{label}_{proj}_d{width}_{spec}_{estimator}'
            moments[prefix+'_mean'] = values.mean(0)
            moments[prefix+'_covariance'] = np.cov(values, rowvar=False)
            moments[prefix+'_correlation'] = correlation(values)
    np.savez_compressed(args.out/'moments.npz', **moments)
    plot_comparison(result, args.binwise_widths[0], args.out)
    plot_binwise(result['binwise'], args.binwise_widths[0], args.out)


if __name__ == '__main__':
    main()
