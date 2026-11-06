"""Covariance requires the amplitude response, not just its squared row sums.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash env.sh python3 analysis/projection_covariance.py
Outputs go to results/highell/bin_correlation_diagnostic, separate from production.
The R Sigma R^T curves are explicitly a rejected approximation, not predictions.
"""
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJS = ('ProjZEA', 'ProjSFL', 'ProjSTG')
SPECS = ('tt', 'ee', 'bb')


def quadratic_moments(amplitude, variance, estimators):
    """Exact Gaussian means and covariance for real quadratic estimators.

    y = A x, E[x x^T] = diag(variance), q_b = y^T Q_b y.
    The real representation includes all independent degrees of freedom;
    no assumption about independent complex Fourier cells is made.
    """
    c = (amplitude * variance[None, :]) @ amplitude.T
    means = np.einsum('bij,ji->b', estimators, c)
    qc = estimators @ c
    return means, 2 * np.einsum('bij,cji->bc', qc, qc)


def mean_response(amplitude, estimators, input_groups):
    """Response of mean quadratic estimators to independent input powers."""
    return np.asarray([
        [np.einsum('ij,ji', q, amplitude[:, group] @ amplitude[:, group].T)
         for group in input_groups] for q in estimators])


def counterexample():
    """Same R, two different covariances, even with orthogonal A."""
    groups = [np.array([0, 1]), np.array([2, 3])]
    q = np.asarray([np.diag([.5, .5, 0, 0]), np.diag([0, 0, .5, .5])])
    a = np.eye(4)[[0, 2, 1, 3]]
    b = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                  [1, 0, -1, 0], [0, 1, 0, -1]]) / np.sqrt(2)
    variance = np.array([1., 1., 4., 4.])
    ra, rb = [mean_response(x, q, groups) for x in (a, b)]
    ma, ca = quadratic_moments(a, variance, q)
    mb, cb = quadratic_moments(b, variance, q)
    np.testing.assert_allclose(ra, rb)
    np.testing.assert_allclose(ma, mb)
    np.testing.assert_allclose(ca, np.diag([8.5, 8.5]))
    np.testing.assert_allclose(cb, [[6.25, 2.25], [2.25, 6.25]])
    return dict(response=ra.tolist(), mean=ma.tolist(),
                permutation_covariance=ca.tolist(), mixing_covariance=cb.tolist(),
                invalid_R_cov_Rt=(ra @ np.diag([1., 16.]) @ ra.T).tolist())


def correlation(data):
    c = np.cov(data, rowvar=False)
    sd = np.sqrt(np.diag(c))
    return c / sd[:, None] / sd[None, :]


def lag_mean(data, ell, lo, hi, lag=1):
    r = correlation(data)
    sel = (ell[:-lag] >= lo) & (ell[lag:] < hi)
    return np.diag(r, lag)[sel].mean()


def jackknife_lag(data, ell, lo, hi, lag=1):
    """Delete entire sky realizations, preserving all inter-bin dependence."""
    n = len(data)
    estimates = np.array([lag_mean(np.delete(data, i, axis=0), ell, lo, hi, lag)
                          for i in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((estimates - estimates.mean())**2))
    return dict(rho=float(lag_mean(data, ell, lo, hi, lag)),
                jackknife_se=float(se), nseed=n, npair=int(np.sum(
                    (ell[:-lag] >= lo) & (ell[lag:] < hi))))


def measure(out, extra=None):
    source = ROOT/'results/highell/fine_validation/field_bandpowers.npz'
    with np.load(source) as z:
        arrays = {key: z[key] for key in z.files}
    if extra:
        # Extra MC files store raw unit-width pseudos. Rebin/decouple identically
        # to the existing finest-spectrum product before making this comparison.
        from validate_fine_spectra import read_matrix, decouple_many
        import spectra as sp
        for proj in PROJS:
            paths = sorted(path for directory in extra
                           for path in directory.glob(f'pseudo_{proj}_seed*_mask0.npz'))
            raw = [np.load(path) for path in paths]
            counts = raw[0]['counts']
            for width in (30, 50, 100):
                edges = np.arange(0, 6001, width)
                for spin, key in ((0, 'tt'), (2, 'pol')):
                    matrix, old_edges = read_matrix(ROOT/'results/highell/fine_field_coupling', proj, 0, spin)
                    matrix = sp.rebin_flat_response(matrix, old_edges, edges, counts, 1 if spin == 0 else 4)
                    cl, _ = decouple_many(matrix, sp.rebin_pseudo(np.array([r[key] for r in raw]), counts, edges))
                    for spec, index in ([('tt', 0)] if spin == 0 else [('ee', 0), ('bb', 3)]):
                        # A deterministic positive pixel window cancels from rho.
                        arrays[f'{proj}_d{width}_{spec}'] = cl[:, index]
            for r in raw:
                r.close()
    rows = []
    for proj in PROJS:
        for width in (30, 50, 100):
            ell = np.arange(width/2, 6000, width)
            for spec in SPECS:
                data = arrays[f'{proj}_d{width}_{spec}']
                for lo, hi in ((500, 2000), (2000, 3500), (3500, 4800), (500, 4800)):
                    for lag in (1, 2, 4):
                        row = dict(projection=proj, spectrum=spec, width=width,
                                   ell_lo=lo, ell_hi=hi, separation=lag*width,
                                   **jackknife_lag(data, ell, lo, hi, lag))
                        rows.append(row)
                        if width == 50 and lag == 1:
                            print(row, flush=True)
    (out/'measured.json').write_text(json.dumps(rows, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.rc()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True, layout='constrained')
    for spec, ax in zip(SPECS, axes):
        ax.axhline(0, color='.7', lw=.7)
        for proj in PROJS:
            # Three non-overlapping ranges; the overall-range point is not shown.
            selected = [r for r in rows if r['projection'] == proj and r['spectrum'] == spec
                        and r['width'] == 50 and r['separation'] == 50
                        and (r['ell_lo'], r['ell_hi']) in [(500, 2000), (2000, 3500), (3500, 4800)]]
            ax.errorbar([(r['ell_lo']+r['ell_hi'])/2 for r in selected],
                        [r['rho'] for r in selected], yerr=[r['jackknife_se'] for r in selected],
                        color=ps.PROJ_COLORS[proj], marker='o', ms=3,
                        label=ps.PROJ_LABEL[proj], lw=1)
        ax.text(.06, .95, spec.upper(), transform=ax.transAxes, va='top')
        ax.set_xlabel(r'$\ell$')
    axes[0].set_ylabel(r'$\mathrm{Corr}(\widehat C_b,\widehat C_{b+1})$')
    axes[0].legend(frameon=False)
    fig.savefig(out/'adjacent.png', dpi=180)
    plt.close(fig)
    return rows


def invalid_radial_propagation(out):
    """Quantify the tempting, but incorrect, R diag(Var C_l) R^T recipe."""
    from field_1500d import pipeline_moments
    import shear_leakage as sl
    import spectra as sp
    import theory
    cache = out/'moments.npz'
    if cache.exists():
        with np.load(cache) as z:
            mom = {key: z[key] for key in z.files}
    else:
        mom = pipeline_moments(nside=512)
        np.savez(cache, **mom)
    th = theory.load()
    sky_ell = np.arange(6001, dtype=float)
    sky_variance = {spec: 2*th[f'unlensed_{spec}'][:6001]**2/(2*sky_ell+1)
                    for spec in ('TT', 'EE')}
    rows, arrays = [], {}
    ell = np.arange(25, 6000, 50.)
    # Average response rows over the output annulus before propagating.
    # A single ell has quadrature spikes that are irrelevant to a bandpower.
    fine_ell = np.arange(2.5, 6000, 5.)
    for proj in PROJS:
        fine_r = sl.coupling_matrix(mom, 0, sp.PROJS.index(proj), fine_ell, sky_ell,
                                    band_limit=6000, n_chi=256)
        r = {key: val.reshape(120, 10, -1).mean(axis=1) for key, val in fine_r.items()}
        del fine_r
        for spec, block, source in [('tt', 'TT<-TT', 'TT'), ('ee', 'EE<-EE', 'EE'), ('bb', 'BB<-EE', 'EE')]:
            matrix = r[block]
            cov = (matrix*sky_variance[source][None, :]) @ matrix.T
            sd = np.sqrt(np.diag(cov))
            corr = cov/sd[:, None]/sd[None, :]
            arrays[f'{proj}_{spec}_rho'] = corr
            for lo, hi in ((500, 2000), (2000, 3500), (3500, 4800)):
                sel = (ell[:-1] >= lo) & (ell[1:] < hi)
                row = dict(projection=proj, spectrum=spec, ell_lo=lo, ell_hi=hi,
                           invalid_adjacent_rho=float(np.diag(corr, 1)[sel].mean()))
                rows.append(row)
                print('INVALID radial propagation', row, flush=True)
    np.savez(out/'invalid_radial.npz', ell=ell, **arrays)
    (out/'invalid_radial.json').write_text(json.dumps(rows, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path,
                   help='Defaults to separate production20 or extra-MC diagnostic directories')
    p.add_argument('--radial', action='store_true')
    p.add_argument('--extra', type=Path, nargs='+',
                   help='Unit-width MC directories to pool (replaces the 20-sky ensemble)')
    args = p.parse_args()
    if args.out is None:
        args.out = ROOT/'results/highell'/('bin_correlation_mc_diagnostic' if args.extra
                                          else 'bin_correlation_diagnostic')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'counterexample.json').write_text(json.dumps(counterexample(), indent=2)+'\n')
    measure(args.out, args.extra)
    if args.radial:
        invalid_radial_propagation(args.out)


if __name__ == '__main__':
    main()
