"""Validate finest pseudo-spectra and diagnose field decoupling widths.

Run through env.sh with OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2.
The regression requires every paper projection, seed and mask.
The field scan rebins the width-five response exactly before inversion.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from scipy.linalg import get_lapack_funcs, lu_factor, lu_solve, LinAlgWarning

import spectra as sp


def read_pseudos(directory, proj, mask):
    paths = [directory / f'pseudo_{proj}_seed{s:04d}_mask{mask}.npz'
             for s in range(sp.NSIMS)]
    pol, tt = [], []
    for path in paths:
        with np.load(path) as z:
            pol.append(z['pol']); tt.append(z['tt'])
            counts = z['counts']
    return np.asarray(pol), np.asarray(tt), counts


def read_matrix(directory, proj, mask, spin):
    path = directory / f'w{spin}{spin}_{proj}_{mask}.fits'
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


def regression(out):
    metrics = []
    edges = np.arange(0, sp.LMAX+1, 50)
    from spt3g.mapspectra.pixel_window import pixwin_flatsky
    correction = pixwin_flatsky(sp.ELL, sp.RES_ARCMIN)**2
    for field in (False, True):
        pseudo_dir = sp.ROOT/'results/highell'/('fine_field_pseudo_spectra' if field else 'fine_pseudo_spectra')
        ws_dir = sp.ROOT/'results/highell'/('field_coupling' if field else 'coupling_matrices')
        for proj in sp.PROJS:
            for mask in range(1 if field else len(sp.PERC)):
                pol, tt, counts = read_pseudos(pseudo_dir, proj, mask)
                actual = {}
                for spin, raw in [(0, tt), (2, pol)]:
                    matrix, source_edges = read_matrix(ws_dir, proj, mask, spin)
                    assert np.array_equal(edges, source_edges)
                    cl, _ = decouple_many(matrix, sp.rebin_pseudo(raw, counts, edges))
                    if spin == 0:
                        actual['tt'] = cl[:, 0]/correction
                    else:
                        actual.update(ee=cl[:, 0]/correction, bb=cl[:, 3]/correction)
                for key in actual:
                    reference = []
                    for seed in range(sp.NSIMS):
                        root = sp.ROOT/'results/highell'
                        if key == 'tt':
                            directory = root/('scalar_tt_field_bandpowers' if field else 'scalar_tt_bandpowers')
                            with np.load(sp._path(proj, seed, directory)) as z:
                                reference.append(z['cl'][mask])
                        else:
                            directory = sp.FIELD_DIR if field else sp.HIGHELL_DIR
                            with np.load(sp._path(proj, seed, directory)) as z:
                                reference.append(z[key][0, 0, mask, 0])
                    reference = np.asarray(reference)
                    delta = actual[key]-reference
                    scatter = reference.std(axis=0, ddof=1)
                    max_sigma = float(np.max(abs(delta)/scatter))
                    max_frac = float(np.max(abs(delta/reference)))
                    metrics.append(dict(field=field, projection=proj, mask=mask,
                                        spectrum=key, max_residual_in_seed_sigma=max_sigma,
                                        max_fractional_residual=max_frac))
                    print('regression', metrics[-1], flush=True)
                    if max_sigma > 1e-5:
                        raise AssertionError(f'width-50 regression exceeds 1e-5 seed sigma: {metrics[-1]}')
    (out/'regression.json').write_text(json.dumps(metrics, indent=2)+'\n')
    return metrics


def field_scan(out):
    widths = [5, 10, 15, 20, 25, 30, 50, 100]
    results, bandpowers = [], {}
    for proj in sp.PROJS:
        pol, tt, counts = read_pseudos(sp.ROOT/'results/highell/fine_field_pseudo_spectra', proj, 0)
        responses = {spin: read_matrix(sp.ROOT/'results/highell/fine_field_coupling', proj, 0, spin)
                     for spin in (0, 2)}
        for width in widths:
            edges = np.arange(0, sp.LMAX+1, width)
            row = dict(projection=proj, delta_ell=width)
            cls = {}
            for spin, raw in [(0, tt), (2, pol)]:
                matrix, source_edges = responses[spin]
                matrix = sp.rebin_flat_response(matrix, source_edges, edges, counts, 1 if spin == 0 else 4)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', LinAlgWarning)
                        cl, condition = decouple_many(matrix, sp.rebin_pseudo(raw, counts, edges))
                except (np.linalg.LinAlgError, LinAlgWarning):
                    row[f'condition_spin{spin}'] = None
                    continue
                row[f'condition_spin{spin}'] = condition
                ell = (edges[:-1]+edges[1:])/2
                from spt3g.mapspectra.pixel_window import pixwin_flatsky
                cl /= pixwin_flatsky(ell, sp.RES_ARCMIN)[None, None, :]**2
                if spin == 0:
                    cls['tt'] = cl[:, 0]
                else:
                    cls.update(ee=cl[:, 0], bb=cl[:, 3])
                for key, arr in cls.items():
                    select = (ell >= 100) & (ell <= 5000)
                    arr = arr[:, select]
                    centered = arr-arr.mean(axis=0)
                    rho = (centered[:, :-1]*centered[:, 1:]).sum(axis=0)
                    rho /= np.sqrt((centered[:, :-1]**2).sum(axis=0)*(centered[:, 1:]**2).sum(axis=0))
                    row[f'{key}_adjacent_rho_mean'] = float(rho.mean())
                    row[f'{key}_adjacent_rho_quantiles'] = np.quantile(rho, [.16, .5, .84]).tolist()
            if width == 50 and len(cls) == 3:
                for key, arr in cls.items():
                    reference = []
                    for seed in range(sp.NSIMS):
                        directory = (sp.ROOT/'results/highell/scalar_tt_field_bandpowers'
                                     if key == 'tt' else sp.FIELD_DIR)
                        with np.load(sp._path(proj, seed, directory)) as z:
                            reference.append(z['cl'][0] if key == 'tt' else z[key][0, 0, 0, 0])
                    reference = np.asarray(reference)
                    sigma = float(np.max(abs(arr-reference)/reference.std(axis=0, ddof=1)))
                    row[f'{key}_width50_residual_in_seed_sigma'] = sigma
                    if sigma > 1e-5:
                        raise AssertionError(f'width-five response aggregation fails width-50 regression: {row}')
            results.append(row)
            if len(cls) == 3:
                bandpowers[proj, width] = (ell, cls)
            print('conditioning', row, flush=True)
    # Explicit numerical convention, reported with the condition numbers.
    acceptable = [width for width in widths if all(
        row.get(f'condition_spin{spin}') is not None and row[f'condition_spin{spin}'] < 1000
        for row in results if row['delta_ell'] == width for spin in (0, 2))]
    chosen = min(acceptable) if acceptable else None
    payload = dict(condition_limit=1000, finest_tested_delta_ell=chosen,
                   tested_widths=widths, width_scope='multiples of five; intervening integers untested',
                   correlation_ell_range=[100, 5000], seeds=sp.NSIMS, scan=results)
    (out/'field_conditioning.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    if chosen is None:
        raise AssertionError('no tested width has condition number below 1000')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    for proj, color in zip(sp.PROJS, ['#4477AA', '#EE6677', '#228833', '#AA3377']):
        for width in [chosen, 50]:
            ell, cls = bandpowers[proj, width]
            dl = ell*(ell+1)/(2*np.pi)*cls['bb'].mean(axis=0)
            ax.plot(ell, dl, color=color, lw=.7 if width == chosen else 1.8,
                    alpha=.7 if width == chosen else 1,
                    label=f'{sp.PROJ_LABEL[proj]}, Δℓ={width}')
    ax.set(xlim=(100, 6000), yscale='log', xlabel=r'$\ell$', ylabel=r'$D_\ell^{BB}$ [$\mu$K$^2$]',
           title=f'1500d field: mean of {sp.NSIMS} seeds; fine and width-50 decoupling')
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out/'field_fine_vs_50.png', dpi=180)
    plt.close(fig)
    np.savez(out/'field_bandpowers.npz', **{
        f'{proj}_d{width}_{key}': arr for (proj, width), (_, cls) in bandpowers.items()
        for key, arr in cls.items()})
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=sp.ROOT/'results/highell/fine_validation')
    p.add_argument('--regression-only', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    regression(args.out)
    if not args.regression_only:
        field_scan(args.out)


if __name__ == '__main__':
    main()
