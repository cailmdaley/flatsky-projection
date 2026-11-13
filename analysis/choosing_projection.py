"""Choosing a projection: chart geometry and a conditional inverse on the 1500d field.

`geometry`: on the field's mask (rebuilt from its `healpix_masks` index file
at a quadrature Nside), each chart's mask-weighted mean squared shear <g^2>
at the default centre and at the centre that minimizes it, its area
magnification and white-noise plane factor, and ZEA with a best
determinant-one diagonal stretch.

`measurement`: per chart, broad-band input amplitudes fitted through the
coupling matrix to the 20 production field skies (`bandpowers`,
`scalar_bandpowers`, simulation windows divided out), TT through the scalar
block and EE/BB jointly, in bands of 250 and 500 over 500 <= l < 4000.
Amplitudes multiply the known simulation spectrum; outside 500--4000 the true
input is held fixed. This is a conditional test, not a model-independent
inverse or a nonzero-B/noisy survey forecast.

Run:  bash env.sh python3 analysis/choosing_projection.py
          --bandpowers results/fiducial/bandpowers
          --scalar-bandpowers results/fiducial/scalar_bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra data/theory_spectra.npz
          --shear-moments results/fiducial/shear_moments
          --healpix-masks results/fiducial/healpix_masks --out <dir>
Out:  <out>/choosing_projection.json, <out>/amplitudes.npz (per-sky band
      amplitudes) and the diagnostic plot comparison.{png,pdf}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import geometry as G
import spectra as S
import shear_leakage as SL
import theory as T
from section5_geometry import footprint_samples

PROJS = ('ProjZEA', 'ProjSTG', 'ProjSFL')


def fit_amplitudes(matrix, data, reference, select, edges, input_ell, shape,
                   spin=False):
    """Fit broad-band amplitudes with equal fractional-output-bin weighting."""
    basis = np.array([shape*((input_ell >= lo) & (input_ell < hi))
                      for lo, hi in zip(edges[:-1], edges[1:])]).T
    design = matrix @ basis if not spin else np.block([
        [matrix['EE<-EE'] @ basis, matrix['EE<-BB'] @ basis],
        [matrix['BB<-EE'] @ basis, matrix['BB<-BB'] @ basis]])
    keep = np.tile(select, 2) if spin else select
    scale = np.tile(reference, 2) if spin else reference
    a = design[keep]/scale[keep, None]
    inverse = np.linalg.pinv(a, rcond=1e-10)
    # Fit deviations from the exact known input, so the fixed exterior is explicit.
    expected = (matrix['EE<-EE'] @ shape if spin else matrix @ shape)
    if spin:
        expected = np.r_[expected, matrix['BB<-EE'] @ shape]
    estimates = ((data-expected)[:, keep]/scale[keep]) @ inverse.T
    return estimates, np.linalg.cond(a), inverse @ a


def measurement(run, out, widths=(250, 500)):
    th = T.load(run.theory_spectra)
    lp = th['ell'].astype(float)
    mom = run.field_moments()
    ee, bb = run.load_field('EE'), run.load_field('BB')
    tt = run.load_scalar('TT', field=True)
    select = (run.ell >= 500) & (run.ell < 4000)
    rows, arrays = [], {}
    for proj in PROJS:
        j = S.PROJS.index(proj)
        r = SL.coupling_matrix(mom, 0, j, run.ell, lp, band_limit=run.lmax)
        for width in widths:
            edges = np.arange(500, 4001, width)
            for spec in ('TT', 'POL'):
                spin = spec == 'POL'
                shape = th['unlensed_EE' if spin else 'unlensed_TT']
                reference = run.bin_theory(shape)
                data = np.c_[ee[j], bb[j]] if spin else tt[j]
                delta, condition, resolution = fit_amplitudes(
                    r if spin else r['TT<-TT'], data, reference, select,
                    edges, lp, shape, spin)
                np.testing.assert_allclose(resolution, np.eye(len(resolution)), atol=1e-7)
                n = len(edges)-1
                for label, value in ([('EE', delta[:, :n]), ('BB', delta[:, n:])]
                                     if spin else [('TT', delta)]):
                    key = f'{proj}_{width}_{label}'
                    arrays[key] = value
                    per_sky = value.mean(1)
                    rows.append(dict(projection=proj, width=width, spectrum=label,
                                     mean_bias=float(per_sky.mean()),
                                     mean_bias_se=float(per_sky.std(ddof=1)/np.sqrt(len(value))),
                                     rms_bin_bias=float(np.sqrt(np.mean(value.mean(0)**2))),
                                     mean_bin_variance=float(value.var(0, ddof=1).mean()),
                                     condition=float(condition),
                                     bin_bias=value.mean(0).tolist(),
                                     bin_se=(value.std(0, ddof=1)/np.sqrt(len(value))).tolist()))
        print(proj, 'done', flush=True)
    for row in rows:
        ref = next(r for r in rows if r['projection']=='ProjZEA'
                   and r['width']==row['width'] and r['spectrum']==row['spectrum'])
        row['variance_over_zea'] = row['mean_bin_variance']/ref['mean_bin_variance']
        x = arrays[f"{row['projection']}_{row['width']}_{row['spectrum']}"]
        y = arrays[f"ProjZEA_{row['width']}_{row['spectrum']}"]
        deleted = np.array([np.delete(x,i,axis=0).var(0,ddof=1).mean()
                            /np.delete(y,i,axis=0).var(0,ddof=1).mean()
                            for i in range(len(x))])
        row['variance_ratio_jackknife_se'] = float(np.sqrt((len(x)-1)/len(x)
                                                  *np.sum((deleted-deleted.mean())**2)))
    np.savez(out/'amplitudes.npz', **arrays)
    return rows


def geometry(healpix_masks, nside=128):
    ra, dec, w = footprint_samples(healpix_masks, nside)['field1500d']
    default = [G.RA_DEG, G.DEC_DEG]

    def metric(params, proj, affine=False):
        kw = dict(alpha_center_deg=params[0], delta_center_deg=params[1])
        jac = G.jacobian(proj, ra, dec, **kw)
        if affine:
            # General positive diagonal stretch, determinant one, on the ZEA chart.
            jac = np.einsum('ij,njk->nik', np.diag(np.exp([params[2], -params[2]])), jac)
        s1, s2 = G.singular_values(jac)
        return np.average(((s1-s2)/(s1+s2))**2, weights=w)

    rows = []
    for proj in PROJS:
        j = G.jacobian(proj, ra, dec)
        d = np.linalg.det(j)
        if proj == 'ProjSTG':
            # Every STG centre has identically zero shear; do not optimize
            # finite-difference roundoff and pretend it selects a centre.
            best, centre = 0., None
        elif proj == 'ProjSFL':
            optimum = minimize(lambda a: metric([a[0], G.DEC_DEG], proj), [G.RA_DEG],
                               method='Nelder-Mead')
            best, centre = float(optimum.fun), [float(optimum.x[0]), G.DEC_DEG]
        else:
            optimum = minimize(metric, default, args=(proj,), method='Nelder-Mead',
                               options=dict(xatol=1e-6, fatol=1e-13))
            best, centre = float(optimum.fun), optimum.x.tolist()
        rows.append(dict(projection=proj, default_g2=float(metric(default, proj)),
                         optimum_g2=best, centre=centre,
                         area_magnification=float(np.average(d, weights=w)),
                         sky_white_noise_plane_factor=float(np.average(d*d,weights=w)/np.average(d,weights=w)),
                         fixed_plane_pixel_sky_area_range=[float(1/d.max()),float(1/d.min())]))
    affine = minimize(metric, default+[0.], args=('ProjZEA',True), method='Nelder-Mead',
                      options=dict(xatol=1e-7, fatol=1e-14))
    return dict(charts=rows, affine_zea=dict(g2=float(affine.fun), parameters=affine.x.tolist()),
                nside=nside)


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.rc()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, layout='constrained')
    for col, spec in enumerate(('TT','EE','BB')):
        for proj in PROJS:
            row = next(r for r in result['measurements'] if r['projection']==proj
                       and r['spectrum']==spec and r['width']==500)
            x = np.arange(750,4000,500)
            axes[0,col].errorbar(x,row['bin_bias'],yerr=row['bin_se'],
                                color=ps.PROJ_COLORS[proj],label=proj[4:],marker='o',ms=3)
            with np.load(out/'amplitudes.npz') as z:
                sd = z[f'{proj}_500_{spec}'].std(0,ddof=1)
            axes[1,col].plot(x,sd,color=ps.PROJ_COLORS[proj],label=proj[4:])
        axes[0,col].axhline(0,color='.6',lw=.6)
        axes[0,col].set_ylabel(spec+' bias'+(' / input EE' if spec=='BB' else ' / input'))
        axes[1,col].set_ylabel(spec+' scatter'+(' / input EE' if spec=='BB' else ' / input'))
        axes[1,col].set_yscale('log')
        axes[1,col].set_xlabel(r'$\ell$')
    axes[0,0].legend(frameon=False)
    fig.savefig(out/'comparison.png',dpi=140)
    fig.savefig(out/'comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(parser, 'bandpowers', 'scalar_bandpowers', 'simulation_windows',
                 'theory_spectra', 'shear_moments', 'healpix_masks')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = S.Run.from_args(args)
    result = dict(geometry=geometry(args.healpix_masks), measurements=measurement(run, args.out),
                  caveat='Known-shape amplitudes, fixed true exterior, zero input B, no instrumental noise; BB units are input EE.')
    (args.out/'choosing_projection.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(result, args.out)


if __name__ == '__main__':
    main()
