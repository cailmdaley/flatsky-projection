"""Controls on the 1500d field's residuals against the forward model.

Five independent checks of what remains after the windowed coupling-matrix
model (`pipeline_windows`) and NaMaster's radial response
(`residual_estimator_response`) are divided out of the production field
spectra:

  independent  The 260 skies of `bin_covariance_sims`, decoupled with the
               production field workspaces, never pooled with the 20
               production skies: their band residuals against the integer-bin,
               radial-response and spherical (k = l + 1/2) references, Hotelling
               T^2 tests of production against independent and of the
               independent mean against each reference, with the
               input-conditioned variant for TT and EE.
  map_upsample The production maps refined twice as finely before the
               estimator (`residual_map_upsample`), with zero, 64 and 128
               pixels of padding: native and refined residuals and the B the
               refinement removes. The native spectra must reproduce
               production, and the 64- and 128-pixel paired changes agree.
  pure_e       Periodic pure-E fields through the field mask and pure-B
               estimator, native and refined (`residual_pure_e_test`): the
               estimator's own B, normalized to the periodic E power.
  rings        The production E alms resynthesized at Nside 4096 with 8x or
               16x subsampling (`residual_ring_test`): BB over input and own
               EE against production, which the native spectra reproduce;
               and the input B/E of the alms those controls zero.
  fft_gate     An independent padded 2D-FFT pseudo-spectrum of the native
               model on the real field mask, against the workspace's
               coupled spectrum, within 0.5% per bin.

Each gate is an assertion: the script fails rather than write a control that
did not pass.

Run:  bash env.sh python3 analysis/residual_controls.py
          --pipeline-windows <dir> --residual-estimator-response <dir>
          --bin-covariance-sims results/fiducial/bin_covariance_sims
          --coupling-matrices results/fiducial/coupling_matrices
          --residual-map-upsample results/fiducial/residual_map_upsample
          --residual-pure-e-test results/fiducial/residual_pure_e_test
          --residual-ring-test results/fiducial/residual_ring_test
          --preprocessed-alms results/fiducial/preprocessed_alms
          --flatsky-masks results/fiducial/flatsky_masks
          --theory-spectra data/theory_spectra.npz
          --lmax l6000 --delta-ell d50 --out <dir>
Out:  <out>/residual_controls.json, {independent_summary, independent_tests,
      map_upsample, pure_e, rings, ring_input_b, fft_gate}, lists of records;
      <out>/independent_means.npz (production and independent field spectra)
      and <out>/map_upsample.npz (native and refined spectra and the model).
"""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from scipy.fft import fft2, fftfreq, ifft2
from scipy.stats import f, t

import spectra as S
import theory as T
from _decisions import parse_delta_ell, parse_lmax
from validate_fine_spectra import decouple_many, read_matrix

BANDS = ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800))


def binned_theory(path, spec, lmax, delta_l):
    th = T.load(path)
    full = np.zeros(int(th['ell'].max())+1)
    full[th['ell']] = th[spec]
    full[0] = 0.
    return np.array([np.nanmean(full[lo:lo+delta_l]) for lo in range(0, lmax, delta_l)])


def load_native(directory, coupling_matrices, edges):
    """Field TT, EE and BB of every sky in `directory`, decoupled per projection."""
    index = json.loads((directory/'index_seed0000.json').read_text())
    projections = [name.split('_')[1] for name in index['pseudo_spectra']]
    result = {}
    for proj in projections:
        raw = [np.load(path) for path in sorted(directory.glob(f'pseudo_{proj}_seed*_mask0.npz'))]
        counts = raw[0]['counts']
        for spin, key in ((0, 'tt'), (2, 'pol')):
            matrix, stored = read_matrix(coupling_matrices, proj, S.FIELD, spin)
            np.testing.assert_array_equal(stored, edges)
            cl, _ = decouple_many(matrix, S.rebin_pseudo(np.array([r[key] for r in raw]), counts, edges))
            for spec, i in ([('tt', 0)] if spin == 0 else [('ee', 0), ('bb', 3)]):
                result[proj, spec] = cl[:, i]
        for r in raw:
            r.close()
    return result


def hotelling_p(t2, dim, n):
    return f.sf((n-dim)*t2/((n-1)*dim), dim, n-dim)


def independent(report, estimator, sims, coupling_matrices, edges, pix):
    """Independent 260-sky check of the production residuals, never pooling skies."""
    ELL = report['ell']
    new = load_native(sims, coupling_matrices, edges)
    result = {'ell': ELL}
    rows, tests = [], []
    for (proj, spec), native in new.items():
        spec = spec.upper()
        prefix = f'field_{spec}_{proj}'
        old = report[prefix+'_raw']
        arr = native/pix
        result[f'{proj}_{spec}_production'] = old
        result[f'{proj}_{spec}_independent'] = arr
        # BB STG has no geometric signal, normalize to EE for floor displays.
        floor = spec == 'BB' and proj == 'ProjSTG'
        model = report[f'field_EE_{proj}_windowed'] if floor else report[prefix+'_windowed']
        radial_model = model if floor else estimator[f'{proj}_{spec}_estimator']
        spherical = model if floor else estimator[f'{proj}_{spec}_spherical']
        print(proj, spec, 'floor/EE' if floor else 'fractional residual', flush=True)
        for lo, hi in BANDS:
            sel = (ELL >= lo) & (ELL < hi)
            for name, value in [('production20', old), ('independent260', arr)]:
                ratio = (value[:, sel]/model[sel]-(0 if floor else 1)).mean(-1)
                radial = (value[:, sel]/radial_model[sel]-(0 if floor else 1)).mean(-1)
                curved = (value[:, sel]/spherical[sel]-(0 if floor else 1)).mean(-1)
                rows.append(dict(projection=proj, spectrum=spec, ensemble=name, lo=lo, hi=hi,
                                 bin_residual=float(ratio.mean()),
                                 bin_se=float(ratio.std(ddof=1)/np.sqrt(len(ratio))),
                                 spherical_residual=float(curved.mean()),
                                 spherical_se=float(curved.std(ddof=1)/np.sqrt(len(curved)))))
                print(f'{lo}-{hi} {name}: bin {100*ratio.mean():+.5f} +/- {100*ratio.std(ddof=1)/np.sqrt(len(ratio)):.5f} %, '
                      f'radial-response {100*radial.mean():+.5f} +/- {100*radial.std(ddof=1)/np.sqrt(len(radial)):.5f} %, '
                      f'spherical {100*curved.mean():+.5f} +/- {100*curved.std(ddof=1)/np.sqrt(len(curved)):.5f} %', flush=True)
        if floor:
            continue
        for lo, hi in ((100, 1000), (500, 1000), (1000, 4000), (4000, 4800), (4500, 4800)):
            sel = (ELL >= lo) & (ELL < hi)
            x, y = old[:, sel]/model[sel], arr[:, sel]/model[sel]
            cov = np.cov(y, rowvar=False)
            d = x.mean(0)-y.mean(0)
            # Test against the independently estimated covariance, not a
            # diagonal chi-square. The difference includes old-alm vs CAMB
            # mismatch, so this is not a pure sample-variance p-value.
            stat = float(d@np.linalg.solve(cov, d)/(1/len(x)+1/len(y)))
            dim, n = len(d), len(y)
            p = hotelling_p(stat, dim, n)
            band_se = float(np.sqrt(cov.mean()*(1/len(x)+1/len(y))))
            tests.append(dict(projection=proj, spectrum=spec, null='old-new', lo=lo, hi=hi,
                              t2=stat, dimension=dim, p=float(p), mean=float(d.mean()), se=band_se,
                              band_mean_p=float(2*t.sf(abs(d.mean()/band_se), n-1))))
            print(f'old-new {lo}-{hi}: T2 {stat:.3f}, dim {dim}, covariance-corrected F p {p:.5g}', flush=True)
            # Remove only the predicted all-sky input-mean difference, not
            # the finite-footprint realization scatter.
            if spec in ('TT', 'EE'):
                offset = (estimator[f'{proj}_{spec}_realized']-spherical)/model
                dc = d-offset[sel]
                tc = float(dc@np.linalg.solve(cov, dc)/(1/len(x)+1/len(y)))
                pc = float(hotelling_p(tc, dim, n))
                # Nominal diagnostic: subtracting a realized all-sky mean also
                # slightly reduces covariance. The unconditioned covariance is
                # kept, with no claim of exact coverage.
                tests.append(dict(projection=proj, spectrum=spec, null='old-new-input-conditioned',
                                  lo=lo, hi=hi, t2=tc, dimension=dim, p=pc,
                                  mean=float(dc.mean()), se=band_se,
                                  band_mean_p=float(2*t.sf(abs(dc.mean()/band_se), n-1)),
                                  covariance='unconditioned-diagnostic'))
                print(f'old-new input-conditioned {lo}-{hi}: T2 {tc:.3f}, F p {pc:.5g}', flush=True)
            for label, theory in [('bin', model), ('radial', radial_model), ('spherical', spherical)]:
                residual = arr[:, sel]/theory[sel]-1
                mean = residual.mean(0)
                covr = np.cov(residual, rowvar=False)
                t2 = float(n*mean@np.linalg.solve(covr, mean))
                prob = hotelling_p(t2, dim, n)
                # A 1-D test of the per-sky band mean complements the
                # full-covariance test of structured bin residuals.
                band = residual.mean(-1)
                se = band.std(ddof=1)/np.sqrt(n)
                p1 = float(2*t.sf(abs(band.mean()/se), n-1))
                tests.append(dict(projection=proj, spectrum=spec, null='new-model-'+label,
                                  lo=lo, hi=hi, t2=t2, dimension=dim, p=float(prob),
                                  mean=float(band.mean()), se=float(se), band_mean_p=p1,
                                  max_bin_residual=float(max(abs(mean)))))
                print(f'new-model {label} {lo}-{hi}: T2 {t2:.3f}, dim {dim}, F p {prob:.5g}, '
                      f'band-mean p {p1:.5g}, max residual {100*max(abs(mean)):.4f} %', flush=True)
    return result, rows, tests


def map_upsample(report, directory, pix):
    """Production-pair estimator refinement, with a mandatory native roundtrip."""
    ELL = report['ell']
    rows, saved = [], {'ell': ELL}
    for proj in S.PROJS:
        data = [np.load(path) for path in sorted(directory.glob(f'upsample_{proj}_seed*.npz'))]
        seeds = [int(d['seed']) for d in data]
        for spec, index in [('EE', 0), ('BB', 3)]:
            floor = spec == 'BB' and proj == 'ProjSTG'
            prefix = f'field_{spec}_{proj}'
            raw = report[prefix+'_raw'][seeds]*pix
            native = np.array([d['pad0_x1_decoupled'][index] for d in data])
            np.testing.assert_allclose(native, raw, rtol=2e-9, atol=1e-25)
            print(proj, spec, 'native production roundtrip passed', flush=True)
            model = (report[f'field_EE_{proj}_windowed'] if floor else
                     report[prefix+'_windowed']+report[prefix+'_alias'])*pix
            saved[f'{proj}_{spec}_model'] = model
            for pad in (0, 64, 128):
                a, b = [np.array([d[f'pad{pad}_x{factor}_decoupled'][index] for d in data])
                        for factor in (1, 2)]
                saved[f'{proj}_{spec}_pad{pad}_native'] = a
                saved[f'{proj}_{spec}_pad{pad}_fine'] = b
                for lo, hi in BANDS:
                    sel = (ELL >= lo) & (ELL < hi)
                    x, y = [(v[:, sel]/model[sel]-(0 if floor else 1)).mean(-1) for v in (a, b)]
                    change = y-x
                    row = dict(projection=proj, spectrum=spec, nseed=len(data), pad=pad,
                               lo=lo, hi=hi, native=float(x.mean()), fine=float(y.mean()),
                               delta=float(change.mean()),
                               native_se=float(x.std(ddof=1)/np.sqrt(len(x))),
                               fine_se=float(y.std(ddof=1)/np.sqrt(len(y))),
                               delta_se=float(change.std(ddof=1)/np.sqrt(len(change))),
                               native_power=float(a[:, sel].mean()),
                               fine_power=float(b[:, sel].mean()),
                               removed_native=float((a-b)[:, sel].mean()),
                               excess_native=float((a[:, sel]-(0 if floor else model[sel])).mean()),
                               removed_fraction=float((a-b)[:, sel].mean()
                                                      / (a[:, sel]-(0 if floor else model[sel])).mean()))
                    rows.append(row)
                    print(f'{proj} {spec} pad{pad} {lo}-{hi}: native {100*x.mean():+.5f}, '
                          f'fine {100*y.mean():+.5f} +/- {100*row["fine_se"]:.5f}, '
                          f'delta {100*change.mean():+.5f} +/- {100*row["delta_se"]:.5f} %', flush=True)
        # Padding changes the Fourier lattice, so compare the *paired change*
        # within each padding, not the absolute spectra between paddings.
        sel = (ELL >= 4000) & (ELL < 4800)
        norm = report[f'field_EE_{proj}_raw'][seeds, :][:, sel]*pix[sel]
        changes = [np.array([(d[f'pad{pad}_x1_decoupled'][3, sel]-d[f'pad{pad}_x2_decoupled'][3, sel])/norm[i]
                             for i, d in enumerate(data)]).mean(-1)
                   for pad in (0, 64, 128)]
        # Absolute tolerance is 1e-7 of E, much smaller than the ~1e-5
        # upper-band STG/ZEA intervention; no relative test for null SFL.
        np.testing.assert_allclose(changes[1], changes[2], rtol=0.02, atol=1e-7)
        print(proj, '64/128 padding paired-change gate passed', flush=True)
        for d in data:
            d.close()
    return rows, saved


def pure_e(report, directory, pix):
    """Paired native/fine pure-E mask-product interventions."""
    rows = []
    for proj in S.PROJS:
        data = [np.load(path) for path in sorted(directory.glob(f'pure_e_{proj}_seed*.npz'))]
        ell = data[0]['ell']
        print(proj, 'seeds', len(data), flush=True)
        for cutoff in (4000, 6000):
            coarse = np.array([z[f'cut{cutoff}_x1_decoupled'][3] for z in data])
            fine = np.array([z[f'cut{cutoff}_x2_decoupled'][3] for z in data])
            # Dimensionless leakage normalized to the same periodic native E
            # power, keeping coarse/fine cancellation exact in numerator.
            ee = np.array([z['cut6000_x1_periodic'][0] for z in data])
            # Compare dimensionful native spectra before any window division.
            prod_bb = report[f'field_BB_{proj}_raw'].mean(0)*pix
            prod_model = (np.zeros_like(ell) if proj == 'ProjSTG' else
                          report[f'field_BB_{proj}_windowed']+report[f'field_BB_{proj}_alias'])*pix
            clean = (ell >= 1000) & (ell < 4800)
            for d in data:
                for factor in (1, 2):
                    periodic = d[f'cut{cutoff}_x{factor}_periodic']
                    assert abs(periodic[3, clean]).max()/periodic[0, clean].mean() < 1e-12
            assert abs(fine[:, clean]).mean() < 1e-4*abs(coarse[:, clean]).mean()
            for lo, hi in BANDS:
                sel = (ell >= lo) & (ell < hi)
                c = (coarse[:, sel]/ee[:, sel]).mean(-1)
                fi = (fine[:, sel]/ee[:, sel]).mean(-1)
                removed = ((coarse-fine)[:, sel]/ee[:, sel]).mean(-1)
                row = dict(projection=proj, nseed=len(data), cutoff=cutoff, lo=lo, hi=hi,
                           native=float(c.mean()), fine=float(fi.mean()), removed=float(removed.mean()),
                           native_se=float(c.std(ddof=1)/np.sqrt(len(c))),
                           fine_se=float(fi.std(ddof=1)/np.sqrt(len(fi))),
                           planar_native=float(coarse[:, sel].mean()),
                           planar_removed_native=float((coarse-fine)[:, sel].mean()),
                           production_bb_native=float(prod_bb[sel].mean()),
                           production_excess_native=float((prod_bb-prod_model)[sel].mean()),
                           planar_over_production_excess=float(coarse[:, sel].mean()/(prod_bb-prod_model)[sel].mean()))
                rows.append(row)
                print(f' cut{cutoff} {lo}-{hi}: native {c.mean():.6g} +/- {row["native_se"]:.2g}, '
                      f'fine {fi.mean():.6g} +/- {row["fine_se"]:.2g}, removed/native {removed.mean()/c.mean():.6g}; '
                      f'planar/native-production-excess {row["planar_over_production_excess"]:.6g}', flush=True)
        for d in data:
            d.close()
    return rows


def rings(report, directory, alms, cl_ee, pix):
    """Paired synthesis-ring and finite-subsample interventions."""
    ELL = report['ell']
    projections = sorted({path.name.split('_')[1] for path in directory.glob('rings_*.npz')})
    seeds = sorted({int(path.stem.split('seed')[1]) for path in directory.glob('rings_*.npz')})
    # Production synthesis retains the tiny map2alm B coefficient, whereas
    # this intervention sets it to zero. Verify that this is irrelevant.
    input_b = []
    for seed in seeds:
        ae, ab = hp.read_alm(str(alms/f'alms_seed{seed:04d}.fits'), hdu=(2, 3))
        e, b = hp.alm2cl(ae), hp.alm2cl(ab)
        for lo, hi in ((100, 1000), (1000, 4000), (4500, 4800)):
            ratio = float(np.mean(b[lo:hi]/e[lo:hi]))
            assert ratio < 1e-10, (seed, lo, hi, ratio)
            input_b.append(dict(seed=seed, lo=lo, hi=hi, bb_ee=ratio))
        del ae, ab
    rows = []
    for proj in projections:
        ds = [np.load(directory/f'rings_{proj}_seed{s:04d}.npz') for s in seeds]
        native = np.array([d['n8192_r8'] for d in ds])
        for spec, index in [('EE', 0), ('BB', 3)]:
            np.testing.assert_allclose(native[:, index], report[f'field_{spec}_{proj}_raw'][seeds]*pix,
                                       rtol=2e-9, atol=1e-25)
        print(proj, 'native roundtrips passed', flush=True)
        normalization = cl_ee*pix
        for lo, hi in BANDS:
            sel = (ELL >= lo) & (ELL < hi)
            for name in ('n8192_r8', 'n4096_r8', 'n4096_r16'):
                value = np.array([d[name] for d in ds])
                bb = (value[:, 3, sel]/normalization[sel]).mean(-1)
                ee = (value[:, 0, sel]/native[:, 0, sel]).mean(-1)
                leakage = (value[:, 3, sel]/value[:, 0, sel]).mean(-1)
                baseline = (native[:, 3, sel]/native[:, 0, sel]).mean(-1)
                coarse = np.array([(d['n4096_r8'][3, sel]/d['n4096_r8'][0, sel]).mean() for d in ds])
                relative = leakage/coarse
                row = dict(projection=proj, intervention=name, lo=lo, hi=hi, nseed=len(ds),
                           bb_input_ee=float(bb.mean()), bb_input_ee_se=float(bb.std(ddof=1)/np.sqrt(len(bb))),
                           ee_native_ratio=float(ee.mean()), ee_native_ratio_se=float(ee.std(ddof=1)/np.sqrt(len(ee))),
                           bb_own_ee=float(leakage.mean()), bb_own_ee_se=float(leakage.std(ddof=1)/np.sqrt(len(leakage))),
                           paired_bb_ee_change=float((leakage-baseline).mean()),
                           paired_bb_ee_change_se=float((leakage-baseline).std(ddof=1)/np.sqrt(len(leakage))),
                           bb_ee_vs_4096_r8=float(relative.mean()),
                           bb_ee_vs_4096_r8_se=float(relative.std(ddof=1)/np.sqrt(len(relative))))
                rows.append(row)
                print(f'{lo}-{hi} {name}: BB/inputEE {bb.mean():.7g} +/- {row["bb_input_ee_se"]:.2g}; '
                      f'EE/native {ee.mean():.7g}; BB/ownEE {leakage.mean():.7g}; paired BB/EE change '
                      f'{row["paired_bb_ee_change"]:+.7g} +/- {row["paired_bb_ee_change_se"]:.2g}', flush=True)
        for d in ds:
            d.close()
    return rows, input_b


def fft_gate(estimator, masks, coupling_matrices, ELL, lmax, delta_l):
    """Independent 2D FFT gate for the radial NaMaster response.

    Checks Fourier-cell weighting, normalization and scalar/EE mask coupling
    on the real production grid. It does not assert stationarity of projected
    skies. The padded FFT and NaMaster radial approximations need not agree
    exactly.
    """
    rows = []
    nbins = lmax//delta_l
    bins = nmt.NmtBinFlat(np.arange(0., lmax, delta_l), np.arange(delta_l, lmax+delta_l, delta_l, dtype=float))
    ell = estimator['input_ell']
    clean = (ELL >= 100) & (ELL < 4800)
    for proj in S.PROJS:
        with np.load(masks/f'mask_{proj}_{S.FIELD}.npz') as d:
            mask = d['mask'].astype(float)
            lx, ly = float(d['lx_rad']), float(d['ly_rad'])
        ny, nx = mask.shape
        dx, dy = lx/nx, ly/ny
        kx, ky = np.meshgrid(fftfreq(nx, 1/nx)*2*np.pi/lx, fftfreq(ny, 1/ny)*2*np.pi/ly)
        k, psi = np.hypot(kx, ky), np.arctan2(ky, kx)
        index = (k/delta_l).astype(int).ravel()
        count = np.bincount(index, minlength=nbins)[:nbins]

        def binmean(power):
            return np.bincount(index, power.ravel(), minlength=nbins)[:nbins]/count

        # No fitted normalization: compare one identical masked white map.
        sky = np.random.default_rng(0).normal(size=mask.shape)
        field = nmt.NmtFieldFlat(lx, ly, mask, [sky])
        nmt_power = nmt.compute_coupled_cell_flat(field, field, bins)[0]
        own_power = binmean(abs(fft2(mask*sky)*dx*dy)**2/(lx*ly))
        np.testing.assert_allclose(nmt_power[clean], own_power[clean], rtol=1e-10, atol=0.)
        del field, sky
        # The mask autocorrelation and covariance use a twice-padded grid.
        # Evaluate the result at the original Fourier cells (even indices).
        NY, NX = 2*ny, 2*nx
        correlation = ifft2(abs(fft2(mask, s=(NY, NX)))**2).real*dx*dy
        qx, qy = np.meshgrid(fftfreq(NX, 1/NX)*np.pi/lx, fftfreq(NY, 1/NY)*np.pi/ly)
        q, phi = np.hypot(qx, qy), np.arctan2(qy, qx)

        def pseudo(cq):
            covariance = ifft2(cq).real*NX*NY/(4*lx*ly)
            return (fft2(correlation*covariance).real*dx*dy/(lx*ly))[::2, ::2]

        for spec, spin in [('TT', 0), ('EE', 2)]:
            native = estimator[f'{proj}_{spec}_native']
            cq = np.interp(q, ell, native)
            if spin == 0:
                power = pseudo(cq)
            else:
                power = .5*(pseudo(cq)+np.cos(4*psi)*pseudo(cq*np.cos(4*phi))
                            + np.sin(4*psi)*pseudo(cq*np.sin(4*phi)))
            ws = nmt.NmtWorkspaceFlat()
            ws.read_from(str(coupling_matrices/f'w{spin}{spin}_{proj}_{S.FIELD}.fits'))
            cl = np.zeros((1 if spin == 0 else 4, len(ell)))
            cl[0] = native
            coupled = ws.couple_cell(ell, cl)[0]
            ratio = binmean(power)/coupled-1
            # Independent quadrature, not an algebraic couple/decouple
            # identity. Review measured <0.3% per bin for ZEA/STG; a
            # conservative 0.5% envelope holds for all four production grids.
            np.testing.assert_allclose(ratio[clean], 0., rtol=0., atol=.005)
            rows.append(dict(projection=proj, spectrum=spec,
                             max_coupled_fraction=float(abs(ratio[clean]).max()),
                             rms_coupled_fraction=float(np.sqrt(np.mean(ratio[clean]**2)))))
            print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'pipeline_windows', 'residual_estimator_response', 'bin_covariance_sims',
                 'coupling_matrices', 'residual_map_upsample', 'residual_pure_e_test',
                 'residual_ring_test', 'preprocessed_alms', 'flatsky_masks', 'theory_spectra')
    ap.add_argument('--lmax', type=parse_lmax, required=True)
    ap.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = np.load(args.pipeline_windows/'pipeline_windows.npz')
    estimator = np.load(args.residual_estimator_response/'residual_estimator_response.npz')
    (res_arcmin,) = {json.loads(q.read_text())['res_arcmin'] for q in args.flatsky_masks.glob('index_*.json')}
    ELL = report['ell']
    pix = S.flat_pixwin(ELL, res_arcmin)
    edges = np.arange(0, args.lmax+1, args.delta_ell)
    cl_ee = binned_theory(args.theory_spectra, 'unlensed_EE', args.lmax, args.delta_ell)

    means, summary, tests = independent(report, estimator, args.bin_covariance_sims,
                                        args.coupling_matrices, edges, pix)
    upsample, upsample_arrays = map_upsample(report, args.residual_map_upsample, pix)
    pure = pure_e(report, args.residual_pure_e_test, pix)
    ring, input_b = rings(report, args.residual_ring_test, args.preprocessed_alms, cl_ee, pix)
    gate = fft_gate(estimator, args.flatsky_masks, args.coupling_matrices, ELL, args.lmax, args.delta_ell)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out/'independent_means.npz', **means)
    np.savez(args.out/'map_upsample.npz', **upsample_arrays)
    (args.out/'residual_controls.json').write_text(json.dumps(dict(
        independent_summary=summary, independent_tests=tests, map_upsample=upsample,
        pure_e=pure, rings=ring, ring_input_b=input_b, fft_gate=gate), indent=2)+'\n')


if __name__ == '__main__':
    main()
