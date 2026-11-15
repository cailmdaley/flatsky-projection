"""Sky-frame window corrections and paired-seed residuals of the production bandpowers.

For the 1500d field, read alone, and the sweep's reference footprint, read
from the sweep (it is the 1500d field too), each projection and each of TT, the
white scalar sky, EE and BB, this evaluates `pipeline_windows.response` (the
local-Jacobian forward integral with the ring-interpolation and rebin windows,
and optionally the nearest reciprocal-grid images) at five multipoles per
bandpower, and normalizes it to the coupling-matrix forward model of the
shear moments (`shear_leakage.coupling_matrix`, band-limited at the run's
lmax). It then divides the production bandpowers, with no simulation window
removed, by that model. Only a ratio of smooth window integrals is used as a
correction; no acoustic structure is approximated away.

STG has no geometric B. Its BB alias uses the EE normalization, and EE's
window only to display the floor. The floor-subtracted BB ratio removes STG's
measured BB from each sheared projection below L_WALL, a partial diagnostic
rather than a calibrated floor model: ZEA's additive floor is larger than
STG's, and above L_WALL STG itself contains aliases.

The paper's production file is made with `--nside 64 --aliases`.

Run:  bash env.sh python3 analysis/report_pipeline_windows.py
          --theory-spectra results/fiducial/theory_spectra
          --healpix-masks results/fiducial/healpix_masks
          --flatsky-masks results/fiducial/flatsky_masks
          --shear-moments results/fiducial/shear_moments
          --bandpowers results/fiducial/bandpowers
          --scalar-bandpowers results/fiducial/scalar_bandpowers
          --nside 64 --aliases --out <dir>
Out:  <out>/pipeline_windows.npz: `ell`, `projs`, `nside`, `nchi`,
      `aliases_enabled`, and per footprint (`field`, `reference`), spectrum and
      projection `_base`, `_windowed`, `_alias` (model, per band), `_raw`
      (production, seed x band), `_ratio` (raw over windowed), `_ratio_total`
      (raw over windowed plus alias), for sheared BB `_ratio_floor_subtracted`,
      and per footprint and spectrum `_correction` (projection x band).
"""
import argparse
from pathlib import Path

import numpy as np

import shear_leakage as SL
import spectra as S
import theory as T
from pipeline_windows import Sampling, response

REFERENCE = S.FIELD
L_WALL = 4800


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(p, 'theory_spectra', 'healpix_masks', 'flatsky_masks', 'shear_moments',
                 'bandpowers', 'scalar_bandpowers')
    p.add_argument('--nside', type=int, default=32)
    p.add_argument('--nchi', type=int, default=64)
    p.add_argument('--aliases', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run = S.Run.from_args(args)
    sampling = Sampling.read(args.healpix_masks, args.flatsky_masks)
    th = T.load(args.theory_spectra)
    ELL, nbins = run.ell, run.nbins
    reference_index = S.SWEEP.index(REFERENCE)
    outputs = {'ell': ELL, 'projs': S.PROJS, 'nside': args.nside, 'nchi': args.nchi,
               'aliases_enabled': args.aliases}
    # Five samples per bandpower.
    step = run.delta_l/5
    ell = np.arange(step/2, run.lmax, step)
    old_fine, old_bin = S.flat_pixwin(ell, sampling.res_arcmin), S.flat_pixwin(ELL, sampling.res_arcmin)
    cl_ee = run.theory('unlensed_EE')

    def sampled(x):
        return x.reshape(nbins, 5).mean(-1)

    def windowed(x):
        return (x*old_fine).reshape(nbins, 5).mean(-1)/old_bin

    for field in (True, False):
        name = 'field' if field else 'reference'
        footprint = S.FIELD if field else REFERENCE
        mom = run.moments([footprint])
        reference = {}
        for j, proj in enumerate(S.PROJS):
            R = SL.coupling_matrix(mom, 0, j, np.arange(run.lmax, dtype=float),
                                   th['ell'], band_limit=run.lmax)
            for spec, block in (('TT', 'TT<-TT'), ('white', 'TT<-TT'), ('EE', 'EE<-EE'), ('BB', 'BB<-EE')):
                cl = (np.where(th['ell'] >= 2, 1., 0.) if spec == 'white'
                      else th['unlensed_TT' if spec == 'TT' else 'unlensed_EE'])
                reference[(proj, spec)] = (R[block]@cl).reshape(nbins, run.delta_l).mean(-1)
            del R
        for spec in ('TT', 'white', 'EE', 'BB'):
            cl = (np.where(th['ell'] >= 2, 1., 0.) if spec == 'white'
                  else th['unlensed_TT' if spec == 'TT' else 'unlensed_EE'])
            if spec in ('TT', 'white'):
                measured = run.load_scalar(spec, field=field, correct_windows=False)
            else:
                measured = (run.load_field(spec, correct_windows=False) if field
                            else run.load(spec, correct_windows=False))
            if not field:
                measured = measured[reference_index]
            correction = []
            for j, proj in enumerate(S.PROJS):
                spin = 'TT' if spec == 'white' else spec
                b, w, a = response(proj, ell, th['ell'], cl, args.healpix_masks, footprint, sampling,
                                   run.lmax, spec=spin, nside=args.nside, nchi=args.nchi,
                                   aliases=args.aliases)
                b, w, a = sampled(b), windowed(w), windowed(a)
                stg_floor = spec == 'BB' and proj == 'ProjSTG'
                if stg_floor:
                    # STG has no geometric B. Use the EE normalization for
                    # the BB alias, and EE's window only to display the floor.
                    eb, ew, _ = response(proj, ell, th['ell'], cl, args.healpix_masks, footprint,
                                         sampling, run.lmax, spec='EE', nside=args.nside,
                                         nchi=args.nchi)
                    eb, ew = sampled(eb), windowed(ew)
                    factor = ew/eb
                    norm = reference[(proj, 'EE')]/eb
                else:
                    factor = w/b
                    norm = reference[(proj, spec)]/b
                correction.append(factor)
                # All three terms must share the same R-reference normalization.
                a = a*norm
                b = np.zeros_like(b) if stg_floor else reference[(proj, spec)]
                w = b*factor
                total = w+a
                prefix = f'{name}_{spec}_{proj}'
                outputs[prefix+'_base'] = b
                outputs[prefix+'_windowed'] = w
                outputs[prefix+'_alias'] = a
                outputs[prefix+'_raw'] = measured[j]
                ratio = measured[j]/factor/cl_ee if stg_floor else measured[j]/w
                ratio_total = np.divide(measured[j], total, out=np.full_like(measured[j], np.nan),
                                        where=total > 0)
                print(name, spec, proj, 'floor/input EE' if stg_floor else 'residual', flush=True)
                for lo, hi in ((1000, 2000), (2000, 3000), (3000, 4000), (4000, 4500), (4500, L_WALL),
                               (L_WALL, 5400)):
                    sel = (ELL >= lo) & (ELL < hi)
                    r = ratio[:, sel].mean(-1)
                    rt = ratio_total[:, sel].mean(-1)
                    denom = reference[(proj, 'EE')] if stg_floor else b
                    print(f' {lo}-{hi}: window-only {r.mean():.6f} +/- {r.std(ddof=1)/np.sqrt(len(r)):.6f} '
                          f'window+alias {rt.mean():.6f} +/- {rt.std(ddof=1)/np.sqrt(len(rt)):.6f} '
                          f'window {factor[sel].mean():.6f} '
                          f'alias/{"EE" if stg_floor else "base"} {np.mean(a[sel]/denom[sel]):.6f}',
                          flush=True)
                outputs[prefix+'_ratio'] = ratio
                outputs[prefix+'_ratio_total'] = ratio_total
                if spec == 'BB' and not stg_floor:
                    floor = measured[S.PROJS.index('ProjSTG')]
                    sub = np.where(ELL[None, :] < L_WALL, (measured[j]-floor)/total, np.nan)
                    outputs[prefix+'_ratio_floor_subtracted'] = sub
                    for lo, hi in ((3000, 4000), (4000, 4500), (4500, L_WALL)):
                        r = sub[:, (ELL >= lo) & (ELL < hi)].mean(-1)
                        print(f' partial paired STG subtraction {lo}-{hi}: {r.mean():.6f} '
                              f'+/- {r.std(ddof=1)/np.sqrt(len(r)):.6f}', flush=True)
            outputs[f'{name}_{spec}_correction'] = np.array(correction)
    for name in ('field', 'reference'):
        for spec in ('TT', 'white', 'EE', 'BB'):
            for proj in S.PROJS:
                if spec == 'BB' and proj == 'ProjSTG':
                    continue
                for kind in ('ratio', 'ratio_total'):
                    r = outputs[f'{name}_{spec}_{proj}_{kind}']-outputs[f'{name}_{spec}_ProjZEA_{kind}']
                    for lo, hi in ((1000, 3000), (3000, 4000), (4000, L_WALL)):
                        d = r[:, (ELL >= lo) & (ELL < hi)].mean(-1)
                        print(f'paired {name} {spec} {kind} {proj}-ZEA {lo}-{hi}: {d.mean():.6f} '
                              f'+/- {d.std(ddof=1)/np.sqrt(len(d)):.6f}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out/'pipeline_windows.npz', **outputs)


if __name__ == '__main__':
    main()
