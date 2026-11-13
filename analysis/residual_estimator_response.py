"""The coupling-matrix model of the 1500d field passed through NaMaster's own radial response.

This isolates radial bin weighting and within-bin mode coupling. Per
projection, the forward model of the field (`shear_leakage.coupling_matrix`
on the field's shear moments, band-limited at the run's lmax) is multiplied
by the smooth native sampling window that `pipeline_windows` tabulates
(ring interpolation, rebin and the flat top-hat), then coupled and decoupled
with the production field workspace and divided by the top-hat. A projected
sky is spatially inhomogeneous and anisotropic, so applying its
footprint-averaged radial spectrum to this response is a diagnostic, not an
exact forward model of that sky.

Three references per spectrum: `estimator` reads the sky at integer
multipoles; `spherical` at k = l + 1/2, the leading large-l
spherical-to-plane correspondence (Legendre/Bessel asymptotics), unfitted,
with the local window ratio held fixed; `realized` reads the mean all-sky
spectrum the 20 production inputs actually contain. The white sky's
reference is the flat level of its model over 1000 <= l < 3000. The
spectra are evaluated to INPUT_LMAX, above the corner of the grid's Fourier
plane.

Run:  bash env.sh python3 analysis/residual_estimator_response.py
          --pipeline-windows <pipeline_windows dir>
          --residual-input-spectra <residual_input_spectra dir>
          --theory-spectra data/theory_spectra.npz
          --shear-moments results/fiducial/shear_moments
          --coupling-matrices results/fiducial/coupling_matrices
          --flatsky-masks results/fiducial/flatsky_masks
          --lmax l6000 --delta-ell d50 --out <dir>
Out:  <out>/residual_estimator_response.npz: `ell`, `input_ell`, and per
      projection and spectrum `_naive` (binned model), `_estimator`,
      `_native` (windowed unbinned model), `_spherical`, `_realized`, and
      `{proj}_EE_to_BB`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pymaster as nmt

import shear_leakage as SL
import spectra as S
import theory as T
from _decisions import parse_delta_ell, parse_lmax

INPUT_LMAX = 8000
WHITE_LEVEL = (1000, 3000)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(p, 'pipeline_windows', 'residual_input_spectra', 'theory_spectra', 'shear_moments',
                 'coupling_matrices', 'flatsky_masks')
    p.add_argument('--lmax', type=parse_lmax, required=True)
    p.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run = S.Run.from_args(args)
    lmax, nbins = args.lmax, args.lmax // args.delta_ell
    report = np.load(args.pipeline_windows/'pipeline_windows.npz')
    inputs = np.load(args.residual_input_spectra/'residual_input_spectra.npz')
    th = T.load(args.theory_spectra)
    (res_arcmin,) = {json.loads(q.read_text())['res_arcmin'] for q in args.flatsky_masks.glob('index_*.json')}
    ELL = report['ell']
    ell = np.arange(INPUT_LMAX+1, dtype=float)
    pix_input, pix_bin = S.flat_pixwin(ell, res_arcmin), S.flat_pixwin(ELL, res_arcmin)
    white = (ELL >= WHITE_LEVEL[0]) & (ELL < WHITE_LEVEL[1])
    mom = run.field_moments()
    result = {'ell': ELL, 'input_ell': ell}
    blocks = [('TT', 'TT<-TT'), ('EE', 'EE<-EE'), ('BB', 'BB<-EE')]
    for j, proj in enumerate(S.PROJS):
        R = SL.coupling_matrix(mom, 0, j, ell, th['ell'], band_limit=lmax)
        sky = {spec: th['unlensed_TT' if spec == 'TT' else 'unlensed_EE'] for spec, _ in blocks}
        spectra = {spec: R[block]@sky[spec] for spec, block in blocks}
        shifted = {spec: R[block]@np.interp(th['ell']-.5, th['ell'], sky[spec]) for spec, block in blocks}
        realized = {spec: R[block]@np.interp(th['ell']-.5, inputs['ell'],
                                             inputs['TT' if spec == 'TT' else 'EE'].mean(0), right=0.)
                    for spec, block in blocks}
        del R
        spectra['white'] = np.full_like(ell, report[f'field_white_{proj}_base'][white].mean())
        for spin in (0, 2):
            ws = nmt.NmtWorkspaceFlat()
            ws.read_from(str(args.coupling_matrices/f'w{spin}{spin}_{proj}_{S.FIELD}.fits'))
            specs = ('TT', 'white') if spin == 0 else ('EE', 'BB')
            for spec in specs:
                b = spectra[spec]
                name = f'field_{spec}_{proj}'
                # Smooth multiplicative native sampling attenuation, tabulated
                # by the independently controlled local window calculation.
                if spec == 'BB' and proj == 'ProjSTG':
                    native = np.zeros_like(ell)
                else:
                    ratio = report[name+'_windowed']/report[name+'_base']
                    native = b*np.interp(ell, ELL, ratio)*pix_input
                if spin == 0:
                    cl = native[None, :]
                else:
                    cl = np.zeros((4, len(ell)))
                    cl[0 if spec == 'EE' else 3] = native
                coupled = ws.couple_cell(ell, cl)
                decoupled = ws.decouple_cell(coupled)/pix_bin[None, :]
                index = 0 if spec != 'BB' else 3
                result[f'{proj}_{spec}_naive'] = b[:lmax].reshape(nbins, -1).mean(-1)
                result[f'{proj}_{spec}_estimator'] = decoupled[index]
                result[f'{proj}_{spec}_native'] = native
                if spec != 'white' and not (spec == 'BB' and proj == 'ProjSTG'):
                    for key, model in (('spherical', shifted), ('realized', realized)):
                        moved = np.zeros_like(cl)
                        moved[index] = native*np.divide(model[spec], b, out=np.ones_like(b), where=b != 0)
                        result[f'{proj}_{spec}_{key}'] = (
                            ws.decouple_cell(ws.couple_cell(ell, moved))/pix_bin[None, :])[index]
                if spin == 2 and spec == 'EE':
                    result[f'{proj}_EE_to_BB'] = decoupled[3]
                if not (spec == 'BB' and proj == 'ProjSTG'):
                    raw = report[name+'_raw']
                    print(proj, spec, flush=True)
                    for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
                        sel = (ELL >= lo) & (ELL < hi)
                        old = (raw[:, sel]/report[name+'_windowed'][sel]-1).mean(-1)
                        new = (raw[:, sel]/decoupled[index, sel]-1).mean(-1)
                        shift = (decoupled[index, sel]/report[name+'_windowed'][sel]-1).mean()
                        print(f' {lo}-{hi}: integer-bin {100*old.mean():+.4f}, response shift '
                              f'{100*shift:+.4f}, estimator {100*new.mean():+.4f} '
                              f'+/- {100*new.std(ddof=1)/np.sqrt(len(new)):.4f} %', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out/'residual_estimator_response.npz', **result)


if __name__ == '__main__':
    main()
