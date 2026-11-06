"""Compare bin averages with the production isotropic NaMaster response.

This isolates radial bin weighting and within-bin mode-coupling effects.
A projected sky is spatially inhomogeneous and anisotropic, so applying its
footprint-averaged radial spectrum to this response is a diagnostic, not an
exact forward model of that sky. Native windows are retained through coupling.
"""
import argparse
from pathlib import Path
import numpy as np
import pymaster as nmt
import spectra as S
import theory as T
import shear_leakage as SL
import field_1500d as FD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=Path('/sptlocal/user/cailmdaley/pol_project/pipeline_windows_production.npz'))
    p.add_argument('--out', type=Path, default=Path('/sptlocal/user/cailmdaley/pol_project/residual_estimator_response.npz'))
    args = p.parse_args()
    report = np.load(args.report)
    th = T.load()
    ell = np.arange(8001, dtype=float)
    mom = FD.pipeline_moments()
    result = {'ell': S.ELL, 'input_ell': ell}
    for j, proj in enumerate(S.PROJS):
        R = SL.coupling_matrix(mom, 0, j, ell, th['ell'], band_limit=S.LMAX)
        spectra = {spec: R[block] @ th['unlensed_TT' if spec == 'TT' else 'unlensed_EE']
                   for spec, block in [('TT', 'TT<-TT'), ('EE', 'EE<-EE'), ('BB', 'BB<-EE')]}
        del R
        spectra['white'] = np.full_like(ell, report[f'field_white_{proj}_base'][20:60].mean())
        for spin in (0, 2):
            ws = nmt.NmtWorkspaceFlat()
            ws.read_from(str(S.ROOT / 'results/highell/field_coupling' / f'w{spin}{spin}_{proj}_0.fits'))
            specs = ('TT', 'white') if spin == 0 else ('EE', 'BB')
            for spec in specs:
                b = spectra[spec]
                name = f'field_{spec}_{proj}'
                # Smooth multiplicative native sampling attenuation, tabulated
                # by the independently controlled local window calculation.
                if spec == 'BB' and proj == 'ProjSTG':
                    native = np.zeros_like(ell)
                else:
                    ratio = report[name+'_windowed'] / report[name+'_base']
                    native = b * np.interp(ell, S.ELL, ratio) * S.flat_pixwin(ell)
                if spin == 0:
                    cl = native[None, :]
                else:
                    cl = np.zeros((4, len(ell)))
                    cl[0 if spec == 'EE' else 3] = native
                coupled = ws.couple_cell(ell, cl)
                decoupled = ws.decouple_cell(coupled) / S.flat_pixwin()[None, :]
                index = 0 if spec != 'BB' else 3
                result[f'{proj}_{spec}_naive'] = S.bin_theory(b[:S.LMAX])
                result[f'{proj}_{spec}_estimator'] = decoupled[index]
                result[f'{proj}_{spec}_native'] = native
                if spin == 2 and spec == 'EE':
                    result[f'{proj}_EE_to_BB'] = decoupled[3]
                # Constant bandpowers must be invariant to couple/decouple.
                constant = np.ones_like(cl)
                check = ws.decouple_cell(ws.couple_cell(ell, constant))
                result[f'{proj}_{spec}_constant_error'] = np.max(abs(check-1))
                print(proj, spec, 'constant response', np.max(abs(check-1)), flush=True)
                if np.any(b):
                    raw = report[name+'_raw']
                    for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
                        sel = (S.ELL >= lo) & (S.ELL < hi)
                        old = (raw[:, sel]/report[name+'_windowed'][sel]-1).mean(-1)
                        new = (raw[:, sel]/decoupled[index, sel]-1).mean(-1)
                        shift = (decoupled[index, sel]/report[name+'_windowed'][sel]-1).mean()
                        print(f' {lo}-{hi}: old {100*old.mean():+.4f}, response shift {100*shift:+.4f}, new {100*new.mean():+.4f} +/- {100*new.std(ddof=1)/np.sqrt(S.NSIMS):.4f} %', flush=True)
        np.savez(args.out, **result)


if __name__ == '__main__':
    main()
