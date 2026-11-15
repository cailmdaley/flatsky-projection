"""Local-affine reciprocal-grid prediction, including spin rotation on folding.

This is an analytic band-limit intervention, not a rerun of spherical maps;
the real-projector controls are `pipeline_window_controls`. The grid sampling
theorem puts copies of the sky at the reciprocal-lattice spacing along each
map axis, and a sky cutoff maps to an ellipse, not a circle, on a sheared
projection. For the 1500d field, each projection and each of TT, the white
scalar sky, EE and BB, `pipeline_windows.response` gives the unwindowed
spectrum, the windowed one and the folded images, at two sky cutoffs: the
production band limit and the cutoff of the `window_bandlimit_test`
intervention. Only complete Fourier circles are predicted, so the bins run
from 3000 to the axis Nyquist multipole.

Run:  bash env.sh python3 analysis/report_grid_aliases.py
          --theory-spectra results/fiducial/theory_spectra
          --healpix-masks results/fiducial/healpix_masks
          --flatsky-masks results/fiducial/flatsky_masks
          --lmax l6000 --delta-ell d50 --test-cutoff 5000 --out <dir>
Out:  <out>/pipeline_aliases.npz: `ell` (bin centres), `projs`, and
      `{cutoff}_{proj}_{spec}_{base,windowed,alias}` per bin.
"""
import argparse
from pathlib import Path

import numpy as np

import spectra as S
import theory as T
from _decisions import parse_delta_ell, parse_lmax
from pipeline_windows import Sampling, response

ELL_MIN = 3000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'theory_spectra', 'healpix_masks', 'flatsky_masks')
    ap.add_argument('--lmax', type=parse_lmax, required=True)
    ap.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    ap.add_argument('--test-cutoff', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    th = T.load(args.theory_spectra)
    sampling = Sampling.read(args.healpix_masks, args.flatsky_masks)

    centres = (np.arange(args.lmax // args.delta_ell)+.5)*args.delta_ell
    # Only complete Fourier circles are predicted. Above the axis Nyquist,
    # a radial estimator samples square corners rather than all azimuths.
    bins = centres[(centres >= ELL_MIN) & (centres < sampling.nyquist)]
    step = args.delta_ell/5
    ell = np.arange(ELL_MIN+step/2, sampling.nyquist, step)
    pix_fine = S.flat_pixwin(ell, sampling.res_arcmin)
    pix_bin = S.flat_pixwin(bins, sampling.res_arcmin)
    out = {'ell': bins, 'projs': S.PROJS}
    for cutoff in (args.lmax, args.test_cutoff):
        for proj in S.PROJS:
            for spec in ('TT', 'white', 'EE', 'BB'):
                cl = (np.where(th['ell'] >= 2, 1., 0.) if spec == 'white'
                      else th['unlensed_TT' if spec == 'TT' else 'unlensed_EE'])
                b, w, a = response(proj, ell, th['ell'], cl, args.healpix_masks, S.FIELD, sampling,
                                   cutoff, spec='TT' if spec == 'white' else spec,
                                   nside=32, nchi=128, aliases=True)
                b = b.reshape(-1, 5).mean(-1)
                w = (w*pix_fine).reshape(-1, 5).mean(-1)/pix_bin
                a = (a*pix_fine).reshape(-1, 5).mean(-1)/pix_bin
                key = f'{cutoff}_{proj}_{spec}'
                out[key+'_base'] = b
                out[key+'_windowed'] = w
                out[key+'_alias'] = a
                print(key, flush=True)
                for l in (3500, 4000, 4500, 4800, 5000, 5375):
                    k = np.argmin(abs(bins-l))
                    relative = a[k]/w[k] if w[k] > 1e-30 else np.nan
                    print(f' {l}: base {b[k]:.8g} window {w[k]:.8g} alias {a[k]:.8g} '
                          f'relative {relative:.6g}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out/'pipeline_aliases.npz', **out)


if __name__ == '__main__':
    main()
