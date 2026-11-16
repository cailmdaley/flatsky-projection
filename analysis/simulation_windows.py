"""The simulation's own windows on the production bandpowers, for the known input.

`spectra.Run.load`, `load_field` and `load_scalar` divide the bandpowers by
these. Each is `pipeline_windows.effective_window`: the ring-interpolation and
finite-rebin windows integrated with the input spectrum over the full local
Jacobian of each footprint and projection, for TT, the white scalar sky, and
EE and BB on both legs. The sampling (synthesis Nside, pixel size, rebin
factor) and the footprints' shapes and apodization come from the masks' index
files; the binning from the band limit and bin width passed in; the spin
blocks' polarization angle from the flattening decision.

Run:  bash env.sh python3 analysis/simulation_windows.py
          --theory-spectra results/fiducial/theory_spectra
          --healpix-masks results/fiducial/healpix_masks
          --flatsky-masks results/fiducial/flatsky_masks
          --lmax l6000 --delta-ell d50 --flattening polar
          --out results/fiducial/simulation_windows
Out:  <out>/simulation_windows.npz: one array per `spectra.window_key(spec,
      leg)`, (footprint, projection, band), with `ell`, `projs` and
      `footprints`.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import pipeline_windows as PW
import spectra as S
from _decisions import parse_delta_ell, parse_flattening, parse_lmax

CASES = ([(spec, S.QU_LEG) for spec in ('TT', 'WHITE', 'EE', 'BB')]
         + [(spec, S.EB_LEG) for spec in ('EE', 'BB')])


def one(job):
    (spec, leg), args = job
    window = PW.effective_window(spec, S.SWEEP, leg, args.theory_spectra, args.healpix_masks,
                                 PW.Sampling.read(args.healpix_masks, args.flatsky_masks),
                                 args.lmax, args.delta_ell, args.flattening)
    print(S.window_key(spec, leg), window.shape, float(window.min()),
          float(window.max()), flush=True)
    return S.window_key(spec, leg), window


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'theory_spectra', 'healpix_masks', 'flatsky_masks')
    ap.add_argument('--lmax', type=parse_lmax, required=True)
    ap.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    ap.add_argument('--flattening', type=parse_flattening, required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    with Pool(args.procs) as pool:
        windows = dict(pool.map(one, [(case, args) for case in CASES], chunksize=1))
    nbins = args.lmax // args.delta_ell
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'simulation_windows.npz', ell=(np.arange(nbins) + .5) * args.delta_ell,
             projs=S.PROJS, footprints=S.SWEEP, **windows)
    print(f"wrote {args.out / 'simulation_windows.npz'}")


if __name__ == '__main__':
    main()
