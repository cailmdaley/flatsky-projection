"""Realized all-sky T and E spectra of every production input sky.

This separates input-spectrum mismatch from finite-footprint variance: the
mean of these over the realizations is what the 20 production skies actually
contain, against which `residual_estimator_response` conditions its
reference. It is not the spectrum realized on the masked 1500d field.

Run:  bash env.sh python3 analysis/residual_input_spectra.py
          --preprocessed-alms results/fiducial/preprocessed_alms --out <dir>
Out:  <out>/residual_input_spectra.npz: `ell` and `TT`, `EE` as (seed, ell).
"""
import argparse
from pathlib import Path

import healpy as hp
import numpy as np

import spectra as S


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'preprocessed_alms')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.preprocessed_alms.glob('alms_seed*.fits'))
    output = {}
    for spec, hdu in [('TT', 1), ('EE', 2)]:
        cls = []
        for seed, path in enumerate(paths):
            cls.append(hp.alm2cl(hp.read_alm(str(path), hdu=hdu)))
            print(spec, seed, flush=True)
        output[spec] = np.array(cls)
    output['ell'] = np.arange(output['TT'].shape[1])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out/'residual_input_spectra.npz', **output)


if __name__ == '__main__':
    main()
