"""Realized all-sky T/E spectra of all 20 original production inputs.

This checks input spectrum mismatch separately from finite-footprint variance.
It is not the spectrum realized on the masked 1500d field.
"""
from pathlib import Path
import numpy as np
import healpy as hp
import spectra as S


def main():
    output = {'ell': np.arange(S.LMAX+1)}
    for spec, hdu in [('TT', 1), ('EE', 2)]:
        cls = []
        for seed in range(S.NSIMS):
            alm = hp.read_alm(str(S.ROOT/'results/highell/preprocessed_alms'/f'alms_seed{seed:04d}.fits'), hdu=hdu)
            cls.append(hp.alm2cl(alm))
            del alm
            print(spec, seed, flush=True)
        output[spec] = np.array(cls)
        np.savez('/sptlocal/user/cailmdaley/pol_project/residual_input_spectra.npz', **output)


if __name__ == '__main__':
    main()
