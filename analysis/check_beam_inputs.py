"""Bound the retained harmonic B residual in the E-only beam experiment."""

import json
from pathlib import Path

import healpy as hp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def run():
    e,b=hp.read_alm(str(ROOT/'results'/'highell'/'preprocessed_alms'/'alms_seed0000.fits'),
                     hdu=(2,3))
    ce,cb=hp.alm2cl(e),hp.alm2cl(b)
    ell=np.arange(len(ce))
    out={}
    for lo,hi in ((500,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6001)):
        keep=(ell>=lo)&(ell<hi)
        ratio=float(np.sum((2*ell[keep]+1)*cb[keep])
                    /np.sum((2*ell[keep]+1)*ce[keep]))
        out[f'{lo}_{hi}']=ratio
        assert ratio<1e-12, 'The E-only reference approximation requires negligible retained B.'
    (ROOT/'paper'/'beam_input_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    run()
