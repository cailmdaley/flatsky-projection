"""Export the covariance MC's TT/EE source into the worker-transferred src tree.

The input is analysis/theory_spectra.npz (analysis/theory.py provenance).
No CAMB regeneration is performed here.
"""
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'analysis/theory_spectra.npz'
out = ROOT/'src/stages/data/bin_covariance_theory.npz'
out.parent.mkdir(parents=True, exist_ok=True)
with np.load(source) as z:
    np.savez(out, ell=z['ell'][:6001], unlensed_TT=z['unlensed_TT'][:6001],
             unlensed_EE=z['unlensed_EE'][:6001],
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             provenance='analysis/theory.py: Planck 2018 best-fit unlensed spectra; raw C_l [uK^2]')
print(out)
