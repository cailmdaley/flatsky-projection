"""Read the `theory_spectra` output: CAMB C_l of the cosmology decision.

`load(directory)` returns the arrays of `<directory>/theory_spectra.npz` --
`ell` and lensed and unlensed TT, EE, BB, TE as raw C_l in uK^2 indexed
directly by multipole -- and the run's parameters as `header`. The stage
that writes it, and its conventions, are in src/stages/theory_spectra.py.
"""

from pathlib import Path

import numpy as np

FILE = "theory_spectra.npz"


def load(directory):
    """The spectra as a dict, header included as a real dict."""
    with np.load(Path(directory) / FILE, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files if k != "header"}
        out["header"] = f["header"][()]
    return out
