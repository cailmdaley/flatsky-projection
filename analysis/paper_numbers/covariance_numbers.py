"""Matched-ensemble bandpower covariance macros, paper/covariance_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/covariance_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def covariance_numbers():
    """Write final matched-ensemble macros; partial comparisons fail closed."""
    from bin_covariance_numbers import numbers

    M = Macros()
    for name, value in numbers().items():
        M.add(name, value)
    path = ROOT / 'paper' / 'covariance_numbers.tex'
    path.write_text(M.tex('covariance_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    covariance_numbers()
