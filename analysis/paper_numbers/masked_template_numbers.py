"""Masked and noisy SFL template macros, paper/masked_template_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/masked_template_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def masked_template_numbers(M=None):
    """Masked/noisy SFL template: isolated autos and split-cross scatter."""
    from masked_template_numbers import numbers
    own = M is None
    if own:
        M = Macros()
    for name, value in numbers().items():
        M.add(name, value)
    if own:
        path = ROOT / 'paper' / 'masked_template_numbers.tex'
        path.write_text(M.tex('masked_template_numbers'))
        print(f'Wrote {path}')


if __name__ == "__main__":
    masked_template_numbers()
