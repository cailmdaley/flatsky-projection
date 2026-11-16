"""Decouple the distortion a map projection imposes on flat-sky power spectra."""

from .coupling import BLOCKS, bin_spectrum, coupling_matrix, decouple, namaster_spectra
from .geometry import Geometry, gsq

__all__ = ["BLOCKS", "Geometry", "bin_spectrum", "coupling_matrix", "decouple",
           "gsq", "namaster_spectra"]
