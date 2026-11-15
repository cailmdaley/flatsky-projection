"""The analysis footprints: one family of RA/dec rectangles, one registry.

Seven rectangles centred on the patch centre (RA 0, dec -56), self-similar
to SPT's 1500d field: scale s gives RA half-width 50 s deg and dec
half-width 14 s deg, each bounded by two meridians and two parallels. The
s = 1 member is the 1500d field itself (RA -50 to +50, dec -70 to -42) and
keeps its name; the others are named by their sky fraction, computed here
from the unapodized area (`rect_0p77` is 0.77% of the sky). Every stage
names a footprint by its key here and writes that name into its filenames,
so the units of a footprint-fanned output never collide when lightcone
gathers them into one directory.

Each footprint's flat-sky cut is the tightest even-sided rectangle around
its support, derived per projection (`project_masks.derived_rect_bounds`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Patch centre: the flat-sky grid and every footprint are centred here.
RA_DEG = 0.0
DEC_DEG = -56.0

# The flat-sky grid every projection shares: GRID_PIX x GRID_PIX pixels of
# RES_ARCMIN, centred on the patch centre. 4500 pixels of 2' is 150 deg
# across, which holds the largest rectangle with its taper in every
# projection.
GRID_PIX = 4500
RES_ARCMIN = 2.0

# SPT's 1500d field, the s = 1 member: RA and dec half-widths, deg.
RA_HALF_DEG = 50.0
DEC_HALF_DEG = 14.0
SCALES = (0.45, 0.6, 0.75, 1.0, 1.25, 1.45, 1.65)
FIELD = "field_1500d"


@dataclass(frozen=True)
class Footprint:
    name: str
    scale: float

    @property
    def box(self) -> tuple[float, float, float]:
        """(RA half-width, dec min, dec max), deg."""
        dec_half = DEC_HALF_DEG * self.scale
        return RA_HALF_DEG * self.scale, DEC_DEG - dec_half, DEC_DEG + dec_half

    @property
    def fsky(self) -> float:
        """Unapodized sky fraction of the rectangle."""
        return rect_fsky(self.scale)


def rect_fsky(scale: float) -> float:
    ra_half = math.radians(RA_HALF_DEG * scale)
    dec_lo = math.radians(DEC_DEG - DEC_HALF_DEG * scale)
    dec_hi = math.radians(DEC_DEG + DEC_HALF_DEG * scale)
    return 2 * ra_half * (math.sin(dec_hi) - math.sin(dec_lo)) / (4 * math.pi)


def _name(scale: float) -> str:
    if scale == 1.0:
        return FIELD
    return "rect_" + f"{100 * rect_fsky(scale):.2f}".rstrip("0").rstrip(".").replace(".", "p")


FOOTPRINTS = {f.name: f for f in (Footprint(_name(s), s) for s in SCALES)}
# The sky-fraction sweep: every footprint, in registry (scale) order, and
# their unapodized sky fractions in percent.
SWEEP = list(FOOTPRINTS)
FSKY_PERCENT = [100 * f.fsky for f in FOOTPRINTS.values()]


def ordered(names) -> list[str]:
    """`names` in registry order."""
    names = set(names)
    unknown = names - FOOTPRINTS.keys()
    if unknown:
        raise KeyError(f"unknown footprints {sorted(unknown)}")
    return [n for n in FOOTPRINTS if n in names]
