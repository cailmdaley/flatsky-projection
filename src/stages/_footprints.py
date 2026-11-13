"""The analysis footprints: one registry, eight names.

Seven circular disks concentric with the flat-sky grid sweep the sky
fraction; SPT's 1500d field is the reference footprint. Every stage names a
footprint by its key here and writes that name into its filenames, so the
units of a footprint-fanned output never collide when lightcone gathers them
into one directory.

Each footprint carries its shape on the sphere (a disk radius, or an RA/dec
box about the patch centre's RA), its flat-sky cut, and, for the disks, the
nominal sky fraction they are labelled by. A disk's cut is a fixed square
window `lo:hi` applied to rows and columns in every projection: the 2021
hand-maintained table, which leaves 1 to 8 degrees beyond the apodized disk
and clips the outer taper of the small disks. It is kept because the window
sets the flat-sky Fourier grid and therefore the bandpowers. The field's cut
is None: the tightest even-sided rectangle around its support, derived per
projection (`project_masks.derived_rect_bounds`).
"""

from __future__ import annotations

from dataclasses import dataclass

# Patch centre: the flat-sky grid and every footprint are centred here.
RA_DEG = 0.0
DEC_DEG = -56.0


@dataclass(frozen=True)
class Footprint:
    name: str
    radius_rad: float | None = None                 # disk
    box: tuple[float, float, float] | None = None   # (RA half-width, dec min, dec max), deg
    cut: tuple[int, int] | None = None              # square lo:hi window; None = tightest rectangle
    fsky_nominal: float | None = None               # disks' label, as a fraction

    @property
    def shape(self) -> str:
        return "disk" if self.radius_rad is not None else "box"


FOOTPRINTS = {f.name: f for f in (
    Footprint("disk_0p75", radius_rad=0.175, cut=(1320, 1980), fsky_nominal=0.0075),
    Footprint("disk_1", radius_rad=0.2, cut=(1290, 2010), fsky_nominal=0.01),
    Footprint("disk_1p25", radius_rad=0.225, cut=(1230, 2070), fsky_nominal=0.0125),
    Footprint("disk_1p5", radius_rad=0.25, cut=(1140, 2160), fsky_nominal=0.015),
    Footprint("disk_3", radius_rad=0.35, cut=(960, 2340), fsky_nominal=0.03),
    Footprint("disk_5", radius_rad=0.45, cut=(780, 2520), fsky_nominal=0.05),
    Footprint("disk_10", radius_rad=0.65, cut=(300, 3000), fsky_nominal=0.10),
    Footprint("field_1500d", box=(50.0, -70.0, -42.0)),
)}

DISKS = [name for name, f in FOOTPRINTS.items() if f.shape == "disk"]
FIELD = "field_1500d"


def ordered(names) -> list[str]:
    """`names` in registry order."""
    names = set(names)
    unknown = names - FOOTPRINTS.keys()
    if unknown:
        raise KeyError(f"unknown footprints {sorted(unknown)}")
    return [n for n in FOOTPRINTS if n in names]
