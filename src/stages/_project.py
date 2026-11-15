"""Shared machinery for the two per-seed flat-sky projection legs.

``project_qu.py`` and ``project_eb.py`` both: read alms for a seed, synthesize
an intermediate Healpix map, call ``maps.maputils.healpix_to_flatsky`` once
per requested projection, and write one ``.g3`` file per (projection, seed).
This module holds everything that is identical between the two legs -- the
``healpix_to_flatsky`` call, the seed/projection loop, the resumability
check, and the ``index.json`` manifest.

What differs between the legs -- which alm components get synthesized, and
whether flattening runs afterwards -- is deliberately *not* abstracted
away behind a flag here. It stays in ``project_qu.py`` / ``project_eb.py`` as
an explicit function each script supplies, so the one line that makes the
Q/U leg leaky and the E/B leg leakage-free is visible at its call site
instead of buried in a branch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from _decisions import (
    add_common_args,
    index_name,
    parse_interpolation,
    parse_nside,
    resolve_seeds,
    set_threads,
    write_index,
)
from _footprints import DEC_DEG, GRID_PIX, RA_DEG, RES_ARCMIN

__all__ = [
    "PROJECTIONS_DEFAULT",
    "Leg",
    "add_project_cli",
    "alm_path",
    "output_path",
    "project_frame",
    "flatten",
    "polar_angle",
    "run_leg",
]

PROJECTIONS_DEFAULT = ["ProjSFL", "ProjSTG", "ProjZEA", "ProjBICEP"]

@dataclass(frozen=True)
class Leg:
    """One projection leg. ``name`` doubles as the output filename prefix."""

    name: str        # "qu" | "eb" | "tt" | "white"
    keys: tuple       # frame keys this leg synthesizes, e.g. ("Q", "U")


def add_project_cli(parser: argparse.ArgumentParser) -> None:
    """Flags shared by project_qu.py and project_eb.py (README: project_qu.py / project_eb.py)."""
    parser.add_argument(
        "--alms", required=True,
        help="sky_alms output directory, holding alms_seed{N:04d}.fits",
    )
    parser.add_argument("--projections", nargs="+", default=list(PROJECTIONS_DEFAULT))
    parser.add_argument("--interpolation", required=True)
    parser.add_argument("--nside", required=True)
    add_common_args(parser, seeds=True)


def alm_path(alms_dir, seed: int) -> Path:
    return Path(alms_dir) / f"alms_seed{seed:04d}.fits"


def output_path(out_dir, leg: Leg, proj: str, seed: int) -> Path:
    return Path(out_dir) / f"{leg.name}_{proj}_seed{seed:04d}.g3"


def project_frame(frame, proj: str, *, interp: bool, rebin: int):
    """Reproject every key of a Healpix G3Frame onto the flat-sky grid of `proj`.

    The grid is `_footprints.GRID_PIX` square pixels of `RES_ARCMIN`,
    centred on the patch centre.

    Heavy imports (``spt3g.core``, ``spt3g.maps``) are deferred to inside this
    function so that importing this module never imports numpy/spt3g before
    ``_decisions.set_threads()`` has had a chance to bound the thread count.
    """
    from spt3g import core, maps

    flatsky_frame = core.G3Frame()
    for key in frame:
        flatsky_frame[key] = maps.maputils.healpix_to_flatsky(
            frame[key],
            x_len=GRID_PIX,
            y_len=GRID_PIX,
            res=RES_ARCMIN * core.G3Units.arcmin,
            proj=maps.MapProjection.names[proj],
            alpha_center=RA_DEG * core.G3Units.deg,
            delta_center=DEC_DEG * core.G3Units.deg,
            interp=interp,
            rebin=rebin,
        )
    return flatsky_frame


def polar_angle(fsm, h: float = 1e-3, rows: int = 256):
    """Twice the polar-decomposition rotation angle, 2 psi_R, at every pixel of `fsm`.

    psi_R is the angle of R in J = R S, the polar decomposition of the
    projection's local Jacobian (paper Eq. polar_angle): with delta the
    declination and alpha the RA, as functions of the pixel coordinates (x, y),

        psi_R = atan2(-dd/dx - cos(delta) da/dy, dd/dy - cos(delta) da/dx).

    The derivatives are central differences of `xy_to_angle` with a step of
    `h` pixels, RA differences wrapped, as `maps.flatten_pol` takes its own.
    Computed `rows` grid rows at a time to bound memory.
    """
    import numpy as np
    from spt3g import core

    def angles(x, y):
        a, d = fsm.xy_to_angle(core.DoubleVector(x), core.DoubleVector(y))
        return np.asarray(a), np.asarray(d)

    def wrap(v):
        return (v + np.pi) % (2 * np.pi) - np.pi

    ny, nx = fsm.shape
    out = np.empty((ny, nx))
    for r0 in range(0, ny, rows):
        y, x = (v.ravel() for v in np.mgrid[r0:min(r0 + rows, ny), 0:nx].astype(float))
        a1, d1 = angles(x - h, y)
        a2, d2 = angles(x + h, y)
        a3, d3 = angles(x, y - h)
        a4, d4 = angles(x, y + h)
        _, dec = angles(x, y)
        cd = np.cos(dec)
        dax, ddx = wrap(a2 - a1) / (2 * h), (d2 - d1) / (2 * h)
        day, ddy = wrap(a4 - a3) / (2 * h), (d4 - d3) / (2 * h)
        out[r0:r0 + rows] = (2 * np.arctan2(-ddx - cd * day, ddy - cd * dax)).reshape(-1, nx)
    return out


def flatten(q, u, flattening: str) -> None:
    """Rotate projected Q, U (FlatSkyMaps of one grid) into the grid's frame, in place.

    ``polar`` rotates by the angle of R in the polar decomposition of the
    projection's Jacobian (`polar_angle`), which removes the rotation
    exactly and leaves only the shear. ``spt3g`` is ``maps.flatten_pol``,
    which rotates by the average of the tilts of grid north and east; the two
    agree where the projection is conformal or its stretch axes lie along the
    sky axes or diagonals, and elsewhere differ at second order in the shear.
    Both apply the angle in the same sense: Q + iU -> e^{i rot} (Q + iU), with
    rot negated for COSMO maps.
    """
    from spt3g import maps

    if flattening == "spt3g":
        maps.flatten_pol(q, u)
        return
    import numpy as np

    rot = polar_angle(q)
    if u.pol_conv == maps.MapPolConv.COSMO:
        rot = -rot
    c, s = np.cos(rot), np.sin(rot)
    del rot
    qa, ua = np.asarray(q), np.asarray(u)
    qa[:], ua[:] = c * qa - s * ua, s * qa + c * ua
    q.flat_pol = True
    u.flat_pol = True


def run_leg(leg: Leg, synthesize: Callable, postprocess: Callable, argv=None,
            index_tag=None, record=None) -> dict:
    """Drive one projection leg end to end. Called by project_qu.py / project_eb.py.

    ``synthesize(alm_file, nside) -> {key: HealpixSkyMap, ...}`` builds this
    leg's intermediate Healpix maps from one seed's alms.

    ``postprocess(flatsky_frame)`` mutates the freshly projected frame in
    place -- ``flatten`` for the QU leg, nothing at all for the EB
    leg. See the comment at each call site in project_qu.py / project_eb.py
    for why.

    ``index_tag`` prefixes a unit's index name, for an output that holds
    several legs; ``record`` adds the leg's own settings to the manifest.

    Resumable: a (projection, seed) whose output file already exists is
    skipped, and the (expensive) alm synthesis for a seed is skipped
    entirely when every projection for that seed is already done.
    """
    parser = argparse.ArgumentParser(
        description=f"Project {'/'.join(leg.keys)} onto flat-sky patches (leg={leg.name}).")
    add_project_cli(parser)
    args = parser.parse_args(argv)

    set_threads(args.max_procs)  # must precede numpy/healpy/spt3g imports

    from spt3g import core

    interp_cfg = parse_interpolation(args.interpolation)
    nside = parse_nside(args.nside)
    projections = list(dict.fromkeys(args.projections))  # de-dup, preserve order
    seeds = resolve_seeds(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for seed in seeds:
        todo = [p for p in projections if not output_path(out_dir, leg, p, seed).exists()]
        done_already = [p for p in projections if p not in todo]
        skipped.extend({"projection": p, "seed": seed} for p in done_already)
        if not todo:
            continue

        alm_file = alm_path(args.alms, seed)
        if not alm_file.exists():
            raise FileNotFoundError(f"seed {seed}: missing alms file {alm_file}")

        healpix_maps = synthesize(alm_file, nside)
        healpix_frame = core.G3Frame()
        for key in leg.keys:
            healpix_frame[key] = healpix_maps[key]

        for proj in todo:
            out_path = output_path(out_dir, leg, proj, seed)
            flatsky_frame = project_frame(
                healpix_frame, proj, interp=interp_cfg["interp"], rebin=interp_cfg["rebin"])
            postprocess(flatsky_frame)
            core.G3Writer(filename=str(out_path))(flatsky_frame)
            written.append({"projection": proj, "seed": seed})
        del healpix_frame, healpix_maps

    files = [
        {
            "projection": proj, "seed": seed,
            "path": output_path(out_dir, leg, proj, seed).name,
            "exists": output_path(out_dir, leg, proj, seed).exists(),
        }
        for seed in seeds for proj in projections
    ]
    manifest = {
        "leg": leg.name,
        "keys": list(leg.keys),
        "alms_dir": str(args.alms),
        "projections": projections,
        "seeds_processed": seeds,
        "interpolation": args.interpolation,
        "interp": interp_cfg["interp"],
        "rebin": interp_cfg["rebin"],
        "nside": nside,
        "res_arcmin": RES_ARCMIN,
        "size_pix": GRID_PIX,
        "ra_deg": RA_DEG,
        "dec_deg": DEC_DEG,
        "written_this_run": len(written),
        "skipped_existing": len(skipped),
        "files": files,
        **(record or {}),
    }
    write_index(
        out_dir, manifest,
        name=index_name(args.seeds, args.projections if args.seeds else None, tag=index_tag),
    )
    return manifest
