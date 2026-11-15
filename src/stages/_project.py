"""Shared machinery for the two per-seed flat-sky projection legs.

``project_qu.py`` and ``project_eb.py`` both: read alms for a seed, synthesize
an intermediate Healpix map, call ``maps.maputils.healpix_to_flatsky`` once
per requested projection, and write one ``.g3`` file per (projection, seed).
This module holds everything that is identical between the two legs -- the
``healpix_to_flatsky`` call, the seed/projection loop, the resumability
check, and the ``index.json`` manifest.

What differs between the legs -- which alm components get synthesized, and
whether ``FlattenPol`` runs afterwards -- is deliberately *not* abstracted
away behind a flag here. It stays in ``project_qu.py`` / ``project_eb.py`` as
an explicit function each script supplies, so the one line that makes the
Q/U leg leaky and the E/B leg leakage-free is visible at its call site
instead of buried in a branch.

Ported from ``/sptlocal/analysis/pol_project2020/workflow/scripts/common.py``
and ``project_QU.py`` / ``project_EB_fullsky.py``.
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
from _footprints import DEC_DEG, RA_DEG

__all__ = [
    "PROJECTIONS_DEFAULT",
    "ALPHA_CENTER_DEG_DEFAULT",
    "Leg",
    "add_project_cli",
    "alm_path",
    "output_path",
    "project_frame",
    "run_leg",
]

PROJECTIONS_DEFAULT = ["ProjSFL", "ProjSTG", "ProjZEA", "ProjBICEP"]

# spt3g.maps.FlatSkyMap's alpha_center defaults to 0 deg when the kwarg is
# omitted -- verified directly against the current build:
#   FlatSkyMap(10, 10, res, proj=..., delta_center=-56*deg).alpha_center == 0
# The 2021 common.py's project() accepted ra_deg but never forwarded it to
# healpix_to_flatsky/FlatSkyMap for exactly this reason: RA 0 is already the
# map default, and RA 0 is the intended patch centre for this whole project.
# We keep that behaviour (project_frame below does not accept ra_deg at all)
# rather than silently reintroducing an alpha_center kwarg that would change
# the geometry, and record the resolved value in every manifest instead.
ALPHA_CENTER_DEG_DEFAULT = 0.0


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
    parser.add_argument("--res-arcmin", type=float, default=2.0)
    parser.add_argument("--size-deg", type=float, default=110.0)
    parser.add_argument("--ra-deg", type=float, default=RA_DEG)
    parser.add_argument("--dec-deg", type=float, default=DEC_DEG)
    add_common_args(parser, seeds=True)


def alm_path(alms_dir, seed: int) -> Path:
    return Path(alms_dir) / f"alms_seed{seed:04d}.fits"


def output_path(out_dir, leg: Leg, proj: str, seed: int) -> Path:
    return Path(out_dir) / f"{leg.name}_{proj}_seed{seed:04d}.g3"


def project_frame(frame, proj: str, *, res_arcmin: float, dec_deg: float,
                   size_deg: float, interp: bool, rebin: int):
    """Reproject every key of a Healpix G3Frame into one flat-sky projection.

    A direct port of the 2021 pipeline's ``common.project()``. Deliberately
    has no ``ra_deg`` parameter -- see ``ALPHA_CENTER_DEG_DEFAULT`` above.

    Heavy imports (``spt3g.core``, ``spt3g.maps``) are deferred to inside this
    function so that importing this module never imports numpy/spt3g before
    ``_decisions.set_threads()`` has had a chance to bound the thread count.
    """
    from spt3g import core, maps

    size_pix = int(round((size_deg * core.G3Units.deg) / (res_arcmin * core.G3Units.arcmin)))
    flatsky_frame = core.G3Frame()
    for key in frame:
        flatsky_frame[key] = maps.maputils.healpix_to_flatsky(
            frame[key],
            x_len=size_pix,
            y_len=size_pix,
            res=res_arcmin * core.G3Units.arcmin,
            proj=maps.MapProjection.names[proj],
            delta_center=dec_deg * core.G3Units.deg,
            interp=interp,
            rebin=rebin,
        )
    return flatsky_frame


def run_leg(leg: Leg, synthesize: Callable, postprocess: Callable, argv=None,
            index_tag=None) -> dict:
    """Drive one projection leg end to end. Called by project_qu.py / project_eb.py.

    ``synthesize(alm_file, nside) -> {key: HealpixSkyMap, ...}`` builds this
    leg's intermediate Healpix maps from one seed's alms.

    ``postprocess(flatsky_frame)`` mutates the freshly projected frame in
    place -- ``maps.FlattenPol`` for the QU leg, nothing at all for the EB
    leg. See the comment at each call site in project_qu.py / project_eb.py
    for why.

    ``index_tag`` prefixes a unit's index name, for an output that holds
    several legs.

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

    size_pix = int(round((args.size_deg * core.G3Units.deg) / (args.res_arcmin * core.G3Units.arcmin)))

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
                healpix_frame, proj,
                res_arcmin=args.res_arcmin, dec_deg=args.dec_deg,
                size_deg=args.size_deg, interp=interp_cfg["interp"],
                rebin=interp_cfg["rebin"],
            )
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
        "res_arcmin": args.res_arcmin,
        "size_deg": args.size_deg,
        "size_pix": size_pix,
        "ra_deg": args.ra_deg,
        "dec_deg": args.dec_deg,
        "alpha_center_deg": ALPHA_CENTER_DEG_DEFAULT,
        "written_this_run": len(written),
        "skipped_existing": len(skipped),
        "files": files,
    }
    write_index(
        out_dir, manifest,
        name=index_name(args.seeds, args.projections if args.seeds else None, tag=index_tag),
    )
    return manifest
