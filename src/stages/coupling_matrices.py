#!/usr/bin/env python
"""Build NaMaster flat-sky mode-coupling workspaces, per (projection, footprint).

The 2021 pipeline never had a rule for this. Its 35 ``w22_{proj}_disk{i}.fits``
files under ``resources/masks/coupling_matrices/`` were built by hand, off the
DAG, and the generating script is gone. This stage is that missing rule,
reading the flat-sky masks (``project_masks.py``'s
``mask_{proj}_{footprint}.npz`` outputs) instead of the pickled arrays the 2021 code read directly off disk.

Two workspaces per (projection, footprint), not one
-----------------------------------------------------
``namaster_spectra.py`` computes two legs. The Q/U leg builds a genuine
spin-2 field from Q and U and needs the spin-2 coupling matrix (EE, EB, BE,
BB) -- that is the ``w22`` file the 2021 script already wrote and read.

The E/B leg is different in kind, not just in name. Its "E" and "B" maps were
synthesized *directly* from the E- and B-mode alms with ``hp.alm2map`` used as
a spin-0 (scalar) transform and projected with no ``FlattenPol`` -- see
``project_eb.py`` in the README. There is no Q/U tensor anywhere in that leg's
history, so there is nothing for a spin-2 ``NmtFieldFlat`` to decompose: the
"E" map already *is* a scalar field whose auto-power is what a real analysis
would call EE, and likewise for "B" and BB. Feeding ``[E, B]`` into a spin-2
field would ask NaMaster to E/B-decompose a Q/U pair that was never Q/U in the
first place -- it would silently manufacture leakage between two fields that
have no spin-2 relationship, which is exactly the artifact this leg exists to
be free of.

pymaster agrees mechanically, not just physically: ``NmtFieldFlat`` raises
``ValueError`` if ``purify_e``/``purify_b`` is requested for anything but a
spin-2 field (verified against pymaster's ``field.py``), because B-mode
purification is a statement about the spin-2 E/B ambiguity at the mask
boundary -- a scalar field has no such ambiguity to purify. So the EB leg's
auto-spectra are two independent spin-0 measurements,
``NmtFieldFlat(lx, ly, mask, [E])`` and ``NmtFieldFlat(lx, ly, mask, [B])``,
each decoupled through the *scalar* (spin-0/spin-0) mode-coupling matrix,
which is a different linear operator from the spin-2 one (different `ncls`:
1 vs. 4; different mixing kernel -- confirmed empirically against pymaster
2.0 on a toy grid before writing this).

That is why this stage writes ``w00_{proj}_{footprint}.fits`` alongside every
``w22_{proj}_{footprint}.fits``: the EB leg cannot be decoupled without it. Both
workspaces are built from a dummy zero-valued field over the same mask, lx,
ly and binning -- the coupling matrix depends only on the mask and the field
spin, never on the map values (this is also why the 2021 script could use an
all-zero dummy Q/U pair).

Resumable: a workspace FITS already on disk is left alone. Written through a
temporary file and renamed into place, so a kill mid-write cannot leave a
truncated FITS that a later run mistakes for done.

One pair per grid job
---------------------
The build cost spans three orders of magnitude across the footprints (the
raw ``grid/coupling`` logs: 11 min on 8 cores for disk_0p75, 33 h on 32
cores for disk_10), so the pairs never share a job. ``--footprint`` names
the footprint(s) a unit builds (``astra.yaml`` says ``--projections
{units.projection} --footprint {units.footprint}``); the manifest is
written under a coordinate-keyed name so the shards do not collide at the
gather, and ``namaster_spectra.py`` reads them all back through
``read_index_entries``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _decisions import (
    add_common_args,
    index_name,
    parse_delta_ell,
    parse_lmax,
    parse_purify_b,
    set_threads,
    write_index,
)
from _footprints import FOOTPRINTS, ordered


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--masks", required=True, help="flatsky_masks output directory")
    p.add_argument("--projections", nargs="+", required=True)
    p.add_argument(
        "--footprint", nargs="+", required=True, choices=list(FOOTPRINTS),
        help="footprints to build workspaces for",
    )
    p.add_argument("--purify-b", required=True, help="purify | standard | true | false")
    p.add_argument("--lmax", required=True, help="l4000 | l6000 | bare integer")
    p.add_argument("--delta-ell", type=parse_delta_ell, default=10)
    add_common_args(p)
    return p


def atomic_write_workspace(workspace, path: Path) -> None:
    """Write a FITS workspace through a temp file, then rename into place.

    ``NmtWorkspaceFlat.write_to`` hands the filename straight to cfitsio; a
    kill mid-write leaves a truncated file at the *final* path if we write
    there directly, and a later run's existence check would wrongly treat
    that as done. Writing to a sibling temp path and using ``Path.replace``
    (atomic on the same filesystem) closes that gap, matching the pattern
    already used for this project's mask cache (see
    ``analysis/crawford_masks.cached_arrays``).
    """
    tmp = path.with_name(path.stem + ".tmp.fits")
    workspace.write_to(str(tmp))
    tmp.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    set_threads(args.max_procs)

    import numpy as np
    import pymaster as nmt

    purify_b = parse_purify_b(args.purify_b)
    lmax = parse_lmax(args.lmax)
    delta_ell = args.delta_ell
    nbins = lmax // delta_ell

    l0 = (np.arange(nbins) * delta_ell).astype(float)
    lf = ((np.arange(nbins) + 1) * delta_ell).astype(float)
    b = nmt.NmtBinFlat(l0, lf)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = Path(args.masks)

    footprints = ordered(args.footprint)

    entries = []
    for proj in args.projections:
        for name in footprints:
            mask_path = masks_dir / f"mask_{proj}_{name}.npz"
            data = np.load(mask_path)
            mask = np.asarray(data["mask"], dtype=np.float64)
            lx_rad = float(data["lx_rad"])
            ly_rad = float(data["ly_rad"])
            zeros = np.zeros_like(mask)

            w22_path = out_dir / f"w22_{proj}_{name}.fits"
            w00_path = out_dir / f"w00_{proj}_{name}.fits"

            if not w22_path.exists():
                f2 = nmt.NmtFieldFlat(
                    lx_rad, ly_rad, mask, [zeros, zeros], purify_b=purify_b
                )
                w22 = nmt.NmtWorkspaceFlat()
                w22.compute_coupling_matrix(f2, f2, b)
                atomic_write_workspace(w22, w22_path)
                print(f"wrote {w22_path}", flush=True)
            else:
                print(f"skip (exists) {w22_path}", flush=True)

            if not w00_path.exists():
                f0 = nmt.NmtFieldFlat(lx_rad, ly_rad, mask, [zeros])
                w00 = nmt.NmtWorkspaceFlat()
                w00.compute_coupling_matrix(f0, f0, b)
                atomic_write_workspace(w00, w00_path)
                print(f"wrote {w00_path}", flush=True)
            else:
                print(f"skip (exists) {w00_path}", flush=True)

            entries.append(
                {
                    "projection": proj,
                    "footprint": name,
                    "w22": w22_path.name,
                    "w00": w00_path.name,
                    "nx": int(mask.shape[1]),
                    "ny": int(mask.shape[0]),
                    "lx_rad": lx_rad,
                    "ly_rad": ly_rad,
                }
            )

    write_index(
        out_dir,
        {
            "projections": list(args.projections),
            "lmax": lmax,
            "delta_ell": delta_ell,
            "nbins": nbins,
            "purify_b": purify_b,
            "workspaces": entries,
        },
        name=index_name(projections=args.projections, footprints=footprints),
    )


if __name__ == "__main__":
    main()
