#!/usr/bin/env python
"""Build apodized circular Healpix masks marking each analysis footprint.

Every projection is tested against the same set of sky cuts, so the masks
live on the sphere rather than on any one flat-sky grid: a circular disk
around the patch centre, at each of seven radii, apodized with NaMaster's
C1 or C2 taper. Building them on the sphere -- on the same intermediate
Healpix grid the maps are synthesized onto -- means mask and map share a
pixelization before either one is resampled onto the flat sky, so the
projection stage introduces no mismatch between the two.

A disk of a given radius does not translate exactly into a sky fraction:
apodization removes area from the edge of the disk, and how much depends on
the taper scale relative to the disk radius (a bigger bite out of a small
disk than a large one). The 2021 config labelled these masks 0.75% to 10%
from the disk area alone and carried its own comment doubting the numbers.
This script does not trust that label -- it measures and records the
realized sky fraction of the mask actually written to disk, at three
different weightings (unapodized, apodized once, apodized squared), since
different estimators use different powers of the mask and each needs its
own normalization.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _decisions import add_common_args, parse_apodization, parse_nside, set_threads, write_index  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build apodized circular Healpix masks at several radii.",
    )
    parser.add_argument(
        "--nside", required=True, type=parse_nside,
        help="Healpix resolution to build the masks at (accepts "
             "nside4096/nside8192 or a bare integer)",
    )
    parser.add_argument(
        "--apodization", required=True, type=parse_apodization,
        help="apodization kernel and scale (accepts c2_2p5deg/c1_2p5deg/c2_5deg)",
    )
    parser.add_argument("--ra-deg", required=True, type=float, help="patch centre, RA")
    parser.add_argument("--dec-deg", required=True, type=float, help="patch centre, dec")
    parser.add_argument(
        "--radii-rad", required=True, type=float, nargs="+",
        help="disk radii in radians, one mask per value, in the order given",
    )
    add_common_args(parser)
    return parser


def disk_mask(nside, center_vec, radius_rad):
    """A binary Healpix mask: 1 inside the disk of the given radius, 0 outside."""
    import numpy as np
    import healpy as hp

    disk_pixels = hp.query_disc(nside, center_vec, radius_rad)
    mask = np.zeros(hp.nside2npix(nside))
    mask[disk_pixels] = 1.0
    return mask


def apodize(binary_mask, apotype, aposize_deg):
    """Taper a binary mask's edge with NaMaster's C1/C2 kernel.

    aposize is in degrees -- NmtWorkspaceFlat, mask_apodization and the
    decisions table all agree on that unit, but it is easy to pass arcmin
    by habit, so this wrapper exists mainly to name the unit at the call site.
    """
    import pymaster as nmt

    return nmt.mask_apodization(binary_mask, aposize=aposize_deg, apotype=apotype)


def realized_fsky(binary_mask, apodized_mask):
    """The three sky-fraction weightings downstream code needs.

    fsky_binary is the unapodized disk area, for reference against the 2021
    nominal labels. fsky_w1 and fsky_w2 are what pseudo-Cl estimators
    actually normalize by: NaMaster's mode-coupling matrix depends on
    <mask^2> (w2), and a simple mask-weighted mean depends on <mask> (w1).
    Apodization always shrinks these relative to fsky_binary.
    """
    import numpy as np

    return {
        "fsky_binary": float(np.mean(binary_mask)),
        "fsky_w1": float(np.mean(apodized_mask)),
        "fsky_w2": float(np.mean(apodized_mask ** 2)),
    }


def write_mask_g3(apodized_mask, out_path: Path) -> None:
    """Write one mask as a G3Frame(T=HealpixSkyMap), through a temp file.

    Matches the 2021 output format exactly (frame key "T", a HealpixSkyMap
    holding the *apodized* mask) so the existing 2021 products stay readable
    by the same downstream code. Written through a temporary and renamed
    into place so a killed run cannot leave a truncated .g3 that a later
    run, or a downstream reader, would mistake for a complete mask.
    """
    from spt3g import core, maps

    tmp_path = out_path.with_name(out_path.stem + ".tmp.g3")
    tmp_path.unlink(missing_ok=True)

    frame = core.G3Frame()
    frame["T"] = maps.HealpixSkyMap(apodized_mask)
    # Calling the writer directly on one frame runs it and drops the
    # reference immediately, which triggers the writer's destructor and
    # flushes the file -- the same one-shot pattern the 2021 script used.
    core.G3Writer(filename=str(tmp_path))(frame)

    tmp_path.replace(out_path)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # Must precede importing numpy/healpy/pymaster/spt3g: this is what
    # bounds the threaded runtimes those libraries spin up.
    set_threads(args.max_procs)

    import healpy as hp

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    center_vec = hp.ang2vec(args.ra_deg, args.dec_deg, lonlat=True)
    apotype = args.apodization["apotype"]
    aposize_deg = args.apodization["aposize"]

    mask_records = []
    for index, radius_rad in enumerate(args.radii_rad):
        t_start = time.time()

        binary_mask = disk_mask(args.nside, center_vec, radius_rad)
        apodized_mask = apodize(binary_mask, apotype, aposize_deg)
        fsky = realized_fsky(binary_mask, apodized_mask)

        out_path = out_dir / f"mask_{index}.g3"
        write_mask_g3(apodized_mask, out_path)

        wall_time = time.time() - t_start
        print(
            f"mask {index}: radius {radius_rad:.4f} rad -> "
            f"fsky_binary {fsky['fsky_binary']:.5f}  "
            f"fsky_w1 {fsky['fsky_w1']:.5f}  fsky_w2 {fsky['fsky_w2']:.5f}  "
            f"({wall_time:.1f}s)"
        )
        sys.stdout.flush()

        mask_records.append({
            "index": index,
            "file": out_path.name,
            "radius_rad": radius_rad,
            **fsky,
        })

    write_index(out_dir, {
        "stage": "make_masks",
        "nside": args.nside,
        "apotype": apotype,
        "aposize_deg": aposize_deg,
        "ra_deg": args.ra_deg,
        "dec_deg": args.dec_deg,
        "max_procs": args.max_procs,
        "masks": mask_records,
    })


if __name__ == "__main__":
    main()
