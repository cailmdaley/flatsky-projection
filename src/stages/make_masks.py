#!/usr/bin/env python
"""Build apodized Healpix masks marking each analysis footprint.

Every projection is tested against the same set of sky cuts, so the masks
live on the sphere rather than on any one flat-sky grid. Each `--footprint`
names an entry of `_footprints.FOOTPRINTS`: a circular disk around the patch
centre, or an RA/dec box (SPT's 1500d field), apodized with NaMaster's C1 or
C2 taper. Building them on the sphere -- on the same intermediate
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

from _decisions import (  # noqa: E402
    add_common_args, index_name, parse_apodization, parse_nside, set_threads, write_index,
)
from _footprints import DEC_DEG, FOOTPRINTS, RA_DEG, ordered  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build apodized Healpix masks, one per named footprint.",
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
    parser.add_argument(
        "--footprint", required=True, nargs="+", choices=list(FOOTPRINTS),
        help="footprints to build, each written as mask_<footprint>.g3.gz",
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


def apodize_disk_analytic(nside, center_vec, radius_rad, apotype, aposize_deg):
    """C1/C2-apodized circular disk, computed analytically in radius.

    nmt.mask_apodization segfaults inside _nmtlib at nside 8192 (805M pixels;
    the C neighbour search appears to overflow), so for the one mask shape this
    project uses -- a circular disk -- the taper is evaluated in closed form
    instead. NaMaster's kernels, per its documentation, weight a pixel at
    angular distance theta from the nearest unobserved pixel by
        x = sqrt((1 - cos(theta)) / (1 - cos(theta_*)))
        C1: w = x - sin(2 pi x) / (2 pi)     (x < 1, else 1)
        C2: w = (1 - cos(pi x)) / 2          (x < 1, else 1)
    For a disk of radius R the distance to the boundary is R - r, with r the
    pixel's angular distance from the centre. The one approximation relative to
    nmt is the definition of the edge: nmt measures to the nearest *pixel
    centre* outside the disk, this measures to the geometric boundary -- a
    half-pixel (~0.2' at nside 8192) offset over a 2.5 deg taper, validated
    at nside 4096 against nmt's own output before adoption.
    """
    import numpy as np
    import healpy as hp

    disk_pixels = hp.query_disc(nside, center_vec, radius_rad)
    vecs = np.array(hp.pix2vec(nside, disk_pixels))
    r = np.arccos(np.clip(np.dot(center_vec, vecs), -1.0, 1.0))
    del vecs

    w = taper(radius_rad - r, apotype, aposize_deg)  # distance to the disk boundary
    mask = np.zeros(hp.nside2npix(nside))
    mask[disk_pixels] = w
    return mask


def taper(theta, apotype, aposize_deg):
    """NaMaster's C1/C2 weight at angular distance `theta` (rad) inside the edge."""
    import numpy as np

    aposize_rad = np.radians(aposize_deg)
    x = np.sqrt((1.0 - np.cos(theta)) / (1.0 - np.cos(aposize_rad)))
    w = np.ones_like(x)
    t = x < 1.0
    if apotype == "C2":
        w[t] = 0.5 * (1.0 - np.cos(np.pi * x[t]))
    elif apotype == "C1":
        w[t] = x[t] - np.sin(2 * np.pi * x[t]) / (2 * np.pi)
    else:
        raise ValueError(f"analytic apodization only supports C1/C2, got {apotype}")
    return w


def rect_mask_analytic(nside, ra_center_deg, ra_half_deg, dec_min_deg,
                        dec_max_deg, apotype, aposize_deg):
    """(pixels, weights) of a C1/C2-apodized RA/dec rectangle, tapered analytically.

    The rectangle is bounded by two meridians and two parallels -- constant
    limits in the coordinates, which is how survey fields are defined. The
    taper is NaMaster's kernel evaluated at each pixel's geodesic distance to
    the boundary, in closed form, for the same reason as the disk:
    nmt.mask_apodization segfaults at nside 8192. For an interior pixel that
    distance is the smallest of four: to each parallel, |dec - dec_edge|
    (along the meridian, whose foot always lies on the edge), and to each
    meridian's great circle, arcsin(cos dec |sin(ra - ra_edge)|). Where that
    great-circle foot falls past a corner it lies outside the field, so the
    geodesic to it crosses the boundary and a parallel is nearer anyway; the
    minimum over the four is therefore the exact distance to the boundary.
    Same half-pixel edge convention as `apodize_disk_analytic`. Returned
    sparse: the rectangle is a few percent of an 805M-pixel sky at nside 8192.
    """
    import numpy as np
    import healpy as hp

    theta_lo = np.radians(90.0 - dec_max_deg)
    theta_hi = np.radians(90.0 - dec_min_deg)
    strip = hp.query_strip(nside, theta_lo, theta_hi, inclusive=False)
    ra, dec = hp.pix2ang(nside, strip, lonlat=True)
    dra = (ra - ra_center_deg + 180.0) % 360.0 - 180.0
    keep = (np.abs(dra) <= ra_half_deg) & (dec >= dec_min_deg) & (dec <= dec_max_deg)
    pix, dra, dec = strip[keep], np.radians(dra[keep]), np.radians(dec[keep])
    del strip, ra, keep

    half = np.radians(ra_half_deg)
    cosd = np.cos(dec)
    dist = np.minimum(dec - np.radians(dec_min_deg), np.radians(dec_max_deg) - dec)
    for edge in (half, -half):
        dist = np.minimum(dist, np.arcsin(np.clip(cosd * np.abs(np.sin(dra - edge)), 0.0, 1.0)))
    del cosd, dra, dec

    return pix, taper(dist, apotype, aposize_deg)


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
    """Write one mask as a G3Frame(T=HealpixSkyMap), gzipped, through a temp file.

    Same frame layout as the 2021 products (key "T", a HealpixSkyMap holding
    the *apodized* mask) so the same downstream reader serves both. The map
    is stored ring-sparse and the file gzipped: a footprint mask is zero over
    ~90% of the sky and exactly 1.0 over most of the rest, so a dense .g3 is
    1.6 GB at nside 4096 and 6.4 GB at nside 8192 while the .g3.gz is ~20 MB
    and ~80 MB -- and HTCondor transfers apparent bytes, so the dense form
    made the nside-8192 stage untransferable on OSPool. G3File reads .g3.gz
    transparently; the round trip is bit-exact. Written through a temporary
    and renamed into place so a killed run cannot leave a truncated file that
    a later run, or a downstream reader, would mistake for a complete mask.
    """
    from spt3g import core, maps

    tmp_path = out_path.with_name(out_path.name.replace(".g3.gz", ".tmp.g3.gz"))
    tmp_path.unlink(missing_ok=True)

    mask_map = maps.HealpixSkyMap(apodized_mask)
    mask_map.ringsparse = True
    frame = core.G3Frame()
    frame["T"] = mask_map
    # Calling the writer directly on one frame runs it and drops the
    # reference immediately, which triggers the writer's destructor and
    # flushes the file -- the same one-shot pattern the 2021 script used.
    core.G3Writer(filename=str(tmp_path))(frame)

    tmp_path.replace(out_path)


def build_mask(fp, nside, apotype, aposize_deg):
    """(apodized mask, sky fractions, shape record) for one footprint."""
    import healpy as hp
    import numpy as np

    if fp.shape == "disk":
        center_vec = hp.ang2vec(RA_DEG, DEC_DEG, lonlat=True)
        binary_mask = disk_mask(nside, center_vec, fp.radius_rad)
        apodized_mask = apodize_disk_analytic(
            nside, center_vec, fp.radius_rad, apotype, aposize_deg)
        return apodized_mask, realized_fsky(binary_mask, apodized_mask), {
            "radius_rad": fp.radius_rad}

    ra_half, dec_min, dec_max = fp.box
    pix, w = rect_mask_analytic(nside, RA_DEG, ra_half, dec_min, dec_max, apotype, aposize_deg)
    npix = hp.nside2npix(nside)
    fsky = {"fsky_binary": pix.size / npix,
            "fsky_w1": float(w.sum()) / npix,
            "fsky_w2": float((w * w).sum()) / npix}
    apodized_mask = np.zeros(npix)
    apodized_mask[pix] = w
    return apodized_mask, fsky, {
        "ra_half_deg": ra_half, "dec_min_deg": dec_min, "dec_max_deg": dec_max}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # Must precede importing numpy/healpy/pymaster/spt3g: this is what
    # bounds the threaded runtimes those libraries spin up.
    set_threads(args.max_procs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    apotype = args.apodization["apotype"]
    aposize_deg = args.apodization["aposize"]
    footprints = ordered(args.footprint)

    mask_records = []
    for name in footprints:
        fp = FOOTPRINTS[name]
        t_start = time.time()
        apodized_mask, fsky, shape = build_mask(fp, args.nside, apotype, aposize_deg)
        out_path = out_dir / f"mask_{name}.g3.gz"
        write_mask_g3(apodized_mask, out_path)
        del apodized_mask
        print(
            f"{name}: fsky_binary {fsky['fsky_binary']:.5f}  "
            f"fsky_w1 {fsky['fsky_w1']:.5f}  fsky_w2 {fsky['fsky_w2']:.5f}  "
            f"({time.time() - t_start:.1f}s)",
            flush=True,
        )
        mask_records.append({
            "footprint": name, "file": out_path.name, "shape": fp.shape, **shape, **fsky,
        })

    write_index(out_dir, {
        "stage": "make_masks",
        "nside": args.nside,
        "apotype": apotype,
        "aposize_deg": aposize_deg,
        "ra_deg": RA_DEG,
        "dec_deg": DEC_DEG,
        "masks": mask_records,
    }, name=index_name(footprints=footprints))


if __name__ == "__main__":
    main()
