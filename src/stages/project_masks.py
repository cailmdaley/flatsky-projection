"""Project the apodized Healpix masks onto the flat sky, then cut each to its
NaMaster analysis window.

Each of the seven masks is reprojected into each of the five flat-sky
projections with the same maps.maputils.healpix_to_flatsky call the map legs
use (project_qu.py, project_eb.py, via _project.project_frame), NaNs outside
the projection's valid domain are zeroed, and the result is cut down to a
centred square -- NaMaster analyses that window, not the full 3300x3300 grid.

Two ways to choose the window, selected by whether --cut-bounds is given:

* derived (default): the smallest centred square that contains every pixel
  with mask > 1e-6, rounded up to an even side. Computed independently per
  (projection, mask), since the same disk can occupy a slightly different
  number of pixels in different projections.
* --cut-bounds lo:hi ... (one per mask, in mask-index order): forces a
  window, the same lo:hi applied to every projection for that mask index --
  matching the 2021 hand-maintained table, which used a single lowbound/
  highbound per mask index across all five projections. The baseline
  universe must pass the 2021 table
  (1320:1980 1290:2010 1230:2070 1140:2160 960:2340 780:2520 300:3000)
  because the cut changes the Fourier grid (Lx, Ly) and therefore the
  bandpowers; it is what reproduces the 2021 numbers. The 2021 windows clip
  the outer part of the apodization taper on the small masks, so derived
  bounds for those masks come out wider than the hand table.

Ported from
/sptlocal/analysis/pol_project2020/workflow/scripts/project_masks.py and
resize_masks.py (the cut_frame step, from powerspectra_functions.py).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _decisions import add_common_args, parse_interpolation, set_threads, write_index
from _project import PROJECTIONS_DEFAULT, project_frame

MASK_INDEX_RE = re.compile(r"(\d+)\.g3$")

# "smallest centred square containing every pixel with mask > 1e-6" (README).
MASK_VALUE_THRESHOLD = 1e-6


def _parse_cut_bounds(value: str):
    """'lo:hi' -> (int, int)."""
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--cut-bounds: cannot parse {value!r}; expected 'lo:hi'")
    try:
        lo, hi = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--cut-bounds: cannot parse {value!r}; expected 'lo:hi'") from exc
    if not (0 <= lo < hi):
        raise argparse.ArgumentTypeError(f"--cut-bounds: need 0 <= lo < hi, got {value!r}")
    return lo, hi


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--healpix-masks", required=True,
        help="directory of Healpix mask .g3 files, one 'T' HealpixSkyMap frame each, "
             "named '*<index>.g3' -- make_masks.py's mask_{i}.g3, or the 2021 "
             "healpix_mask{i}.g3, both match",
    )
    parser.add_argument("--projections", nargs="+", default=list(PROJECTIONS_DEFAULT))
    parser.add_argument("--interpolation", required=True)
    parser.add_argument("--res-arcmin", type=float, default=2.0)
    parser.add_argument("--size-deg", type=float, default=110.0)
    parser.add_argument("--ra-deg", type=float, default=0.0)
    parser.add_argument("--dec-deg", type=float, default=-56.0)
    parser.add_argument(
        "--cut-bounds", nargs="+", type=_parse_cut_bounds, default=None, metavar="lo:hi",
        help="one 'lo:hi' per mask, in mask-index order, applied to every projection for "
             "that mask -- e.g. the 2021 table 1320:1980 1290:2010 1230:2070 1140:2160 "
             "960:2340 780:2520 300:3000. Omit for the derived default (see module docstring).",
    )
    add_common_args(parser, seeds=False)
    return parser.parse_args(argv)


def discover_masks(healpix_masks_dir):
    """[(index, path), ...] sorted by index, from '*<index>.g3' files in a directory."""
    found = []
    for path in sorted(Path(healpix_masks_dir).glob("*.g3")):
        m = MASK_INDEX_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        raise FileNotFoundError(f"no '<index>.g3' files found under {healpix_masks_dir}")
    found.sort(key=lambda t: t[0])
    return found


def derived_bounds(mask_arr, threshold=MASK_VALUE_THRESHOLD):
    """Smallest centred square with mask > threshold, side rounded up to even.

    'Centred' is the map's own pixel-grid centre (size // 2 -- 1650 for the
    3300-pixel grid this project always uses), which is where
    healpix_to_flatsky places (ra_deg, dec_deg) since alpha_center and
    delta_center are set to the map centre by construction. Returns
    (lo, hi) such that mask_arr[lo:hi, lo:hi] contains every pixel above
    threshold, with hi - lo guaranteed even because it is built as twice a
    single integer half-width.
    """
    import numpy as np

    ny, nx = mask_arr.shape
    if ny != nx:
        raise ValueError(f"expected a square map, got {mask_arr.shape}")
    center = ny // 2
    ys, xs = np.nonzero(mask_arr > threshold)
    if ys.size == 0:
        raise ValueError("mask has no pixels above threshold; cannot derive a cutout")
    i_min = int(min(ys.min(), xs.min()))
    i_max = int(max(ys.max(), xs.max()))
    # Half-width needed to keep pixel i_min inside [lo, hi) on the low side
    # and pixel i_max inside on the high side, both measured from `center`.
    half = max(center - i_min, i_max - center + 1)
    return center - half, center + half


def main(argv=None):
    args = parse_args(argv)
    set_threads(args.max_procs)  # must precede numpy/healpy/spt3g imports

    import numpy as np
    # maps must be imported (not just used through _project.project_frame)
    # so that spt3g's cereal polymorphic-type registry knows about
    # HealpixSkyMap before we deserialize one below.
    from spt3g import core, maps  # noqa: F401

    interp_cfg = parse_interpolation(args.interpolation)
    projections = list(dict.fromkeys(args.projections))
    masks = discover_masks(args.healpix_masks)

    if args.cut_bounds is not None and len(args.cut_bounds) != len(masks):
        raise SystemExit(
            f"--cut-bounds has {len(args.cut_bounds)} entries but {len(masks)} "
            f"masks were found under {args.healpix_masks}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_rad_per_pix = args.res_arcmin / 60.0 * np.pi / 180.0

    records = []
    for order, (index, mask_path) in enumerate(masks):
        frame = next(core.G3File(str(mask_path)))
        forced = args.cut_bounds[order] if args.cut_bounds is not None else None

        for proj in projections:
            flatsky_frame = project_frame(
                frame, proj,
                res_arcmin=args.res_arcmin, dec_deg=args.dec_deg,
                size_deg=args.size_deg, interp=interp_cfg["interp"],
                rebin=interp_cfg["rebin"],
            )
            arr = np.array(flatsky_frame["T"], dtype=np.float64)  # copy, decoupled from the G3 buffer

            # ApplyMapThreshold (spt3g.mapspectra.basicmaputils) exists in
            # this build, but it unconditionally does frame.pop(k) for k in
            # "TQU" -- a KeyError on a mask frame, which only ever carries
            # 'T'. Zero the NaNs directly with numpy instead. This
            # reproduces what ApplyMapThreshold(frame) would do with its own
            # defaults (vmin=inf, vmax=None -> vmin, vmax = -inf, inf, a
            # no-op range; only the NaN -> vcut=0.0 branch ever fires).
            arr[np.isnan(arr)] = 0.0

            if forced is not None:
                lo, hi = forced
            else:
                lo, hi = derived_bounds(arr)

            cut = np.ascontiguousarray(arr[lo:hi, lo:hi])
            lx_rad = (hi - lo) * res_rad_per_pix
            ly_rad = lx_rad  # square cut: same bounds on both axes

            out_path = out_dir / f"mask_{proj}_{index}.npz"
            np.savez(
                out_path,
                mask=cut,
                lo=lo, hi=hi, lx_rad=lx_rad, ly_rad=ly_rad,
            )
            records.append({
                "projection": proj, "mask": index, "path": out_path.name,
                "lo": lo, "hi": hi, "lx_rad": lx_rad, "ly_rad": ly_rad,
                "cut_bounds_mode": "explicit" if forced is not None else "derived",
            })
        del frame

    manifest = {
        "healpix_masks_dir": str(args.healpix_masks),
        "mask_indices": [i for i, _ in masks],
        "projections": projections,
        "interpolation": args.interpolation,
        "interp": interp_cfg["interp"],
        "rebin": interp_cfg["rebin"],
        "res_arcmin": args.res_arcmin,
        "size_deg": args.size_deg,
        "ra_deg": args.ra_deg,
        "dec_deg": args.dec_deg,
        "mask_value_threshold": MASK_VALUE_THRESHOLD,
        "cut_bounds_mode": "explicit" if args.cut_bounds is not None else "derived",
        "cut_bounds_requested": (
            [f"{lo}:{hi}" for lo, hi in args.cut_bounds] if args.cut_bounds is not None else None
        ),
        "records": records,
    }
    write_index(out_dir, manifest)
    return manifest


if __name__ == "__main__":
    main()
