"""Project the apodized Healpix masks onto the flat sky, then cut each to its
NaMaster analysis window.

Each of the seven masks is reprojected into each of the five flat-sky
projections with the same maps.maputils.healpix_to_flatsky call the map legs
use (project_qu.py, project_eb.py, via _project.project_frame), NaNs outside
the projection's valid domain are zeroed, and the result is cut down to a
window -- NaMaster analyses that window, not the full 3300x3300 grid.

Three ways to choose the window:

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
* --rectangular: the tightest box, not necessarily square or centred, that
  contains every pixel with mask > 1e-6, each side rounded up to even. For a
  footprint far from round -- SPT's 1500d field is about three times wider
  than tall on the grid -- a square window would multiply the coupling-matrix
  cost, which grows roughly as the square of the window's pixel count.
  Only the mask's heaviest connected component is kept (`main_footprint`).

Every cut is written with its row and column bounds (`ylo:yhi`, `xlo:xhi`);
a square cut also carries `lo:hi`, which equal both.

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

MASK_INDEX_RE = re.compile(r"(\d+)\.g3(\.gz)?$")

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
        help="directory of Healpix mask .g3/.g3.gz files, one 'T' HealpixSkyMap frame "
             "each, named '*<index>.g3[.gz]' -- make_masks.py's mask_{i}.g3.gz, or the "
             "2021 healpix_mask{i}.g3, both match",
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
    parser.add_argument(
        "--rectangular", action="store_true",
        help="derive the tightest even-sided box around each mask instead of a centred square",
    )
    add_common_args(parser, seeds=False)
    return parser.parse_args(argv)


def discover_masks(healpix_masks_dir):
    """[(index, path), ...] sorted by index, from '*<index>.g3[.gz]' files in a directory."""
    found = []
    for path in sorted(Path(healpix_masks_dir).glob("*.g3*")):
        m = MASK_INDEX_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        raise FileNotFoundError(f"no '<index>.g3[.gz]' files found under {healpix_masks_dir}")
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


def main_footprint(mask_arr, threshold=MASK_VALUE_THRESHOLD):
    """`mask_arr` with every component but the heaviest zeroed, and the count dropped.

    A cylindrical grid 110 deg tall centred at dec -56 runs past the south
    pole, and healpix_to_flatsky wraps those rows onto the far side of the
    sphere, so a field reaching dec -70 reappears as slivers at the grid's
    bottom corners. Those are not part of the footprint.
    """
    import numpy as np
    from scipy import ndimage

    labels, n = ndimage.label(mask_arr > threshold)
    if n == 0:
        raise ValueError("mask has no pixels above threshold; cannot derive a cutout")
    weight = ndimage.sum(mask_arr, labels, index=np.arange(1, n + 1))
    keep = labels == 1 + int(np.argmax(weight))
    dropped = int(np.count_nonzero(labels) - np.count_nonzero(keep))
    return np.where(keep, mask_arr, 0.0), dropped


def derived_rect_bounds(mask_arr, threshold=MASK_VALUE_THRESHOLD):
    """Tightest (ylo, yhi, xlo, xhi) box with mask > threshold, sides even.

    Each side is widened by one pixel on its high edge when odd (or its low
    edge when the high edge is the grid boundary).
    """
    import numpy as np

    ys, xs = np.nonzero(mask_arr > threshold)
    if ys.size == 0:
        raise ValueError("mask has no pixels above threshold; cannot derive a cutout")
    out = []
    for idx, n in ((ys, mask_arr.shape[0]), (xs, mask_arr.shape[1])):
        lo, hi = int(idx.min()), int(idx.max()) + 1
        if (hi - lo) % 2:
            if hi < n:
                hi += 1
            else:
                lo -= 1
        out += [lo, hi]
    return tuple(out)


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

    if args.cut_bounds is not None and args.rectangular:
        raise SystemExit("--cut-bounds and --rectangular are exclusive")
    if args.cut_bounds is not None and len(args.cut_bounds) != len(masks):
        raise SystemExit(
            f"--cut-bounds has {len(args.cut_bounds)} entries but {len(masks)} "
            f"masks were found under {args.healpix_masks}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_rad_per_pix = args.res_arcmin / 60.0 * np.pi / 180.0

    cut_mode = ("rectangular" if args.rectangular
                else "explicit" if args.cut_bounds is not None else "derived")
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

            if args.rectangular:
                arr, dropped = main_footprint(arr)
                ylo, yhi, xlo, xhi = derived_rect_bounds(arr)
                square = {"dropped_pixels": dropped}
            else:
                lo, hi = forced if forced is not None else derived_bounds(arr)
                ylo, yhi, xlo, xhi = lo, hi, lo, hi
                square = {"lo": lo, "hi": hi}

            cut = np.ascontiguousarray(arr[ylo:yhi, xlo:xhi])
            lx_rad = (xhi - xlo) * res_rad_per_pix
            ly_rad = (yhi - ylo) * res_rad_per_pix
            bounds = {"ylo": ylo, "yhi": yhi, "xlo": xlo, "xhi": xhi, **square}

            out_path = out_dir / f"mask_{proj}_{index}.npz"
            np.savez(out_path, mask=cut, lx_rad=lx_rad, ly_rad=ly_rad, **bounds)
            records.append({
                "projection": proj, "mask": index, "path": out_path.name,
                **bounds, "lx_rad": lx_rad, "ly_rad": ly_rad,
                "cut_bounds_mode": cut_mode,
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
        "cut_bounds_mode": cut_mode,
        "cut_bounds_requested": (
            [f"{lo}:{hi}" for lo, hi in args.cut_bounds] if args.cut_bounds is not None else None
        ),
        "records": records,
    }
    write_index(out_dir, manifest)
    return manifest


if __name__ == "__main__":
    main()
