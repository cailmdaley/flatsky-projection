"""Project the apodized Healpix masks onto the flat sky, then cut each to its
NaMaster analysis window.

Each footprint's mask is reprojected into each flat-sky projection with the
same maps.maputils.healpix_to_flatsky call the map legs use (project_qu.py,
project_eb.py, via _project.project_frame), NaNs outside the projection's
valid domain are zeroed, and the result is cut down to a window -- NaMaster
analyses that window, not the full grid.

The window is the tightest box, not necessarily square or centred, that
contains every pixel with mask > 1e-6, each side rounded up to even; it sets
the Fourier grid (Lx, Ly) and therefore the bandpowers. The rectangles are
about three times wider than tall on the grid, and a square window would
multiply the coupling-matrix cost, which grows roughly as the square of the
window's pixel count. Only the mask's heaviest connected component is kept
(`main_footprint`). Every cut is written with its row and column bounds
(`ylo:yhi`, `xlo:xhi`) and its margin: the fewest pixels between the support
and the grid's edge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _decisions import add_common_args, index_name, parse_interpolation, set_threads, write_index
from _footprints import DEC_DEG, FOOTPRINTS, GRID_PIX, RA_DEG, RES_ARCMIN, ordered
from _project import PROJECTIONS_DEFAULT, project_frame

# A pixel above this weight is part of the footprint's support.
MASK_VALUE_THRESHOLD = 1e-6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--healpix-masks", required=True,
        help="healpix_masks output directory, holding mask_<footprint>.g3.gz",
    )
    parser.add_argument(
        "--footprint", required=True, nargs="+", choices=list(FOOTPRINTS),
        help="footprints to project and cut",
    )
    parser.add_argument("--projections", nargs="+", default=list(PROJECTIONS_DEFAULT))
    parser.add_argument("--interpolation", required=True)
    add_common_args(parser, seeds=False)
    return parser.parse_args(argv)


def main_footprint(mask_arr, threshold=MASK_VALUE_THRESHOLD):
    """`mask_arr` with every component but the heaviest zeroed, and the count dropped.

    A cylindrical grid 150 deg tall centred at dec -56 runs past the south
    pole, and healpix_to_flatsky wraps those rows onto the far side of the
    sphere, so a footprint reaching far south reappears as slivers at the
    grid's bottom corners. Those are not part of the footprint.
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
    footprints = ordered(args.footprint)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_rad_per_pix = RES_ARCMIN / 60.0 * np.pi / 180.0

    records = []
    for name in footprints:
        frame = next(core.G3File(str(Path(args.healpix_masks) / f"mask_{name}.g3.gz")))

        for proj in projections:
            flatsky_frame = project_frame(
                frame, proj, interp=interp_cfg["interp"], rebin=interp_cfg["rebin"])
            arr = np.array(flatsky_frame["T"], dtype=np.float64)  # copy, decoupled from the G3 buffer

            # ApplyMapThreshold (spt3g.mapspectra.basicmaputils) exists in
            # this build, but it unconditionally does frame.pop(k) for k in
            # "TQU" -- a KeyError on a mask frame, which only ever carries
            # 'T'. Zero the NaNs directly with numpy instead. This
            # reproduces what ApplyMapThreshold(frame) would do with its own
            # defaults (vmin=inf, vmax=None -> vmin, vmax = -inf, inf, a
            # no-op range; only the NaN -> vcut=0.0 branch ever fires).
            arr[np.isnan(arr)] = 0.0

            arr, dropped = main_footprint(arr)
            ylo, yhi, xlo, xhi = derived_rect_bounds(arr)
            margin = min(ylo, xlo, GRID_PIX - yhi, GRID_PIX - xhi)

            window = np.ascontiguousarray(arr[ylo:yhi, xlo:xhi])
            lx_rad = (xhi - xlo) * res_rad_per_pix
            ly_rad = (yhi - ylo) * res_rad_per_pix
            bounds = {"ylo": ylo, "yhi": yhi, "xlo": xlo, "xhi": xhi}

            out_path = out_dir / f"mask_{proj}_{name}.npz"
            np.savez(out_path, mask=window, lx_rad=lx_rad, ly_rad=ly_rad, **bounds)
            records.append({
                "projection": proj, "footprint": name, "path": out_path.name,
                **bounds, "lx_rad": lx_rad, "ly_rad": ly_rad,
                "dropped_pixels": dropped, "margin_pix": margin,
            })
        del frame

    manifest = {
        "projections": projections,
        "interpolation": args.interpolation,
        "interp": interp_cfg["interp"],
        "rebin": interp_cfg["rebin"],
        "res_arcmin": RES_ARCMIN,
        "size_pix": GRID_PIX,
        "ra_deg": RA_DEG,
        "dec_deg": DEC_DEG,
        "mask_value_threshold": MASK_VALUE_THRESHOLD,
        "records": records,
    }
    write_index(out_dir, manifest, name=index_name(footprints=footprints))
    return manifest


if __name__ == "__main__":
    main()
