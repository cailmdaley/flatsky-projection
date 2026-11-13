"""W = 0.5 contours of the disk masks as each projection's grid holds them.

Reads the pipeline's own flat-sky masks (`flatsky_masks`), each a cut window
of the full grid, mean-pools them 4 x 4 (the taper is degrees wide) and
contours W = 0.5 in map-plane degrees from the grid centre. The largest disk's
pooled image is kept too, for figures that contour it themselves.

Run:  bash env.sh python3 analysis/mask_contours.py
          --flatsky-masks results/fiducial/flatsky_masks --out <dir>
Out:  <out>/mask_contours.npz: `{proj}_{footprint}_{k}` (k-th contour path,
      (n, 2) in degrees), `bg_{proj}` (pooled image of `footprints[-1]`),
      `extent` (its (x0, x1, y0, y1) in degrees), and `footprints`, the disks
      in registry order.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import spectra as S

BIN = 4  # mean-pool factor before contouring


def _pool(a, n=BIN):
    ny, nx = a.shape
    return a[: ny // n * n, : nx // n * n].reshape(ny // n, n, nx // n, n).mean((1, 3))


def contours(path, centre, res_deg):
    """(paths, pooled image, extent) of one mask file."""
    with np.load(path) as z:
        arr = _pool(np.asarray(z["mask"], dtype=np.float32))
        x0 = (int(z["xlo"]) - centre) * res_deg
        y0 = (int(z["ylo"]) - centre) * res_deg
    ext = (x0, x0 + arr.shape[1] * BIN * res_deg, y0, y0 + arr.shape[0] * BIN * res_deg)
    x = np.linspace(ext[0], ext[1], arr.shape[1])
    y = np.linspace(ext[2], ext[3], arr.shape[0])
    cs = plt.contour(x, y, arr, levels=[0.5])
    paths = [np.asarray(v, dtype=np.float32) for v in cs.allsegs[0]]
    plt.close("all")
    return paths, arr, np.array(ext)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "flatsky_masks")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    (geometry,) = {(i["res_arcmin"], i["size_deg"]) for i in
                   (json.loads(p.read_text()) for p in args.flatsky_masks.glob("index_*.json"))}
    res_arcmin, size_deg = geometry
    res_deg = res_arcmin / 60
    centre = round(size_deg / res_deg) // 2
    out = {"footprints": S.DISKS}
    for proj in S.PROJS:
        for footprint in S.DISKS:
            paths, arr, ext = contours(args.flatsky_masks / f"mask_{proj}_{footprint}.npz",
                                       centre, res_deg)
            for k, v in enumerate(paths):
                out[f"{proj}_{footprint}_{k}"] = v
        out[f"bg_{proj}"] = arr.astype(np.float16)
        out["extent"] = ext
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "mask_contours.npz", **out)
    print(f"wrote {args.out / 'mask_contours.npz'}")


if __name__ == "__main__":
    main()
