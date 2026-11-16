"""Mask-weighted shear moments of every analysis footprint, from the mask pixels.

The leakage law and its forward model (`shear_leakage`) see a footprint only
through the distribution of the projection's local Jacobian over it, weighted
by the apodized mask squared. This script evaluates that distribution on the
pixels of the pipeline's own masks -- `healpix_masks`' mask_<footprint>.g3.gz,
at their native Nside -- for every footprint in `_footprints.FOOTPRINTS` and
every projection, and writes it as one file that every consumer reads.

Per (footprint, projection) it stores the exact per-pixel moments

    g2, q2, lnq2     the three reductions of (s1, s2) in `shear_leakage.REDUCTIONS`,
                     weight mask^2
    g2_det, g2_det2  <g^2> under weight mask^2 |det J| and mask^2 |det J|^2
    g2_nopole        <g^2> with a `shear_leakage.POLAR_CUT_DEG` polar cap removed
    sum_w2, sum_w2_det, sum_w2_det2, sum_w2_nopole   the normalisations
    s1_max, s2_min   the extremes of the singular values

and an (n_bins, n_bins) weight histogram in (log s1, log s2) carrying, per
occupied cell, the weighted centroid (cell_s1, cell_s2) and the weighted means
of sin^2(2 Delta) and sin(2 Delta), Delta =
`shear_leakage.flattening_residual` of the `flattening` decision: the
residual rotation the maps' flattening leaves, identically zero under
`polar`. Those cells are what `shear_leakage.predicted_spectra` and
`coupling_matrix` integrate over.

The centroid matters. Placing cells at their geometric midpoints biases <g^2>
by 4% for BICEP at n_bins = 64, because BICEP's magnification spans a factor
of four across the largest footprints and g^2 is strongly convex in log s.
Carrying the first moment inside each cell drops that to 1e-4. Bin assignment
is clipped to edges drawn from a ~200k-pixel subsample; the accumulated log s
values are not clipped, so a pixel outside the subsampled range still lands
at its true position.

Run:  bash env.sh python3 analysis/shear_moments.py
          --healpix-masks results/fiducial/healpix_masks --flattening polar
          --out results/fiducial/shear_moments
Out:  <out>/shear_moments.npz, arrays stacked (footprint, projection, ...) in
      registry and `paper_style.PROJS` order, with `footprints`, `projs`,
      `n_bins` and `flattening`; and <out>/index.json recording the mask
      Nside, the footprints and the flattening.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "stages"))
import geometry as G
import shear_leakage as SL
from _decisions import parse_flattening
from _footprints import FOOTPRINTS
from paper_style import PROJS

N_BINS = 64


def read_mask(path):
    """(nside, ring pixel indices, mask values) of a mask's nonzero pixels."""
    from spt3g import core, maps  # noqa: F401  (maps registers HealpixSkyMap)

    m = next(core.G3File(str(path)))["T"]
    if m.nested:
        raise ValueError(f"{path}: expected a RING map")
    pix, values = m.nonzero_pixels()
    return m.nside, np.asarray(pix, dtype=np.int64), np.asarray(values)


def _bin_edges(proj, alpha, delta, n_bins, pad=0.05):
    """Log-s edges for the (s1, s2) histogram, from a subsample of the mask."""
    s1, s2 = G.singular_values(G.jacobian(proj, alpha, delta))
    out = []
    for s in (s1, s2):
        lo, hi = np.log(s.min()), np.log(s.max())
        span = max(hi - lo, 1e-9)
        out.append(np.linspace(lo - pad * span, hi + pad * span, n_bins + 1))
    return out


def footprint_moments(nside, pix, mask, flattening, n_bins=N_BINS, chunk=2_000_000):
    """Shear moments of one footprint, leading axis in `PROJS` order.

    `pix` are the RING indices of the mask's nonzero pixels at `nside` and
    `mask` their apodized weights; every moment is weighted by `mask`^2.
    `flattening` is the decision's option, which sets the residual rotation.
    """
    import healpy as hp

    w2 = mask ** 2
    npj = len(PROJS)
    out = {k: np.zeros(npj) for k in
           ("sum_w2", "sum_w2_det", "sum_w2_det2", "sum_w2_nopole",
            "g2_det", "g2_det2", "g2_nopole", "s1_max", "s2_min")}
    out["s2_min"][:] = np.inf
    for key in SL.REDUCTIONS:
        out[key] = np.zeros(npj)
    nb2 = n_bins * n_bins
    hist, m1, m2, mf, ms = (np.zeros((npj, nb2)) for _ in range(5))

    stride = max(1, pix.size // 200_000)
    a_s, d_s = hp.pix2ang(nside, pix[::stride], lonlat=True)
    edges = {p: _bin_edges(p, a_s, d_s, n_bins) for p in PROJS}

    for start in range(0, pix.size, chunk):
        sl = slice(start, start + chunk)
        alpha, delta = hp.pix2ang(nside, pix[sl], lonlat=True)
        w = w2[sl]
        off_pole = (90.0 - np.abs(delta)) > SL.POLAR_CUT_DEG
        for j, proj in enumerate(PROJS):
            jac = G.jacobian(proj, alpha, delta)
            s1, s2 = G.singular_values(jac)
            sin_2d = np.sin(2 * SL.flattening_residual(jac, flattening))
            det = s1 * s2
            gg = SL.REDUCTIONS["g2"][1](s1, s2)
            out["sum_w2"][j] += w.sum()
            out["sum_w2_det"][j] += (w * det).sum()
            out["sum_w2_det2"][j] += (w * det ** 2).sum()
            for key, (_, f) in SL.REDUCTIONS.items():
                out[key][j] += (w * f(s1, s2)).sum()
            out["g2_det"][j] += (w * det * gg).sum()
            out["g2_det2"][j] += (w * det ** 2 * gg).sum()
            out["sum_w2_nopole"][j] += w[off_pole].sum()
            out["g2_nopole"][j] += (w[off_pole] * gg[off_pole]).sum()
            out["s1_max"][j] = max(out["s1_max"][j], s1.max())
            out["s2_min"][j] = min(out["s2_min"][j], s2.min())

            l1, l2 = np.log(s1), np.log(s2)
            e1, e2 = edges[proj]
            k1 = np.clip(((l1 - e1[0]) / (e1[1] - e1[0])).astype(np.intp), 0, n_bins - 1)
            k2 = np.clip(((l2 - e2[0]) / (e2[1] - e2[0])).astype(np.intp), 0, n_bins - 1)
            flat = k1 * n_bins + k2
            for acc, x in ((hist, w), (m1, w * l1), (m2, w * l2),
                           (mf, w * sin_2d ** 2), (ms, w * sin_2d)):
                acc[j] += np.bincount(flat, weights=x, minlength=nb2)

    for key in SL.REDUCTIONS:
        out[key] /= out["sum_w2"]
    out["g2_det"] /= out["sum_w2_det"]
    out["g2_det2"] /= out["sum_w2_det2"]
    out["g2_nopole"] /= out["sum_w2_nopole"]

    def cell_mean(acc):
        return np.divide(acc, hist, out=np.zeros_like(acc), where=hist > 0)

    shape = (npj, n_bins, n_bins)
    out["hist"] = hist.reshape(shape)
    out["cell_s1"] = np.exp(cell_mean(m1)).reshape(shape)
    out["cell_s2"] = np.exp(cell_mean(m2)).reshape(shape)
    out["cell_sin2_2delta"] = cell_mean(mf).reshape(shape)
    out["cell_sin_2delta"] = cell_mean(ms).reshape(shape)
    # Empty cells hold exp(0) = 1 in the centroids; consumers select hist > 0.
    return out


def _one(args):
    name, path, flattening = args
    nside, pix, mask = read_mask(path)
    print(f"{name}: Nside {nside}, {pix.size / 1e6:.1f} M pixels", flush=True)
    return name, nside, footprint_moments(nside, pix, mask, flattening)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--healpix-masks", type=Path, required=True,
                    help="directory holding mask_<footprint>.g3.gz")
    ap.add_argument("--flattening", type=parse_flattening, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    names = list(FOOTPRINTS)
    jobs = [(n, args.healpix_masks / f"mask_{n}.g3.gz", args.flattening) for n in names]
    with Pool(args.procs) as pool:
        results = pool.map(_one, jobs, chunksize=1)
    (nside,) = {r[1] for r in results}
    moments = [r[2] for r in results]

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "shear_moments.npz", footprints=names, projs=PROJS,
             n_bins=N_BINS, flattening=args.flattening, **{k: np.stack([m[k] for m in moments]) for k in moments[0]})
    (args.out / "index.json").write_text(json.dumps({
        "nside": nside, "n_bins": N_BINS, "footprints": names, "projs": PROJS,
        "flattening": args.flattening,
    }, indent=2) + "\n")
    print(f"wrote {args.out / 'shear_moments.npz'}")


if __name__ == "__main__":
    main()
