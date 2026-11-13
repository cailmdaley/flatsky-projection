#!/usr/bin/env python
"""Unit-width TT and pure-B polarization pseudo-spectra from projected QU maps.

No decoupling or pixel-window correction is applied.
Fourier-cell counts make integer-edge rebinning exact, including empty annuli.
All four polarization rows (EE, EB, BE, BB) are retained for later decoupling.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _decisions import index_name, parse_lmax, parse_purify_b, set_threads, write_index
from _footprints import FOOTPRINTS
from namaster_spectra import atomic_save, finite_in_mask, load_mask


def mode_counts(shape, lx, ly, lmax):
    """Full Fourier-grid cell counts in [ell, ell+1), matching NaMaster."""
    import numpy as np

    ny, nx = shape
    kx = np.minimum(np.arange(nx), nx - np.arange(nx)) * (2 * np.pi / lx)
    ky = np.minimum(np.arange(ny), ny - np.arange(ny)) * (2 * np.pi / ly)
    ell = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    # NaMaster's upper bin edge is exclusive, including the final edge.
    return np.bincount(np.floor(ell[ell < lmax]).astype(int), minlength=lmax)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qu-maps', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--tt-maps', type=Path, required=True)
    p.add_argument('--projection', required=True)
    p.add_argument('--footprint', required=True, choices=list(FOOTPRINTS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--purify-b', type=parse_purify_b, required=True)
    p.add_argument('--lmax', type=parse_lmax, required=True)
    p.add_argument('--max-procs', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    set_threads(args.max_procs)
    import numpy as np
    import pymaster as nmt
    from spt3g import core, maps  # noqa: F401

    proj, seed, fp = args.projection, args.seed, args.footprint
    frame = next(core.G3File(str(args.qu_maps / f'qu_{proj}_seed{seed:04d}.g3')))
    tframe = next(core.G3File(str(args.tt_maps / f'tt_{proj}_seed{seed:04d}.g3')))
    bins = nmt.NmtBinFlat(np.arange(args.lmax, dtype=float),
                          np.arange(1, args.lmax + 1, dtype=float))
    args.out.mkdir(parents=True, exist_ok=True)
    mask, window, lx, ly = load_mask(args.masks, proj, fp)
    path = args.out / f'pseudo_{proj}_{fp}_seed{seed:04d}.npz'
    q, u = [finite_in_mask(np.asarray(frame[k], dtype=float)[window], mask)
            for k in ('Q', 'U')]
    t = finite_in_mask(np.asarray(tframe['T'], dtype=float)[window], mask)
    f2 = nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=args.purify_b,
                          lite=True)
    pol = nmt.compute_coupled_cell_flat(f2, f2, bins)
    del f2, q, u
    f0 = nmt.NmtFieldFlat(lx, ly, mask, [t], lite=True)
    tt = nmt.compute_coupled_cell_flat(f0, f0, bins)
    del f0, t
    atomic_save(path, pol=pol, tt=tt,
                counts=mode_counts(mask.shape, lx, ly, args.lmax),
                ell_lo=np.arange(args.lmax), ell_hi=np.arange(1, args.lmax+1),
                projection=proj, seed=seed, footprint=fp,
                purify_b=args.purify_b, pixel_window='none',
                nx=mask.shape[1], ny=mask.shape[0], lx_rad=lx, ly_rad=ly,
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    print(f'{proj} {fp} seed{seed:04d} saved', flush=True)
    write_index(args.out, dict(pseudo_spectra=[path.name], projection=proj, seed=seed,
                              footprint=fp, delta_ell=1, lmax=args.lmax,
                              purify_b=args.purify_b, pixel_window='none'),
                name=index_name([seed], [proj], [fp]))


if __name__ == '__main__':
    main()
