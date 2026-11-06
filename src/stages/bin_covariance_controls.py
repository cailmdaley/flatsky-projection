"""Matched isotropic Gaussian controls for projection covariance.

prepare: radial TT/EE/BB means from the existing 20 projected field skies,
mask-decoupled but with their native sampling window retained.
simulate: independent homogeneous flat-sky skies with those radial powers,
through precisely the same field masks, purification and unit-width bins.
This tests whether a standard homogeneous Gaussian covariance suffices once
the mean spectrum is matched. It is not a simulation of the projection.
"""
from pathlib import Path
import argparse

from _decisions import set_threads, write_index
from namaster_spectra import load_mask, atomic_save

PROJS = ('ProjZEA', 'ProjSFL', 'ProjSTG')


def prepare(args):
    import numpy as np
    import pymaster as nmt
    data = {}
    for proj in PROJS:
        raw = [np.load(args.pseudos/f'pseudo_{proj}_seed{s:04d}_mask0.npz') for s in range(20)]
        counts = raw[0]['counts']
        weight = counts.reshape(120, 50)
        total = weight.sum(-1)
        ell = np.arange(25, 6000, 50.)
        data['ell'] = np.r_[0., ell, 6000.]
        for spin, key in ((0, 'tt'), (2, 'pol')):
            cl = np.mean([r[key] for r in raw], axis=0)
            coupled = (cl.reshape(-1, 120, 50)*weight[None, :, :]).sum(-1)/total[None, :]
            w = nmt.NmtWorkspaceFlat()
            w.read_from(str(args.workspaces/f'w{spin}{spin}_{proj}_0.fits'))
            decoupled = w.decouple_cell(coupled)
            for spec, index in ([('tt', 0)] if spin == 0 else [('ee', 0), ('bb', 3)]):
                power = decoupled[index]
                # The Gaussian controls need a nonnegative spectrum; only the
                # first low-ell band can have a negative numerical-floor BB mean.
                negative = np.flatnonzero(power < 0)
                if np.any(negative >= 2):
                    raise ValueError(f'{proj} {spec}: negative mean outside ell<100')
                power = np.maximum(power, 0)
                data[f'{proj}_{spec}'] = np.r_[0., power, power[-1]]
        for r in raw:
            r.close()
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_save(args.out/'control_spectra.npz', **data)
    write_index(args.out, dict(projections=PROJS, source_seeds=20,
                              native_sampling_window=True, isotropic=True))


def simulate(args):
    import numpy as np
    import pymaster as nmt
    from fine_pseudo_spectra import mode_counts
    import time
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    ell = np.arange(6001, dtype=float)
    bins = nmt.NmtBinFlat(ell[:-1], ell[1:])
    records = []
    with np.load(args.spectra/'control_spectra.npz') as z:
        for j, proj in enumerate(PROJS):
            mask, _, lx, ly = load_mask(args.masks, proj, 0)
            ny, nx = mask.shape
            tt, ee, bb = [np.interp(ell, z['ell'], z[f'{proj}_{spec}'])
                          for spec in ('tt', 'ee', 'bb')]
            zeros = np.zeros_like(ell)
            cls = [tt, zeros, zeros, ee, zeros, bb]
            counts = mode_counts(mask.shape, lx, ly, 6000)
            for seed in range(args.batch*args.batch_size, (args.batch+1)*args.batch_size):
                t, q, u = nmt.synfast_flat(nx, ny, lx, ly, cls, [0, 2],
                                           seed=200000+1000*j+seed)
                f2 = nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=True, lite=True)
                pol = nmt.compute_coupled_cell_flat(f2, f2, bins)
                del f2, q, u
                f0 = nmt.NmtFieldFlat(lx, ly, mask, [t], lite=True)
                cltt = nmt.compute_coupled_cell_flat(f0, f0, bins)
                del f0, t
                path = args.out/f'pseudo_{proj}_seed{seed:04d}_mask0.npz'
                atomic_save(path, pol=pol, tt=cltt, counts=counts,
                            projection=proj, seed=seed, isotropic=True,
                            purify_b=True, pixel_window='none')
                records.append(path.name)
                print(f'{proj} flat control {seed}: {time.monotonic()-start:.1f}s', flush=True)
    write_index(args.out, dict(pseudo_spectra=records, batch=args.batch,
                              batch_size=args.batch_size, isotropic=True,
                              seconds=time.monotonic()-start),
                name=f'index_batch{args.batch:04d}.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'simulate'])
    p.add_argument('--pseudos', type=Path)
    p.add_argument('--workspaces', type=Path)
    p.add_argument('--spectra', type=Path)
    p.add_argument('--masks', type=Path)
    p.add_argument('--batch', type=int)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-procs', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    set_threads(args.max_procs)
    (prepare if args.mode == 'prepare' else simulate)(args)


if __name__ == '__main__':
    main()
