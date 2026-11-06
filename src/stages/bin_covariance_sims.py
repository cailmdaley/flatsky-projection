"""Field-only Gaussian MC for TT, EE and projection-induced BB covariance.

Synthesizes an independent unlensed sky per unit, using the production
Nside, interpolation, FlattenPol, mask and pure-B estimator. No projected
maps are retained. Unit-width coupled spectra permit later arbitrary binning.
T and E are independent draws: this output tests auto-spectrum covariance,
not TE or covariance between TT and EE. The true sky B is exactly zero.
"""
from pathlib import Path
import argparse
import time

from _decisions import parse_lmax, parse_nside, parse_purify_b, parse_interpolation, set_threads, write_index
from namaster_spectra import load_mask, finite_in_mask, atomic_save


def mask_stub(proj, window):
    """Production grid restricted to precisely the mask's rectangular window."""
    from spt3g import core, maps
    rows, cols = window
    template = maps.FlatSkyMap(3300, 3300, 2*core.G3Units.arcmin,
                               proj=maps.MapProjection.names[proj],
                               delta_center=-56*core.G3Units.deg,
                               pol_conv=maps.MapPolConv.COSMO)
    nx, ny = cols.stop-cols.start, rows.stop-rows.start
    return template.extract_patch(cols.start+nx//2, rows.start+ny//2, nx, ny)


def project_array(hmap, stub, interpolation):
    from spt3g import maps
    return maps.maputils.healpix_to_flatsky(hmap, map_stub=stub, **interpolation)


def run(args):
    set_threads(args.max_procs)
    import numpy as np
    import healpy as hp
    import pymaster as nmt
    from spt3g import maps
    from fine_pseudo_spectra import mode_counts

    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.theory) as z:
        tt, ee = [z[key][:args.lmax+1] for key in ('unlensed_TT', 'unlensed_EE')]
    bins = nmt.NmtBinFlat(np.arange(args.lmax, dtype=float), np.arange(1, args.lmax+1, dtype=float))
    interpolation = parse_interpolation(args.interpolation)
    setup = {}
    for proj in args.projections:
        mask, window, lx, ly = load_mask(args.masks, proj, 0)
        stub = mask_stub(proj, window)
        setup[proj] = (mask, stub, lx, ly)

    # Draw E and scalar T independently, bounding peak full-sky memory.
    # Offset +100000 separates these seeds from all production ensembles.
    np.random.seed(100000+args.seed)
    ae = hp.synalm(ee, lmax=args.lmax)
    q, u = hp.alm2map_spin([ae, np.zeros_like(ae)], args.nside, 2, args.lmax)
    del ae
    hq = maps.HealpixSkyMap(q, pol_conv=maps.MapPolConv.COSMO)
    del q
    hu = maps.HealpixSkyMap(u, pol_conv=maps.MapPolConv.COSMO)
    del u
    pols = {}
    for proj, (mask, stub, lx, ly) in setup.items():
        qmap, umap = [project_array(h, stub, interpolation) for h in (hq, hu)]
        maps.flatten_pol(qmap, umap)
        q, u = [finite_in_mask(np.asarray(m), mask) for m in (qmap, umap)]
        field = nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=args.purify_b, lite=True)
        pols[proj] = nmt.compute_coupled_cell_flat(field, field, bins)
        del field, q, u, qmap, umap
        print(f'{proj} polarization seed {args.seed}: {time.monotonic()-start:.1f}s', flush=True)
    del hq, hu
    at = hp.synalm(tt, lmax=args.lmax)
    t = hp.alm2map(at, args.nside, lmax=args.lmax)
    del at
    ht = maps.HealpixSkyMap(t, pol_conv=maps.MapPolConv.COSMO)
    del t
    paths = []
    for proj, (mask, stub, lx, ly) in setup.items():
        tmap = project_array(ht, stub, interpolation)
        t = finite_in_mask(np.asarray(tmap), mask)
        field = nmt.NmtFieldFlat(lx, ly, mask, [t], lite=True)
        cltt = nmt.compute_coupled_cell_flat(field, field, bins)
        del field, t, tmap
        path = args.out/f'pseudo_{proj}_seed{args.seed:04d}_mask0.npz'
        atomic_save(path, pol=pols[proj], tt=cltt,
                    counts=mode_counts(mask.shape, lx, ly, args.lmax),
                    projection=proj, seed=args.seed, nside=args.nside,
                    lmax=args.lmax, purify_b=args.purify_b,
                    interpolation=args.interpolation, true_b=0., te=0.,
                    lx_rad=lx, ly_rad=ly, pixel_window='none')
        paths.append(path.name)
        print(f'{proj} TT seed {args.seed}: {time.monotonic()-start:.1f}s', flush=True)
    write_index(args.out, dict(seed=args.seed, pseudo_spectra=paths,
                              nside=args.nside, lmax=args.lmax,
                              independent_T_E=True, seconds=time.monotonic()-start),
                name=f'index_seed{args.seed:04d}.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--theory', type=Path, default=Path(__file__).resolve().parent/'data/bin_covariance_theory.npz')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--nside', type=parse_nside, required=True)
    p.add_argument('--lmax', type=parse_lmax, required=True)
    p.add_argument('--purify-b', type=parse_purify_b, required=True)
    p.add_argument('--interpolation', required=True)
    p.add_argument('--projections', nargs='+', default=['ProjZEA', 'ProjSFL', 'ProjSTG'])
    p.add_argument('--max-procs', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
