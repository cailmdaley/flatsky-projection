"""Paired HEALPix synthesis-grid and finite-subsample interventions.

The original E alms, projection, field mask and pure-B estimator stay fixed.
Synthesize at Nside4096 and project with 8x or 16x averaging. Original8192/8x
maps provide the untouched production reference. This diagnoses incoherent
interpolation/averaging B that a coherent multiplicative window cannot remove.
"""
from pathlib import Path
import argparse
from _decisions import set_threads, write_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alms', 'maps', 'masks', 'workspaces', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--projection', required=True)
    args = p.parse_args()
    set_threads(2)
    import numpy as np
    import healpy as hp
    import pymaster as nmt
    from spt3g import core, maps
    from bin_covariance_sims import mask_stub, project_array
    from namaster_spectra import load_mask, finite_in_mask, atomic_save

    ae = hp.read_alm(str(args.alms/f'alms_seed{args.seed:04d}.fits'), hdu=2)
    q, u = hp.alm2map_spin([ae, np.zeros_like(ae)], 4096, 2, 6000)
    del ae
    hq = maps.HealpixSkyMap(q, pol_conv=maps.MapPolConv.COSMO)
    hu = maps.HealpixSkyMap(u, pol_conv=maps.MapPolConv.COSMO)
    del q, u
    mask, window, lx, ly = load_mask(args.masks, args.projection, 0)
    stub = mask_stub(args.projection, window)
    bins = nmt.NmtBinFlat(np.arange(0., 6000., 50.), np.arange(50., 6050., 50.))
    ws = nmt.NmtWorkspaceFlat()
    ws.read_from(str(args.workspaces/f'w22_{args.projection}_0.fits'))
    result = dict(ell=bins.get_effective_ells(), seed=args.seed, projection=args.projection)
    for rebin in (8, 16):
        # Exact same production pixel centres. Only subsample density changes.
        qmap, umap = [project_array(h, stub, dict(interp=True, rebin=rebin)) for h in (hq, hu)]
        maps.flatten_pol(qmap, umap)
        q, u = [finite_in_mask(np.asarray(a, dtype=float), mask) for a in (qmap, umap)]
        field = nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=True, lite=True)
        coupled = nmt.compute_coupled_cell_flat(field, field, bins)
        result[f'n4096_r{rebin}'] = ws.decouple_cell(coupled)
        del field, qmap, umap, q, u
        print(args.projection, args.seed, '4096', rebin, 'done', flush=True)
    del hq, hu
    frame = next(core.G3File(str(args.maps/f'qu_{args.projection}_seed{args.seed:04d}.g3')))
    q, u = [finite_in_mask(np.asarray(frame[key], dtype=float)[window], mask) for key in ('Q', 'U')]
    field = nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=True, lite=True)
    result['n8192_r8'] = ws.decouple_cell(nmt.compute_coupled_cell_flat(field, field, bins))
    args.out.mkdir(parents=True, exist_ok=True)
    path = f'rings_{args.projection}_seed{args.seed:04d}.npz'
    atomic_save(args.out/path, **result)
    write_index(args.out, dict(projection=args.projection, seed=args.seed, outputs=[path]),
                name=f'index_{args.projection}_seed{args.seed:04d}.json')


if __name__ == '__main__':
    main()
