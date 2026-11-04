"""Spin-zero auto-bandpowers on the disk masks or the 1500d field."""
import argparse
from pathlib import Path
from _decisions import (add_common_args, parse_lmax, parse_delta_ell,
                        parse_pixel_window, resolve_seeds, set_threads,
                        write_index, index_name)
from namaster_spectra import (load_workspace_manifest, load_mask, finite_in_mask,
                              compute_pixel_window, atomic_save)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('maps', 'masks', 'workspaces', 'pixel-window'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--projections', nargs='+', required=True)
    p.add_argument('--lmax', type=parse_lmax, required=True)
    p.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    add_common_args(p, seeds=True)
    args = p.parse_args()
    set_threads(args.max_procs)
    import numpy as np
    import pymaster as nmt
    from spt3g import core, maps
    seeds = resolve_seeds(args)
    meta, entries, masks = load_workspace_manifest(Path(args.workspaces), args.projections)
    if (meta['lmax'], meta['delta_ell']) != (args.lmax, args.delta_ell):
        raise ValueError('Scalar workspace binning does not match requested bins')
    lo = np.arange(args.lmax // args.delta_ell) * args.delta_ell
    bins = nmt.NmtBinFlat(lo, lo + args.delta_ell)
    ell = bins.get_effective_ells()
    mode = parse_pixel_window(args.pixel_window)
    correction = compute_pixel_window(mode, bins, ell, args.lmax)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for proj in args.projections:
        for seed in seeds:
            frame = next(core.G3File(str(Path(args.maps) / f'scalar_{proj}_seed{seed:04d}.g3')))
            sky = np.asarray(frame['T'], dtype=np.float64)
            cl = []
            for i in masks:
                mask, window, lx, ly = load_mask(Path(args.masks), proj, i)
                field = nmt.NmtFieldFlat(lx, ly, mask, [finite_in_mask(sky[window], mask)])
                workspace = nmt.NmtWorkspaceFlat()
                workspace.read_from(str(Path(args.workspaces) / entries[proj][i]['w00']))
                cl.append(workspace.decouple_cell(nmt.compute_coupled_cell_flat(field, field, bins))[0] / correction)
            atomic_save(out / f'bandpowers_{proj}_seed{seed:04d}.npz',
                        cl=np.asarray(cl), ell_eff=ell, masks=np.asarray(masks), seed=seed)
            print(f'{proj} seed{seed:04d} done', flush=True)
    write_index(out, {'projections': args.projections, 'seeds': seeds, 'masks': masks,
                      'lmax': args.lmax, 'delta_ell': args.delta_ell, 'pixel_window': mode},
                name=index_name(seeds, args.projections))


if __name__ == '__main__':
    main()
