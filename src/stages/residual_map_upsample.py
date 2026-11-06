"""Paired estimator de-aliasing on the *same* projected production Q/U.

No sky is synthesized or reprojected. Fourier interpolation doubles the map
sampling before mask multiplication and pure-B derivatives. Padded/tapered
maps check that a non-periodic crop boundary does not drive the intervention.
Native results must reproduce production before any interpretation.
"""
from pathlib import Path
import argparse
from _decisions import set_threads, write_index
from residual_pure_e_test import refine_periodic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('maps', 'masks', 'workspaces', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--projection', required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    set_threads(2)
    import numpy as np
    import pymaster as nmt
    from spt3g import core, maps  # maps registers FlatSkyMap deserialization
    from scipy.ndimage import map_coordinates
    from namaster_spectra import load_mask, atomic_save, finite_in_mask

    mask, window, lx, ly = load_mask(args.masks, args.projection, 0)
    ny, nx = mask.shape
    frame = next(core.G3File(str(args.maps/f'qu_{args.projection}_seed{args.seed:04d}.g3')))
    full = [np.asarray(frame[key], dtype=float).copy() for key in ('Q', 'U')]
    del frame
    bins = nmt.NmtBinFlat(np.arange(0., 6000., 50.), np.arange(50., 6050., 50.))
    ws = nmt.NmtWorkspaceFlat()
    ws.read_from(str(args.workspaces/f'w22_{args.projection}_0.fits'))
    result = dict(ell=bins.get_effective_ells(), projection=args.projection, seed=args.seed)
    for pad in (0, 64, 128):
        ys, xs = window
        yp, xp = slice(ys.start-pad, ys.stop+pad), slice(xs.start-pad, xs.stop+pad)
        if min(yp.start, xp.start) < 0 or yp.stop > full[0].shape[0] or xp.stop > full[0].shape[1]:
            raise ValueError('Padding exceeds production maps')
        m = np.pad(mask, pad)
        q, u = [finite_in_mask(a[yp, xp].copy(), m) for a in full]
        if pad:
            # The taper is exactly unity on the original crop, hence on all
            # nonzero mask pixels and all production mask derivatives.
            taper_y, taper_x = np.ones(len(q)), np.ones(q.shape[1])
            ramp = np.sin(np.pi/2*np.arange(pad)/pad)**2
            for taper in (taper_y, taper_x):
                taper[:pad], taper[-pad:] = ramp, ramp[::-1]
            envelope = taper_y[:, None]*taper_x[None, :]
            q *= envelope
            u *= envelope
        nyp, nxp = m.shape
        lxp, lyp = lx*nxp/nx, ly*nyp/ny
        for factor in (1, 2):
            if factor == 2:
                yy, xx = np.meshgrid(np.arange(2*nyp)/2, np.arange(2*nxp)/2, indexing='ij')
                fine_mask = np.clip(map_coordinates(m, [yy, xx], order=3, mode='grid-wrap'), 0., 1.)
                del yy, xx
                qr, ur = refine_periodic(q), refine_periodic(u)
            else:
                fine_mask, qr, ur = m, q, u
            field = nmt.NmtFieldFlat(lxp, lyp, fine_mask, [qr, ur], purify_b=True, lite=True)
            coupled = nmt.compute_coupled_cell_flat(field, field, bins)
            # Coupled power scales as integral(mask^2)/map area. Restore the
            # original area normalization before using its fixed workspace.
            coupled *= (lxp*lyp)/(lx*ly)
            result[f'pad{pad}_x{factor}_coupled'] = coupled
            result[f'pad{pad}_x{factor}_decoupled'] = ws.decouple_cell(coupled)
            del field, qr, ur, fine_mask
            print(args.projection, args.seed, pad, factor, 'done', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = f'upsample_{args.projection}_seed{args.seed:04d}.npz'
    atomic_save(args.out/path, **result)
    write_index(args.out, dict(projection=args.projection, seed=args.seed, outputs=[path]),
                name=f'index_{args.projection}_seed{args.seed:04d}.json')


if __name__ == '__main__':
    main()
