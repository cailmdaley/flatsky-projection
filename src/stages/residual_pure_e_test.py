"""Zero-B flat controls for the production pure-B estimator.

The same native E Fourier coefficients are interpolated onto a twice-finer
periodic grid before masking. This retains precisely the same physical E
field while changing only the sampling of mask products and derivatives.
Coarse and fine masked pseudo-spectra are both decoupled with the production
workspace: a diagnostic of additive numerical contamination, not a new
precision calibration. Unmasked periodic controls gate the zero-B input.
"""
from pathlib import Path
import argparse
from _decisions import set_threads, write_index


def refine_periodic(array):
    from scipy.signal import resample
    return resample(resample(array, 2*array.shape[0], axis=0), 2*array.shape[1], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--workspaces', type=Path, required=True)
    p.add_argument('--theory', type=Path, default=Path('src/stages/data/bin_covariance_theory.npz'))
    p.add_argument('--projection', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    set_threads(2)
    import numpy as np
    import pymaster as nmt
    from namaster_spectra import load_mask, atomic_save
    from scipy.ndimage import map_coordinates

    mask, _, lx, ly = load_mask(args.masks, args.projection, 0)
    ny, nx = mask.shape
    # Reflect original sampling locations exactly, rather than changing the
    # physical support or taper. Padding is already in the production mask.
    yy, xx = np.meshgrid(np.arange(2*ny)/2, np.arange(2*nx)/2, indexing='ij')
    fine_mask = map_coordinates(mask, [yy, xx], order=3, mode='grid-wrap')
    fine_mask = np.clip(fine_mask, 0., 1.)
    del yy, xx
    ell = np.arange(6001, dtype=float)
    with np.load(args.theory) as z:
        ee = z['unlensed_EE'][:len(ell)]
    bins = nmt.NmtBinFlat(np.arange(0., 6000., 50.), np.arange(50., 6050., 50.))
    ws = nmt.NmtWorkspaceFlat()
    ws.read_from(str(args.workspaces/f'w22_{args.projection}_0.fits'))
    output = {'ell': bins.get_effective_ells(), 'projection': args.projection,
              'seed': args.seed, 'mask_shape': mask.shape}
    for cutoff in (4000, 6000):
        cl = ee * (ell <= cutoff)
        # NaMaster uses a constant seed but identical grid / stream layout,
        # so the lower-cutoff control retains exactly the same low modes.
        q, u = nmt.synfast_flat(nx, ny, lx, ly, [cl, np.zeros_like(cl), np.zeros_like(cl)],
                                 [2], seed=900000+1024*args.seed)
        for factor, m in ((1, mask), (2, fine_mask)):
            qr, ur = (q, u) if factor == 1 else (refine_periodic(q), refine_periodic(u))
            f = nmt.NmtFieldFlat(lx, ly, m, [qr, ur], purify_b=True, lite=True)
            coupled = nmt.compute_coupled_cell_flat(f, f, bins)
            output[f'cut{cutoff}_x{factor}_coupled'] = coupled
            output[f'cut{cutoff}_x{factor}_decoupled'] = ws.decouple_cell(coupled)
            del f
            f = nmt.NmtFieldFlat(lx, ly, np.ones_like(m), [qr, ur], purify_b=False, lite=True)
            full = nmt.compute_coupled_cell_flat(f, f, bins)
            output[f'cut{cutoff}_x{factor}_periodic'] = full
            del f, qr, ur
            # Even-sized periodic Nyquist coefficients are ambiguous under
            # interpolation, so test only the in-axis-Nyquist circular bins.
            sel = (output['ell'] >= 500) & (output['ell'] < 4800)
            ratio = np.max(abs(full[3, sel])) / np.mean(full[0, sel])
            print(args.projection, cutoff, factor, 'periodic BB/EE', ratio, flush=True)
            if ratio > 1e-12:
                raise ValueError('Input refinement does not preserve pure E')
    args.out.mkdir(parents=True, exist_ok=True)
    path = f'pure_e_{args.projection}_seed{args.seed:04d}.npz'
    atomic_save(args.out/path, **output)
    write_index(args.out, dict(projection=args.projection, seed=args.seed,
                              outputs=[path]), name=f'index_{args.projection}_seed{args.seed:04d}.json')


if __name__ == '__main__':
    main()
