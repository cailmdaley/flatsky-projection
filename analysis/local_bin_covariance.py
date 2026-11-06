"""Leading local-Gaussian, finite-annulus covariance (not an exact full operator).

Keep the position and direction dependence discarded by the radial response R.
For a slowly varying local spectrum P(x,k), the masked Fourier covariance is
approximately FFT_x[W(x)^2 P(x,(k+k')/2)](k-k'). Wick squares that kernel.
Locally straight annulus boundaries give the radial pair-overlap factor
max(0, 1-|d_parallel-separation|/width). This retains finite-bin coupling,
unlike the wide-bin W^4 <P^2> asymptote. Pure-B boundary terms, curvature of
annuli, reciprocal-grid alias images and nonlocal/curvature corrections to
local stationarity are omitted.
Compare coupled bandpowers, not mask-decoupled bandpowers, to this prediction.

"""
import argparse
import json
from pathlib import Path

import numpy as np


def overlap_kernel(power, mask, pixel_size, angles, width, separations):
    """Return Wick pair sums with normalization common to all separations.

    power has shape (ny,nx,nangle); the mask and pixel_size use the same
    coarse spatial grid. Its x coordinate MUST increase along cos(angle),
    and y along sin(angle). Reflect native westward map pixels before calling
    this function with an eastward analytical power. FFT normalization cancels
    in correlations and projected/control fractional-variance ratios.
    """
    ny, nx = mask.shape
    dx = 2*np.pi*np.fft.fftfreq(nx, pixel_size)
    dy = 2*np.pi*np.fft.fftfreq(ny, pixel_size)
    parallel = (dy[:, None, None]*np.sin(angles)[None, None, :]
                + dx[None, :, None]*np.cos(angles)[None, None, :])
    kernel = np.abs(np.fft.fft2(mask[:, :, None]**2*power, axes=(0, 1)))**2
    return np.array([np.mean(np.sum(kernel*np.maximum(
        0, 1-np.abs(parallel-separation)/width), axis=(0, 1)))
        for separation in separations])


def local_power(proj, mask, window, stride, nchi):
    """Geometry sampled in native map coordinates, with sky-area-free weights."""
    import geometry as G
    import pipeline_windows as W
    from bin_covariance_sims import mask_stub
    yy, xx = np.meshgrid(np.arange(stride//2, mask.shape[0], stride),
                         np.arange(stride//2, mask.shape[1], stride), indexing='ij')
    # Include zero-weight pixels so the FFT has the actual footprint geometry.
    sample = mask[yy, xx]
    ra, dec = mask_stub(proj, window).xy_to_angle(xx.astype(float).ravel(),
                                               yy.astype(float).ravel())
    ra, dec = np.asarray(ra), np.asarray(dec)
    jac = G.jacobian(proj, np.degrees(ra), np.degrees(dec))
    angles = (np.arange(nchi)+.5)*2*np.pi/nchi
    u = np.stack([np.cos(angles), np.sin(angles)], -1)
    q_unit = np.einsum('pji,cj->pci', jac, u)
    rotation = W.flattened_rotation(jac)
    delta = np.arctan2(q_unit[..., 1], q_unit[..., 0])+rotation[:, None]-angles
    leakage = np.sin(2*delta)**2
    determinant = np.linalg.det(jac)[:, None]
    shape = sample.shape+(nchi,)

    def evaluate(ell, theory, spec):
        q = ell*q_unit
        qnorm = np.linalg.norm(q, axis=-1)
        cl = theory['unlensed_TT' if spec == 'tt' else 'unlensed_EE']
        p = determinant*np.interp(qnorm, theory['ell'], cl, left=0, right=0)
        p *= qnorm <= 6000
        if spec != 'tt':
            p *= leakage if spec == 'bb' else 1-leakage
        p *= W.interpolation_power(q[..., 0], q[..., 1], dec[:, None])
        p *= W.grid_power(ell*u[:, 0], ell*u[:, 1])[None, :]
        # geometry.jacobian uses eastward x; native pixel indices run west.
        # Reflect both spatial arrays before Fourier transforming their kernel.
        return p.reshape(shape)[:, ::-1, :]
    return sample[:, ::-1], angles, evaluate


def main():
    import sys
    import hashlib
    import spectra as S
    import theory as T
    import pipeline_windows as W
    sys.path.insert(0, str(S.ROOT/'src/stages'))
    from namaster_spectra import load_mask
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--nchi', type=int, default=64)
    p.add_argument('--ells', type=float, nargs='+', default=[2250, 2750, 3750, 4250, 4750])
    p.add_argument('--widths', type=int, nargs='+', default=[30, 50, 100])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    th = T.load()
    rows = []
    for proj in ('ProjZEA', 'ProjSFL', 'ProjSTG'):
        mask, window, _, _ = load_mask(S.ROOT/'results/highell/field_flatsky_masks', proj, 0)
        w, angles, evaluate = local_power(proj, mask, window, args.stride, args.nchi)
        ones = np.ones(w.shape+(args.nchi,))
        controls = {width: overlap_kernel(ones, w, np.deg2rad(2/60)*args.stride,
                                          angles, width, width*np.array([0, 1, 2, 4]))
                    for width in args.widths}
        for spec in ('tt', 'ee', 'bb'):
            if proj == 'ProjSTG' and spec == 'bb':
                continue
            for ell in args.ells:
                power = evaluate(ell, th, spec)
                mean = np.sum(w**2*power.mean(-1))/np.sum(w**2)
                asymptote = np.sum(w**4*(power**2).mean(-1))/np.sum(w**4)/mean**2
                for width in args.widths:
                    v = overlap_kernel(power, w, np.deg2rad(2/60)*args.stride,
                                       angles, width, width*np.array([0, 1, 2, 4]))
                    c = controls[width]
                    row = dict(projection=proj, spectrum=spec, ell=ell, width=width,
                               fractional_variance_ratio=float(v[0]/c[0]/mean**2),
                               wide_bin_ratio=float(asymptote),
                               projected_rho=(v[1:]/v[0]).tolist(),
                               control_rho=(c[1:]/c[0]).tolist())
                    rows.append(row)
                    print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(W.__file__), S.ROOT/'analysis/geometry.py',
               S.ROOT/'analysis/shear_leakage.py', S.ROOT/'analysis/theory_spectra.npz',
               S.ROOT/'results/highell/field_flatsky_masks/.lightcone-manifest.json']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    args.out.write_text(json.dumps(dict(approximation=__doc__, stride=args.stride,
                                       nchi=args.nchi, source_hashes=hashes,
                                       rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
