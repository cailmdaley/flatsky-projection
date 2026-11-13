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

Evaluated on each chart's 1500d field mask (`flatsky_masks`) for the unlensed
TT and EE of `theory_spectra`, band-limited at the sky's ell_max, with the
ring-interpolation and finite-rebin sampling the masks' index files record.
`--stride` and `--nchi` set the spatial and angular quadrature; `--ells` the
bin centres and `--widths` the bin widths evaluated.

Run:  bash env.sh python3 analysis/local_bin_covariance.py
          --flatsky-masks results/fiducial/flatsky_masks
          --healpix-masks results/fiducial/healpix_masks
          --theory-spectra data/theory_spectra.npz --lmax l6000
          --stride 8 --nchi 128 --ells 2025 2075 ... --widths 50 100 --out <dir>
Out:  <out>/bin_covariance_local_prediction.json
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


def local_power(proj, mask, window, stride, nchi, sampling, lmax):
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
        p *= qnorm <= lmax
        if spec != 'tt':
            p *= leakage if spec == 'bb' else 1-leakage
        p *= W.interpolation_power(q[..., 0], q[..., 1], dec[:, None], sampling.nside)
        p *= W.grid_power(ell*u[:, 0], ell*u[:, 1], sampling.res_arcmin, sampling.rebin)[None, :]
        # geometry.jacobian uses eastward x; native pixel indices run west.
        # Reflect both spatial arrays before Fourier transforming their kernel.
        return p.reshape(shape)[:, ::-1, :]
    return sample[:, ::-1], angles, evaluate


def main():
    import spectra as S
    import theory as T
    import pipeline_windows as W
    from _decisions import parse_lmax
    from compare_bin_covariance import PROJS, SPECS
    from namaster_spectra import load_mask
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(p, 'flatsky_masks', 'healpix_masks', 'theory_spectra')
    p.add_argument('--lmax', type=parse_lmax, required=True)
    p.add_argument('--stride', type=int, required=True)
    p.add_argument('--nchi', type=int, required=True)
    p.add_argument('--ells', type=float, nargs='+', required=True)
    p.add_argument('--widths', type=int, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    th = T.load(args.theory_spectra)
    sampling = W.Sampling.read(args.healpix_masks, args.flatsky_masks)
    pixel = np.deg2rad(sampling.res_arcmin/60)*args.stride
    rows = []
    for proj in PROJS:
        mask, window, _, _ = load_mask(args.flatsky_masks, proj, S.FIELD)
        w, angles, evaluate = local_power(proj, mask, window, args.stride, args.nchi,
                                          sampling, args.lmax)
        ones = np.ones(w.shape+(args.nchi,))
        controls = {width: overlap_kernel(ones, w, pixel, angles, width,
                                          width*np.array([0, 1, 2, 4]))
                    for width in args.widths}
        for spec in SPECS:
            if proj == 'ProjSTG' and spec == 'bb':
                continue
            for ell in args.ells:
                power = evaluate(ell, th, spec)
                mean = np.sum(w**2*power.mean(-1))/np.sum(w**2)
                asymptote = np.sum(w**4*(power**2).mean(-1))/np.sum(w**4)/mean**2
                for width in args.widths:
                    v = overlap_kernel(power, w, pixel, angles, width,
                                       width*np.array([0, 1, 2, 4]))
                    c = controls[width]
                    row = dict(projection=proj, spectrum=spec, ell=ell, width=width,
                               fractional_variance_ratio=float(v[0]/c[0]/mean**2),
                               wide_bin_ratio=float(asymptote),
                               projected_rho=(v[1:]/v[0]).tolist(),
                               control_rho=(c[1:]/c[0]).tolist())
                    rows.append(row)
                    print(row, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'bin_covariance_local_prediction.json').write_text(json.dumps(
        dict(approximation=__doc__, stride=args.stride, nchi=args.nchi, rows=rows),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
