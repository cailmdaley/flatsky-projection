"""Zero-B spherical transforms: point samples versus finite pixel averages.

No remapping or flat Fourier E/B split is used. CAR uses exact Fejer-1
quadrature through ducc0. HEALPix inversion is iteratively refined. The
footprint statistic selects *globally reconstructed scalar E and B*, and
therefore excludes cut-sky mask leakage; it is not a cut-sky pseudo-Cl test.
Pixel averages are explicit high-resolution equal-area quadrature (HEALPix)
or sin(theta)-weighted quadrature (CAR), with and without parallel transport.
"""
import argparse
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

from pathlib import Path
import json
import healpy as hp
import numpy as np
from ducc0 import sht


def synth(alm, ntheta, spin=2):
    return sht.synthesis_2d(alm=np.asarray(alm), spin=spin,
        lmax=hp.Alm.getlmax(alm.shape[-1]), geometry='F1',
        ntheta=ntheta, nphi=2*ntheta, nthreads=2)


def analyse(m, lmax):
    return sht.analysis_2d(map=m, spin=2, lmax=lmax, geometry='F1', nthreads=2)


def transport(theta, phi, theta0, phi0):
    """exp(2i alpha), transporting the child theta basis to its parent.

    Rodrigues' minimal rotation maps the child radial vector to the parent.
    For tangent v, Rv = v - (v dot parent)*(child+parent)/(1+child dot parent).
    """
    def basis(t, p):
        t, p = np.broadcast_arrays(t, p)
        r = np.stack([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)], -1)
        et = np.stack([np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)], -1)
        ep = np.stack([-np.sin(p), np.cos(p), np.zeros_like(p)], -1)
        return r, et, ep
    r, et, _ = basis(theta, phi)
    r0, et0, ep0 = basis(theta0, phi0)
    v = et - np.sum(et*r0, -1)[..., None]*(r+r0)/(1+np.sum(r*r0, -1))[..., None]
    z = np.sum(v*et0, -1) + 1j*np.sum(v*ep0, -1)
    return z*z


def hp_inverse(m, lmax, iterations=8):
    a = np.asarray(hp.map2alm_spin(m, spin=2, lmax=lmax))
    for _ in range(iterations):
        residual = m - hp.alm2map_spin(a, hp.npix2nside(m.shape[-1]), 2, lmax)
        a += hp.map2alm_spin(residual, 2, lmax=lmax)
    return a


def hp_average(a, nside, factor):
    hi = hp.alm2map_spin(a, nside*factor, 2, hp.Alm.getlmax(a.shape[-1]))
    hi = np.asarray([hp.reorder(m, r2n=True) for m in hi])
    out = np.empty((2, hp.nside2npix(nside)))
    raw = np.empty_like(out)
    count = factor**2
    for start in range(0, out.shape[1], 8192):
        ids = np.arange(start, min(start+8192, out.shape[1]))
        t0, p0 = hp.pix2ang(nside, ids, nest=True)
        children = ids[:, None]*count + np.arange(count)
        t, p = hp.pix2ang(nside*factor, children, nest=True)
        z = hi[0, children]+1j*hi[1, children]
        zr = z.mean(axis=1)
        zp = (z*transport(t, p, t0[:, None], p0[:, None])).mean(axis=1)
        raw[:, ids] = [zr.real, zr.imag]
        out[:, ids] = [zp.real, zp.imag]
    return np.asarray([hp.reorder(m, n2r=True) for m in out]), np.asarray([hp.reorder(m, n2r=True) for m in raw])


def car_average(a, ntheta, factor):
    raw, out = np.empty((2, ntheta, 2*ntheta)), np.empty((2, ntheta, 2*ntheta))
    # synthesis_2d's first longitude is 0, so shift all high-resolution
    # longitudes to the subpixel centres of coarse cells centred at phi=0.
    # A harmonic phase does this without interpolating the map.
    shifted = a.copy()
    _, mm = hp.Alm.getlm(hp.Alm.getlmax(a.shape[-1]))
    shift = -np.pi/(2*ntheta) + np.pi/(2*ntheta*factor)
    shifted *= np.exp(1j*mm*shift)
    high = synth(shifted, ntheta*factor)
    p = np.arange(2*ntheta*factor)*np.pi/(ntheta*factor)+shift
    p0 = np.arange(2*ntheta)*np.pi/ntheta
    for row in range(ntheta):
        t = (row*factor+np.arange(factor)+.5)*np.pi/(ntheta*factor)
        t0 = (row+.5)*np.pi/ntheta
        z = high[0, row*factor:(row+1)*factor]+1j*high[1, row*factor:(row+1)*factor]
        weights = np.sin(t)[:, None]
        zr = (z*weights).reshape(factor, 2*ntheta, factor).sum((0, 2))/(weights.sum()*factor)
        phase = transport(t[:, None], p[None, :], t0, np.repeat(p0, factor)[None, :])
        zp = (z*phase*weights).reshape(factor, 2*ntheta, factor).sum((0, 2))/(weights.sum()*factor)
        raw[:, row] = [zr.real, zr.imag]
        out[:, row] = [zp.real, zp.imag]
    return out, raw


def regions(ntheta):
    theta = (np.arange(ntheta)+.5)*np.pi/ntheta
    dec = 90-np.degrees(theta)
    ra = np.degrees(np.arange(2*ntheta)*np.pi/ntheta)
    dra = (ra+180)%360-180
    # Same C2 2.5-degree boundary taper as production's field mask.
    d = np.minimum(np.minimum(dec[:, None]+70, -42-dec[:, None]),
        np.degrees(np.arcsin(np.cos(np.radians(dec[:, None]))*
                             np.sin(np.radians(50-np.abs(dra)[None, :])))))
    x = np.clip(d/2.5, 0, 1)
    mask = (x-np.sin(2*np.pi*x)/(2*np.pi))**2
    result = {'field1500': mask, 'fullsky': np.ones_like(mask)}
    for centre in (0, -30, -56, -75):
        result[f'dec{centre}'] = np.broadcast_to((abs(dec[:, None]-centre)<3).astype(float), mask.shape)
    return {k: v*np.sin(theta[:, None]) for k, v in result.items()}


def statistics(a, truth, lmax):
    edges = [x for x in (32, 64, 128, 192) if x <= lmax] + [lmax+1]
    ell, _ = hp.Alm.getlm(lmax)
    reg = regions(384)
    values = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (ell>=lo)&(ell<hi)
        e0 = synth((truth[0]*band)[None], 384, spin=0)[0]
        e = synth((a[0]*band)[None], 384, spin=0)[0]
        b = synth((a[1]*band)[None], 384, spin=0)[0]
        for name, w in reg.items():
            norm = np.sum(w*e0**2)
            values.append(dict(region=name, lo=lo, hi=hi,
                amplitude=np.sum(w*e*e0)/norm,
                ee=np.sum(w*e**2)/norm, bb=np.sum(w*b**2)/norm,
                residual_ee=np.sum(w*(e-e0)**2)/norm))
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nside', type=int, default=128)
    p.add_argument('--factor', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lmax', type=int, default=256)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    np.random.seed(args.seed)
    cl = np.ones(args.lmax+1)
    cl[:2] = 0
    e = hp.synalm(cl, lmax=args.lmax, new=True)
    truth = np.array([e, e*0])
    records = {}
    ntheta = 3*args.nside  # CAR spacing pi/(3 Nside), close to HEALPix sqrt(area).
    point = np.asarray(hp.alm2map_spin(truth, args.nside, 2, args.lmax))
    records['healpix_point'] = hp_inverse(point, args.lmax)
    del point
    records['car_point'] = analyse(synth(truth, ntheta), args.lmax)
    averaged, raw = hp_average(truth, args.nside, args.factor)
    records['healpix_average'] = hp_inverse(averaged, args.lmax)
    records['healpix_coordinate_average'] = hp_inverse(raw, args.lmax)
    del averaged, raw
    averaged, raw = car_average(truth, ntheta, args.factor)
    records['car_average'] = analyse(averaged, args.lmax)
    records['car_coordinate_average'] = analyse(raw, args.lmax)
    del averaged, raw
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/'alms.npz', truth=truth, **records)
    result = dict(settings=vars(args), statistics={})
    for name, a in records.items():
        print('Statistics', name, flush=True)
        result['statistics'][name] = statistics(a, truth, args.lmax)
    (output/'statistics.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
