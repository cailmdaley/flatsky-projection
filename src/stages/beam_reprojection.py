"""Compare a Gaussian on the SFL grid with the same Gaussian on the sphere.

The input is an existing lc Q/U map and its E-only harmonic realization.
Filtered and unfiltered paths are paired so their transfer ratios divide out
their own sampling windows. ZEA is the response target; STG is a conformal
control for B modes. Differential B power is also compared with the mismatch
between the unfiltered paths, which measures their sampling floor.
"""

import argparse
from pathlib import Path
import sys
import time

from _decisions import set_threads


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def mask_slices(metadata):
    """Production masks store their crop in the original full target map."""
    if 'lo' in metadata:
        lo, hi = int(metadata['lo']), int(metadata['hi'])
        return slice(lo, hi), slice(lo, hi)
    return (slice(int(metadata['ylo']), int(metadata['yhi'])),
            slice(int(metadata['xlo']), int(metadata['xhi'])))


def run(args):
    set_threads(args.max_procs)
    import gc
    import healpy as hp
    import numpy as np
    import pymaster as nmt
    from scipy.fft import rfft2, irfft2
    from spt3g import core, maps

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = next(core.G3File(args.qu))
    q, u = frame['Q'], frame['U']
    ny, nx = q.shape
    res = q.res
    sigma = np.deg2rad(args.fwhm/60) / np.sqrt(8*np.log(2))
    ky = np.fft.fftfreq(ny, res)[:, None] * 2*np.pi
    kx = np.fft.rfftfreq(nx, res)[None, :] * 2*np.pi
    beam = np.exp(-.5*sigma**2*(kx*kx+ky*ky))

    def target(proj):
        return maps.FlatSkyMap(nx, ny, res, proj=getattr(maps.MapProjection, proj),
                              alpha_center=q.alpha_center, delta_center=q.delta_center,
                              pol_conv=q.pol_conv, weighted=False)

    projections = ('ProjZEA', 'ProjSTG')

    def save_projected(qsrc, usrc, tag):
        for proj in projections:
            pair = []
            for src in (qsrc, usrc):
                dst = target(proj)
                maps.reproj_map(src, dst, interp=True, rebin=args.rebin)
                pair.append(dst)
            maps.flatten_pol(*pair)
            np.save(out / f'{tag}_{proj}.npy', np.stack([np.asarray(m) for m in pair]))
            log(f'{tag} -> {proj}')

    # The SFL Gaussian acts on flattened Q/U, as it does in a flat analysis.
    for filtered, tag in ((False, 'sfl_base'), (True, 'sfl_beam')):
        qsrc, usrc = q.clone(), u.clone()
        if filtered:
            for src in (qsrc, usrc):
                np.asarray(src)[:] = irfft2(rfft2(np.asarray(src))*beam, s=(ny,nx))
        maps.flatten_pol(qsrc, usrc, invert=True)
        save_projected(qsrc, usrc, tag)
        del qsrc, usrc
        gc.collect()
    del frame, u, beam
    gc.collect()

    # Each reference pair has identical synthesis and pixel integration.
    e, b = hp.read_alm(args.alms, hdu=(2,3))
    lmax = hp.Alm.getlmax(e.size)
    for filtered, tag in ((False, 'sphere_base'), (True, 'sphere_beam')):
        ep = hp.almxfl(e, hp.gauss_beam(np.deg2rad(args.fwhm/60), lmax=lmax,
                                       pol=True)[:,1]) if filtered else e
        qm, um = hp.alm2map_spin([ep,b], args.nside, 2, lmax=lmax)
        qs = maps.HealpixSkyMap(qm, pol_conv=maps.MapPolConv.COSMO)
        del qm
        us = maps.HealpixSkyMap(um, pol_conv=maps.MapPolConv.COSMO)
        del um
        save_projected(qs, us, tag)
        del qs, us, ep
        gc.collect()
    del e, b

    edges = np.arange(500., 4100., 100.)
    sectors = np.linspace(0, np.pi, 13)

    result = dict(ell=(edges[1:]+edges[:-1])/2, chi=(sectors[1:]+sectors[:-1])/2,
                  fwhm=args.fwhm, nside=args.nside, rebin=args.rebin)
    binner = nmt.NmtBinFlat(edges[:-1],edges[1:])
    for proj in projections:
        pairs = {tag: np.load(out/f'{tag}_{proj}.npy') for tag in
                 ('sfl_base','sfl_beam','sphere_base','sphere_beam')}
        for footprint, directory, mi in (('disk5',args.disk_masks,5),
                                          ('field1500d',args.field_masks,0)):
            metadata = np.load(Path(directory)/f'mask_{proj}_{mi}.npz')
            mask = metadata['mask']
            ys, xs = mask_slices(metadata)
            cropped = {tag: pair[:, ys, xs] for tag, pair in pairs.items()}
            my, mx = mask.shape
            lx, ly = float(metadata['lx_rad']), float(metadata['ly_rad'])
            ky = np.fft.fftfreq(my, res)[:, None]*2*np.pi
            kx = np.fft.rfftfreq(mx, res)[None, :]*2*np.pi
            ell = np.hypot(kx, ky)
            # Physical east points opposite the array x axis.
            chi = np.mod(np.arctan2(ky, -kx), np.pi)
            ib = np.digitize(ell.ravel(), edges)-1
            ic = np.digitize(chi.ravel(), sectors)-1
            valid = (ib>=0)&(ib<len(edges)-1)&(ic>=0)&(ic<12)
            bins2d = ib[valid]*12+ic[valid]
            column_weight = np.full(mx//2+1, 2.)
            column_weight[0] = 1.
            if mx % 2 == 0:
                column_weight[-1] = 1.
            multiplicity = np.broadcast_to(column_weight, ell.shape)

            def bin_power(power):
                return np.bincount(bins2d, weights=(power*multiplicity).ravel()[valid],
                                   minlength=(len(edges)-1)*12).reshape(-1,12)

            powers = {}
            for tag, pair in cropped.items():
                power = sum(np.abs(rfft2(m*mask))**2 for m in pair)
                powers[tag] = bin_power(power)
            for origin in ('sfl','sphere'):
                base, filt = powers[f'{origin}_base'], powers[f'{origin}_beam']
                result[f'{footprint}_{proj}_{origin}_response2d'] = filt/base
                result[f'{footprint}_{proj}_{origin}_response'] = filt.sum(1)/base.sum(1)

            # NaMaster's purified pseudo-B excludes ambiguous boundary modes.
            # Ratios share the same mask and need no pixel-window correction.
            baseline = cropped['sphere_base']
            mismatch = cropped['sfl_base']-baseline
            differential = ((cropped['sfl_beam']-cropped['sfl_base'])
                            -(cropped['sphere_beam']-baseline))
            for tag, pair in (('baseline',baseline), ('mismatch',mismatch),
                              ('differential',differential)):
                fld = nmt.NmtFieldFlat(lx,ly,mask,list(pair),purify_b=True)
                cl = nmt.compute_coupled_cell_flat(fld,fld,binner)
                result[f'{footprint}_{proj}_{tag}_ee'] = cl[0]
                result[f'{footprint}_{proj}_{tag}_bb'] = cl[3]
                del fld
            log(f'spectra {footprint} {proj}')
        del pairs
        gc.collect()
    np.savez(out/'summary.npz', **result)
    log('wrote summary.npz')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qu',required=True)
    p.add_argument('--alms',required=True)
    p.add_argument('--disk-masks',required=True)
    p.add_argument('--field-masks',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--fwhm',type=float,default=1.)
    p.add_argument('--nside',type=int,default=4096)
    p.add_argument('--rebin',type=int,default=4)
    p.add_argument('--max-procs',type=int,default=2)
    run(p.parse_args())
