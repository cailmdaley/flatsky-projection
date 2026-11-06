"""Controlled monochromatic synthesis through spt3g's actual ring interpolator.

A plane wave in the local east/north frame has exactly known input power.
Compare its coherent projected amplitude with interpolation on/off, and the
exact 8x8 average with a continuous top-hat. No random CMB realization or fitted
transfer is used. Run under the SPT env-shell.
"""
import json
import numpy as np
from spt3g import core, maps
import healpy as hp
from pipeline_windows import interpolation_power, grid_power


def run(n=512, nside=8192, dec_deg=-56.):
    from scipy.stats import qmc
    dec0 = np.deg2rad(dec_deg)
    # Average complete longitude phases and one complete between-ring phase.
    # A tiny random patch leaves a boundary-dependent coherent bias, especially
    # on polar rings. Integer azimuthal modes are periodic on the full ring.
    stencil,_=hp.get_interp_weights(nside,np.pi/2-dec0,0.)
    ring_dec=np.pi/2-hp.pix2ang(nside,stencil)[0]
    zlo,zhi=np.sin(ring_dec).min(),np.sin(ring_dec).max()
    uv=qmc.Sobol(2,scramble=True,seed=123).random_base2(int(np.log2(n*n)))
    ra=2*np.pi*uv[:,0]
    dec=np.arcsin(zlo+(zhi-zlo)*uv[:,1])
    pixels, _ = hp.get_interp_weights(nside, np.pi/2-dec, ra)
    ids = np.unique(pixels)
    th, ph = hp.pix2ang(nside, ids)
    north = np.pi / 2 - th - dec0
    a = core.DoubleVector(ra)
    d = core.DoubleVector(dec)
    rows = []
    for ell in (1000., 3000., 4800., 6000.):
        for angle in (0., np.pi / 4, np.pi / 2):
            m=round(ell*np.cos(angle)*np.cos(dec0))
            qe, qn = m/np.cos(dec0), ell * np.sin(angle)
            phase = m * ph + qn * north
            real = maps.HealpixSkyMap(ids, np.cos(phase), nside, False, False)
            imag = maps.HealpixSkyMap(ids, np.sin(phase), nside, False, False)
            value = np.asarray(real.get_interp_values(a, d)) + 1j*np.asarray(imag.get_interp_values(a, d))
            amp = value * np.exp(-1j*(qe*ra*np.cos(dec0) + qn*(dec-dec0)))
            measured = abs(amp.mean())**2
            predicted = float(interpolation_power(qe, qn, dec0, nside))
            rows.append(dict(dec=dec_deg, ell=ell, angle=float(angle), qe=float(qe), qn=float(qn), actual=measured,
                             predicted=predicted, residual=measured/predicted-1))
            print(rows[-1], flush=True)
    # Rebin alone: apply the actual sample-centre offsets of an 8x8 pixel.
    d = np.deg2rad(2 / 60)
    offsets = ((np.arange(8)+.5)/8-.5)*d
    for ell in (3000.,4800.,6000.):
        kx, ky = ell*np.cos(.37), ell*np.sin(.37)
        direct = abs(np.exp(1j*kx*offsets).mean()*np.exp(1j*ky*offsets).mean())**2
        predicted = float(grid_power(kx,ky))
        assert abs(direct-predicted)<1e-12
    assert max(abs(r['residual']) for r in rows)<5e-4
    return rows


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='/sptlocal/user/cailmdaley/pol_project/pipeline_window_control.json')
    args=p.parse_args()
    rows=[r for dec in (-70., -56., -42., -35.) for r in run(dec_deg=dec)]
    with open(args.out,'w') as f:
        json.dump(rows,f,indent=2)
