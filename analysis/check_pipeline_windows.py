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


def run(n=128, nside=8192):
    dec0 = np.deg2rad(-56.)
    rng = np.random.default_rng(123)
    # Sample many subpixel phases at the field centre. Populate only the
    # interpolation stencil, so this test never allocates a full Nside8192 map.
    ra = rng.uniform(-.004, .004, n*n)
    dec = dec0 + rng.uniform(-.004, .004, n*n)
    pixels, _ = hp.get_interp_weights(nside, np.pi/2-dec, ra)
    ids = np.unique(pixels)
    th, ph = hp.pix2ang(nside, ids)
    east = ((ph + np.pi) % (2*np.pi) - np.pi) * np.cos(dec0)
    north = np.pi / 2 - th - dec0
    a = core.DoubleVector(ra)
    d = core.DoubleVector(dec)
    rows = []
    for ell in (1000., 3000., 4800., 6000.):
        for angle in (0., np.pi / 4, np.pi / 2):
            qe, qn = ell * np.cos(angle), ell * np.sin(angle)
            phase = qe * east + qn * north
            real = maps.HealpixSkyMap(ids, np.cos(phase), nside, False, False)
            imag = maps.HealpixSkyMap(ids, np.sin(phase), nside, False, False)
            value = np.asarray(real.get_interp_values(a, d)) + 1j*np.asarray(imag.get_interp_values(a, d))
            amp = value * np.exp(-1j*(qe*ra*np.cos(dec0) + qn*(dec-dec0)))
            measured = abs(amp.mean())**2
            predicted = float(interpolation_power(qe, qn, dec0, nside))
            rows.append(dict(ell=ell, angle=float(angle), actual=measured,
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
    with open(args.out,'w') as f:
        json.dump(run(),f,indent=2)
