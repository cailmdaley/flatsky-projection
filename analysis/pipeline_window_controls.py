"""Monochromatic skies through spt3g's own interpolator and projector.

Three controls of the closed-form windows of `pipeline_windows`, each with an
exactly known input and nothing fitted:

  ring       a plane wave in the local east/north frame, evaluated with the
             HEALPix ring interpolator, against `interpolation_power`; and
             the exact rebin-subsample average against `grid_power`.
  projector  a unit complex wave in known projected coordinates, through
             maps.reproj_map with interpolation and the production rebin
             (and the controls without either), against the interpolation
             window integrated over the true sample positions times
             `grid_power`.
  spin       Q/U of a polarized wave through reproj_map and FlattenPol,
             against the flattened spin angle `response` uses.

The synthesis Nside, flat pixel and rebin factor are the production sampling,
read from the masks' index files; the patch centre comes from `_footprints`.
The probe positions, multipoles and angles are choices of this control.

Run:  bash env.sh python3 analysis/pipeline_window_controls.py
          --healpix-masks results/fiducial/healpix_masks
          --flatsky-masks results/fiducial/flatsky_masks --out <dir>
Out:  <out>/pipeline_window_controls.json, {"ring": [...], "projector":
      [...], "spin": [...]}, one record per probe.
"""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from spt3g import core, maps

import geometry as G
import shear_leakage as SL
import spectra as S
from pipeline_windows import (Sampling, flattened_rotation, grid_power, interpolation_power,
                              spin_angle)


def sample_ids(out, n, sampling):
    """HEALPix pixels the interpolator touches for every subsample of an n x n map."""
    r = sampling.rebin
    yy, xx = np.mgrid[:n*r, :n*r]/r + .5/r - .5
    ra, dec = out.xy_to_angle(core.DoubleVector(xx.ravel()), core.DoubleVector(yy.ravel()))
    ids = np.unique(hp.get_interp_weights(sampling.nside, np.pi/2-np.asarray(dec), np.asarray(ra))[0])
    th, ph = hp.pix2ang(sampling.nside, ids)
    return ids, th, ph


def centred_map(proj, n, ra0, dec0, sampling):
    out = G._flat_map(proj, res_arcmin=sampling.res_arcmin, n=n)
    x, y = out.angle_to_xy(np.deg2rad(ra0), np.deg2rad(dec0))
    out.x_center += (n-1)/2-x
    out.y_center += (n-1)/2-y
    return out


def ring(sampling, n=512, dec_deg=-56.):
    """Plane waves through the ring interpolator, and the rebin average alone."""
    from scipy.stats import qmc
    nside = sampling.nside
    dec0 = np.deg2rad(dec_deg)
    # Average complete longitude phases and one complete between-ring phase.
    # A tiny random patch leaves a boundary-dependent coherent bias, especially
    # on polar rings. Integer azimuthal modes are periodic on the full ring.
    stencil, _ = hp.get_interp_weights(nside, np.pi/2-dec0, 0.)
    ring_dec = np.pi/2-hp.pix2ang(nside, stencil)[0]
    zlo, zhi = np.sin(ring_dec).min(), np.sin(ring_dec).max()
    uv = qmc.Sobol(2, scramble=True, seed=123).random_base2(int(np.log2(n*n)))
    ra = 2*np.pi*uv[:, 0]
    dec = np.arcsin(zlo+(zhi-zlo)*uv[:, 1])
    pixels, _ = hp.get_interp_weights(nside, np.pi/2-dec, ra)
    ids = np.unique(pixels)
    th, ph = hp.pix2ang(nside, ids)
    north = np.pi/2-th-dec0
    a = core.DoubleVector(ra)
    d = core.DoubleVector(dec)
    rows = []
    for ell in (1000., 3000., 4800., 6000.):
        for angle in (0., np.pi/4, np.pi/2):
            m = round(ell*np.cos(angle)*np.cos(dec0))
            qe, qn = m/np.cos(dec0), ell*np.sin(angle)
            phase = m*ph+qn*north
            real = maps.HealpixSkyMap(ids, np.cos(phase), nside, False, False)
            imag = maps.HealpixSkyMap(ids, np.sin(phase), nside, False, False)
            value = np.asarray(real.get_interp_values(a, d))+1j*np.asarray(imag.get_interp_values(a, d))
            amp = value*np.exp(-1j*(qe*ra*np.cos(dec0)+qn*(dec-dec0)))
            measured = abs(amp.mean())**2
            predicted = float(interpolation_power(qe, qn, dec0, nside))
            rows.append(dict(dec=dec_deg, ell=ell, angle=float(angle), qe=float(qe), qn=float(qn),
                             actual=measured, predicted=predicted, residual=measured/predicted-1))
            print(rows[-1], flush=True)
    # Rebin alone: apply the actual sample-centre offsets of one flat pixel.
    r = sampling.rebin
    d = np.deg2rad(sampling.res_arcmin/60)
    offsets = ((np.arange(r)+.5)/r-.5)*d
    for ell in (3000., 4800., 6000.):
        kx, ky = ell*np.cos(.37), ell*np.sin(.37)
        direct = abs(np.exp(1j*kx*offsets).mean()*np.exp(1j*ky*offsets).mean())**2
        predicted = float(grid_power(kx, ky, sampling.res_arcmin, r))
        assert abs(direct-predicted) < 1e-12
    assert max(abs(row['residual']) for row in rows) < 5e-4
    return rows


def projector(proj, sampling, ell=4800., angle=.37, n=32, ra0=35., dec0=-60.):
    """A unit complex wave through reproj_map, with and without interpolation and rebin."""
    res = np.deg2rad(sampling.res_arcmin/60)
    out = centred_map(proj, n, ra0, dec0, sampling)
    ids, th, ph = sample_ids(out, n, sampling)
    sx, sy = out.angle_to_xy(core.DoubleVector(ph), core.DoubleVector(np.pi/2-th))
    kx, ky = ell*np.cos(angle), ell*np.sin(angle)
    source_phase = -kx*np.asarray(sx)*res+ky*np.asarray(sy)*res
    src = [maps.HealpixSkyMap(ids, v, sampling.nside, False, False)
           for v in (np.cos(source_phase), np.sin(source_phase))]
    y, x = np.mgrid[:n, :n]
    phase = -kx*x*res+ky*y*res
    a, d = out.xy_to_angle(core.DoubleVector(x.ravel().astype(float)),
                           core.DoubleVector(y.ravel().astype(float)))
    jac = G.jacobian(proj, np.rad2deg(a), np.rad2deg(d))
    q = np.einsum('pji,j->pi', jac, np.array([kx, ky]))
    expected_interp = np.sqrt(interpolation_power(q[:, 0], q[:, 1], np.asarray(d),
                                                  sampling.nside)).mean()**2
    rows = []
    for interp, rebin in ((True, 1), (True, sampling.rebin), (False, sampling.rebin)):
        values = []
        for m in src:
            dest = out.clone(False)
            maps.reproj_map(m, dest, interp=interp, rebin=rebin)
            values.append(np.asarray(dest).copy())
        amp = ((values[0]+1j*values[1])*np.exp(-1j*phase)).mean()
        grid = float(grid_power(kx, ky, sampling.res_arcmin, rebin)) if rebin > 1 else 1.
        row = dict(proj=proj, ra=ra0, dec=dec0, ell=ell, interp=interp, rebin=rebin,
                   power=float(abs(amp)**2), prediction=expected_interp*grid)
        row['ratio'] = row['power']/row['prediction']
        print(row, flush=True)
        rows.append(row)
        if interp:
            assert abs(row['ratio']-1) < .001
    return rows


def spin(sampling, proj='ProjSFL', ra0=38., dec0=-62., angle=.37, ell=3000., n=24):
    """The signed COSMO spin angle through real Q/U reprojection and FlattenPol."""
    res = np.deg2rad(sampling.res_arcmin/60)
    out = centred_map(proj, n, ra0, dec0, sampling)
    out.pol_conv = maps.MapPolConv.COSMO
    ids, th, ph = sample_ids(out, n, sampling)
    sx, sy = out.angle_to_xy(core.DoubleVector(ph), core.DoubleVector(np.pi/2-th))
    k = ell*np.array([np.cos(angle), np.sin(angle)])
    phase = (-k[0]*np.asarray(sx)+k[1]*np.asarray(sy))*res
    jac = G.jacobian(proj, np.rad2deg(ph), 90-np.rad2deg(th))
    q = np.einsum('pji,j->pi', jac, k)
    aq = np.arctan2(q[:, 1], q[:, 0])
    yy, xx = np.mgrid[:n, :n]
    demod = np.exp(-1j*(-k[0]*xx+k[1]*yy)*res)
    amps = []
    for carrier in (np.cos(phase), np.sin(phase)):
        dests = []
        for stokes, pol in ((np.cos(2*aq), maps.MapPolType.Q),
                            (np.sin(2*aq), maps.MapPolType.U)):
            src = maps.HealpixSkyMap(ids, carrier*stokes, sampling.nside, False, False)
            src.pol_conv = maps.MapPolConv.COSMO
            src.pol_type = pol
            dst = out.clone(False)
            dst.pol_type = pol
            maps.reproj_map(src, dst, interp=True, rebin=sampling.rebin)
            dests.append(dst)
        maps.flatten_pol(*dests)
        amps.append([np.asarray(dst).copy() for dst in dests])
    qa = ((amps[0][0]+1j*amps[1][0])*demod).mean()
    ua = ((amps[0][1]+1j*amps[1][1])*demod).mean()
    # Q and U share the scalar complex wave amplitude.
    measured = .5*np.arctan2((ua/qa).real, 1.)
    jc = G.jacobian(proj, [ra0], [dec0])
    qc = jc[0].T@k
    delta = SL.flattening_residual(jc)[0]
    predicted = spin_angle(qc[None, None, :], flattened_rotation(jc), np.array([angle]))[0, 0]
    # Gate the exact relative-angle expression used by response().
    # Polarization axes are modulo pi/2 when inferred from U/Q.
    error = (measured-angle-predicted+np.pi/4) % (np.pi/2)-np.pi/4
    row = dict(proj=proj, ra=ra0, dec=dec0, ell=ell, angle=angle,
               spin_error_deg=float(np.rad2deg(error)), delta_deg=float(np.rad2deg(delta)))
    print(row, flush=True)
    assert abs(row['spin_error_deg']) < .01
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'healpix_masks', 'flatsky_masks')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sampling = Sampling.read(args.healpix_masks, args.flatsky_masks)

    rings = [r for dec in (-70., -56., -42., -35.) for r in ring(sampling, dec_deg=dec)]
    rows = [r for point in ((35., -60.), (-35., -45.)) for p in S.PROJS
            for r in projector(p, sampling, ra0=point[0], dec0=point[1])]
    # Reciprocal-grid images and the HEALPix equatorial-belt branch.
    rows += [r for p in S.PROJS for ell, angle in ((5800., 0.), (6500., 1.57), (7000., .8))
             for r in projector(p, sampling, ell=ell, angle=angle, n=64)]
    rows += projector('ProjZEA', sampling, ra0=20., dec0=-35., n=64)
    spins = [spin(sampling, ra0=ra, dec0=dec, angle=angle)
             for ra, dec in ((38., -62.), (45., -48.)) for angle in (.37, .8)]
    spins += [spin(sampling, 'ProjBICEP'), spin(sampling, ra0=-38., dec0=-62.),
              spin(sampling, 'ProjZEA', ra0=40., dec0=-70.), spin(sampling, 'ProjSTG')]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'pipeline_window_controls.json').write_text(
        json.dumps(dict(ring=rings, projector=rows, spin=spins), indent=2)+'\n')


if __name__ == '__main__':
    main()
