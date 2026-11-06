"""Parameter-free windows introduced by sampling a sky onto the simulation grid.

HEALPix synthesis evaluates a field at pixel centres; it does not average over
HEALPix pixels. Reprojection linearly interpolates along longitude and between
rings (in z in spt3g). Locally its coherent amplitude is a product of two tent
transforms, sinc^2 in east and north. These windows act before the projection,
not at the output multipole. The 8x8 average is instead in the map frame.
"""
from functools import lru_cache
import numpy as np


def ring_spacing(dec, nside=8192):
    """Local east/north ring separations in radians, at declination in radians."""
    z = np.abs(np.sin(dec))
    polar = z > 2 / 3
    nr = nside * np.sqrt(3 * (1 - z))
    dphi = np.pi / (2 * np.where(polar, nr, nside))
    dz = np.where(polar, 2 * nr / (3 * nside**2), 2 / (3 * nside))
    return np.cos(dec) * dphi, dz / np.cos(dec)


def interpolation_power(qeast, qnorth, dec, nside=8192):
    """Coherent interpolation power, locally uniform HEALPix rings."""
    dx, dy = ring_spacing(dec, nside)
    return (np.sinc(qeast * dx / (2 * np.pi)) *
            np.sinc(qnorth * dy / (2 * np.pi)))**4


def grid_power(kx, ky, res_arcmin=2., rebin=8):
    """Exact finite-subsample average power (not a continuous pixel top-hat)."""
    d = np.deg2rad(res_arcmin / 60)
    def axis(k):
        # sin(x)/(n sin(x/n)), continuously extended at the origin.
        return np.sinc(k * d / (2 * np.pi)) / np.sinc(k * d / (2 * np.pi * rebin))
    return (axis(kx) * axis(ky))**2


@lru_cache(maxsize=16)
def footprint_samples(field=True, mask=5, nside=64):
    """Sky samples and squared apodization weights; no geometry histogram."""
    import healpy as hp
    import sys
    import json
    import spectra as S
    sys.path.insert(0, str(S.ROOT / 'src' / 'stages'))
    import make_masks as MM
    directory = 'field_healpix_mask' if field else 'healpix_masks'
    idx = json.loads((S.ROOT / 'results' / 'highell' / directory / 'index.json').read_text())
    rec = idx['masks'][0 if field else mask]
    sample_nside = 1024
    if field:
        ids, values = MM.rect_mask_analytic(sample_nside, idx['ra_deg'], rec['ra_half_deg'],
                                          rec['dec_min_deg'], rec['dec_max_deg'],
                                          idx['apotype'], idx['aposize_deg'])
        m = np.zeros(hp.nside2npix(sample_nside))
        m[ids] = values
    else:
        center = hp.ang2vec(idx['ra_deg'], idx['dec_deg'], lonlat=True)
        m = MM.apodize_disk_analytic(sample_nside, center, rec['radius_rad'],
                                    idx['apotype'], idx['aposize_deg'])
    w = hp.ud_grade(m**2, nside)
    ids = np.flatnonzero(w)
    ra, dec = hp.pix2ang(nside, ids, lonlat=True)
    return ra, dec, w[ids]


def response(proj, ell, ell_sky, cl, spec='TT', field=True, mask=5,
             nside=64, nchi=64, band_limit=6000, aliases=False):
    """Unwindowed and windowed forward spectra using the full local Jacobian.

    If aliases=True, additionally sum the nearest reciprocal-grid images.
    E-only input is assumed for EE/BB; TT is scalar. The old production
    azimuthal top-hat division is included only in the windowed result.
    """
    import geometry as G
    import spectra as S
    import shear_leakage as SL
    ra, dec, w = footprint_samples(field, mask, nside)
    jac = G.jacobian(proj, ra, dec)
    det = np.linalg.det(jac)
    scale = w * det**2 / np.sum(w * det)
    chi = (np.arange(nchi) + .5) * 2 * np.pi / nchi
    u = np.stack([np.cos(chi), np.sin(chi)], -1)
    # Plane principal axes differ from the sky axes by the polar rotation.
    polar_angle = np.arctan2(jac[:, 1, 0] - jac[:, 0, 1], jac[:, 0, 0] + jac[:, 1, 1])
    residual = SL.flattening_residual(jac)
    def evaluate(k, out_angle):
        q = np.einsum('pji,cj->pci', jac, k)
        qnorm = np.linalg.norm(q, axis=-1)
        power = np.interp(qnorm, ell_sky, cl, left=0, right=0)
        power *= qnorm <= band_limit
        if spec != 'TT':
            # COSMO Stokes convention and eastward plane x: FlattenPol's
            # residual enters with the opposite sign to flattening_residual.
            angle = np.arctan2(q[..., 1], q[..., 0]) + polar_angle[:, None] - out_angle[None, :] - residual[:, None]
            spin = np.sin(2 * angle)**2
            power *= spin if spec == 'BB' else 1 - spin
        win = interpolation_power(q[..., 0], q[..., 1], np.deg2rad(dec)[:, None])
        return power, win
    base, filtered, folded = [], [], []
    reciprocal = 2 * np.pi / np.deg2rad(2 / 60)
    for l in np.atleast_1d(ell):
        k = l * u
        power, win = evaluate(k, chi)
        b = np.sum(scale * power.mean(-1))
        v = np.sum(scale * (power * win * grid_power(k[:, 0], k[:, 1])[None, :]).mean(-1))
        a = 0.
        if aliases:
            for nx, ny in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                ka = k + reciprocal * np.array([nx, ny])
                pa, wa = evaluate(ka, chi)
                a += np.sum(scale * (pa * wa * grid_power(ka[:, 0], ka[:, 1])[None, :]).mean(-1))
        pw = S.flat_pixwin(np.array([l]))[0]
        base.append(b)
        filtered.append(v / pw)
        folded.append(a / pw)
    return np.array(base), np.array(filtered), np.array(folded)


@lru_cache(maxsize=16)
def effective_window(spec='EE', field=False, leg=0):
    """Production-bin correction for the *known simulation input*, per projection.

    Returns (projection, band) on the field, (mask, projection, band) on disks.
    This is R_window C_input / R C_input, not a universal transfer function for
    an unknown sky. Aliases are intentionally not deconvolved. STG's B is an
    estimator floor with no nonzero geometric BB model; its EE window is used
    as a plotting convention, not claimed to calibrate the floor.
    """
    import hashlib
    import os
    from pathlib import Path
    import spectra as S
    import theory as T
    spec = spec.upper()
    sources = [Path(__file__), S.ROOT / 'analysis' / 'geometry.py',
               S.ROOT / 'analysis' / 'shear_leakage.py',
               S.ROOT / 'analysis' / 'spectra.py',
               S.ROOT / 'src' / 'stages' / 'make_masks.py']
    payload = b''.join(path.read_bytes() for path in sources) + Path(S.THEORY_NPZ).read_bytes()
    for directory in ('healpix_masks', 'field_healpix_mask'):
        payload += (S.ROOT / 'results' / 'highell' / directory / 'index.json').read_bytes()
    key = hashlib.sha256(payload).hexdigest()[:16]
    directory = Path('/sptlocal/user/cailmdaley/pol_project/pipeline_window_cache')
    cache = directory / f'{key}_{spec}_{field}_{leg}.npz'
    if cache.exists():
        with np.load(cache) as z:
            return z['window']
    th = T.load()
    cl = np.where(th['ell'] >= 2, 1., 0.) if spec == 'WHITE' else th['unlensed_TT' if spec == 'TT' else 'unlensed_EE']
    fine = np.arange(5., S.LMAX, 10.)
    old_fine = S.flat_pixwin(fine)
    old_bin = S.flat_pixwin(S.ELL)
    out = np.ones((1 if field else len(S.PERC), len(S.PROJS), S.NBINS))
    for mask in range(out.shape[0]):
        for j, proj in enumerate(S.PROJS):
            spin = 'TT' if spec in ('TT', 'WHITE') or leg == S.EB_LEG else spec
            if spec == 'BB' and proj == 'ProjSTG' and leg == S.QU_LEG:
                spin = 'EE'
            b, w, _ = response(proj, fine, th['ell'], cl, spec=spin,
                               field=field, mask=mask, nside=32, nchi=64)
            b = b.reshape(S.NBINS, -1).mean(-1)
            # Production divides the bandpower by the top-hat at ell_eff,
            # rather than dividing the integrand before averaging the bin.
            w = (w * old_fine).reshape(S.NBINS, -1).mean(-1) / old_bin
            out[mask, j] = np.divide(w, b, out=np.ones_like(w), where=b != 0)
    if field:
        out = out[0]
    directory.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez(tmp, window=out, projs=S.PROJS, ell=S.ELL)
    tmp.replace(cache)
    return out
