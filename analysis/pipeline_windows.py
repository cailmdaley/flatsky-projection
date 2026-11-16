"""Parameter-free windows introduced by sampling a sky onto the simulation grid.

HEALPix synthesis evaluates a field at pixel centres; it does not average over
HEALPix pixels. Reprojection linearly interpolates along longitude and between
rings (in z in spt3g). Locally its coherent amplitude is a product of two tent
transforms, sinc^2 in east and north. These windows act before the projection,
not at the output multipole. The rebin average is instead in the map frame.

How the simulation sampled the sky -- the synthesis Nside, the flat pixel and
its oversampling -- is a `Sampling`, read from the mask outputs' index files.
"""
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src' / 'stages'))


@dataclass(frozen=True)
class Sampling:
    """The synthesis HEALPix Nside, the flat pixel size and its rebin factor."""
    nside: int
    res_arcmin: float
    rebin: int

    @classmethod
    def read(cls, healpix_masks, flatsky_masks):
        """From the `healpix_masks` and `flatsky_masks` index files."""
        def one(directory, key):
            (value,) = {json.loads(p.read_text())[key]
                        for p in Path(directory).glob('index_*.json')}
            return value
        return cls(nside=one(healpix_masks, 'nside'),
                   res_arcmin=one(flatsky_masks, 'res_arcmin'),
                   rebin=one(flatsky_masks, 'rebin'))

    @property
    def nyquist(self):
        """The grid's Nyquist multipole along each axis."""
        return np.pi / np.deg2rad(self.res_arcmin / 60)


def ring_spacing(dec, nside):
    """Local east/north ring separations in radians, at declination in radians."""
    z = np.abs(np.sin(dec))
    polar = z > 2 / 3
    nr = nside * np.sqrt(3 * (1 - z))
    dphi = np.pi / (2 * np.where(polar, nr, nside))
    dz = np.where(polar, 2 * nr / (3 * nside**2), 2 / (3 * nside))
    return np.cos(dec) * dphi, dz / np.cos(dec)


def interpolation_power(qeast, qnorth, dec, nside):
    """Coherent interpolation power, locally uniform HEALPix rings."""
    dx, dy = ring_spacing(dec, nside)
    return (np.sinc(qeast * dx / (2 * np.pi)) *
            np.sinc(qnorth * dy / (2 * np.pi)))**4


def grid_power(kx, ky, res_arcmin, rebin):
    """Exact finite-subsample average power (not a continuous pixel top-hat)."""
    d = np.deg2rad(res_arcmin / 60)
    def axis(k):
        # sin(x)/(n sin(x/n)), continuously extended at the origin.
        return np.sinc(k * d / (2 * np.pi)) / np.sinc(k * d / (2 * np.pi * rebin))
    return (axis(kx) * axis(ky))**2


@lru_cache(maxsize=16)
def footprint_samples(healpix_masks, footprint, nside=64):
    """Sky samples and squared apodization weights of one footprint's mask.

    The mask is rebuilt analytically from its `healpix_masks` index file at
    Nside 1024 and degraded to `nside`; no geometry histogram.
    """
    import healpy as hp
    import make_masks as MM
    idx = json.loads((Path(healpix_masks) / f'index_{footprint}.json').read_text())
    (rec,) = idx['masks']
    sample_nside = 1024
    ids, values = MM.rect_mask_analytic(sample_nside, idx['ra_deg'], rec['ra_half_deg'],
                                        rec['dec_min_deg'], rec['dec_max_deg'],
                                        idx['apotype'], idx['aposize_deg'])
    m = np.zeros(hp.nside2npix(sample_nside))
    m[ids] = values
    w = hp.ud_grade(m**2, nside)
    ids = np.flatnonzero(w)
    ra, dec = hp.pix2ang(nside, ids, lonlat=True)
    return ra, dec, w[ids]


def flattened_rotation(jac, flattening):
    """Sky-to-eastward-map polarization angle after the `flattening` decision's rotation.

    The polar rotation of J, less the flattening's residual
    (`shear_leakage.flattening_residual`, zero under `polar`), which is in
    the unflipped pixel frame and so enters with a minus sign.
    """
    import shear_leakage as SL
    phi=np.arctan2(jac[:,1,0]-jac[:,0,1],jac[:,0,0]+jac[:,1,1])
    return phi-SL.flattening_residual(jac, flattening)


def spin_angle(q, rotation, out_angle):
    """Flattened polarization angle relative to the observed Fourier direction.

    q has shape (position, mode, 2); a folded image keeps the observed
    out_angle, not the direction of its unsampled source mode.
    """
    return np.arctan2(q[...,1],q[...,0])+rotation[:,None]-out_angle[None,:]


def response(proj, ell, ell_sky, cl, healpix_masks, footprint, sampling, band_limit,
             flattening, spec='TT', nside=64, nchi=64, aliases=False):
    """Unwindowed and windowed forward spectra using the full local Jacobian.

    If aliases=True, additionally sum the nearest reciprocal-grid images.
    E-only input is assumed for EE/BB; TT is scalar. The production
    azimuthal top-hat division is included only in the windowed result.
    """
    import geometry as G
    import spectra as S
    ra, dec, w = footprint_samples(Path(healpix_masks), footprint, nside)
    jac = G.jacobian(proj, ra, dec)
    det = np.linalg.det(jac)
    scale = w * det**2 / np.sum(w * det)
    chi = (np.arange(nchi) + .5) * 2 * np.pi / nchi
    u = np.stack([np.cos(chi), np.sin(chi)], -1)
    rotation = flattened_rotation(jac, flattening)
    def evaluate(k, out_angle):
        q = np.einsum('pji,cj->pci', jac, k)
        qnorm = np.linalg.norm(q, axis=-1)
        power = np.interp(qnorm, ell_sky, cl, left=0, right=0)
        power *= qnorm <= band_limit
        if spec != 'TT':
            angle = spin_angle(q, rotation, out_angle)
            spin = np.sin(2 * angle)**2
            power *= spin if spec == 'BB' else 1 - spin
        win = interpolation_power(q[..., 0], q[..., 1], np.deg2rad(dec)[:, None],
                                  sampling.nside)
        return power, win
    def grid(k):
        return grid_power(k[:, 0], k[:, 1], sampling.res_arcmin, sampling.rebin)
    base, filtered, folded = [], [], []
    reciprocal = 2 * np.pi / np.deg2rad(sampling.res_arcmin / 60)
    for l in np.atleast_1d(ell):
        k = l * u
        power, win = evaluate(k, chi)
        b = np.sum(scale * power.mean(-1))
        v = np.sum(scale * (power * win * grid(k)[None, :]).mean(-1))
        a = 0.
        if aliases:
            for nx, ny in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                ka = k + reciprocal * np.array([nx, ny])
                pa, wa = evaluate(ka, chi)
                a += np.sum(scale * (pa * wa * grid(ka)[None, :]).mean(-1))
        pw = S.flat_pixwin(np.array([l]), sampling.res_arcmin)[0]
        base.append(b)
        filtered.append(v / pw)
        folded.append(a / pw)
    return np.array(base), np.array(filtered), np.array(folded)


def effective_window(spec, footprints, leg, theory, healpix_masks, sampling, lmax, delta_l,
                     flattening):
    """Production-bin correction for the *known simulation input*, per projection.

    Returns (footprint, projection, band). This is R_window C_input / R
    C_input, not a universal transfer function for an unknown sky. Aliases are
    intentionally not deconvolved. STG's B is an estimator floor with no
    nonzero geometric BB model; its EE window is used as a plotting
    convention, not claimed to calibrate the floor. Above the axis Nyquist the
    full-circle model is invalid; correction=1 leaves those production bins
    unchanged rather than applying an invalid response.
    """
    import spectra as S
    import theory as T
    spec = spec.upper()
    th = T.load(theory)
    cl = np.where(th['ell'] >= 2, 1., 0.) if spec == 'WHITE' else th['unlensed_TT' if spec == 'TT' else 'unlensed_EE']
    nbins = lmax // delta_l
    ell = (np.arange(nbins) + .5) * delta_l
    fine = np.arange(5., lmax, 10.)
    old_fine = S.flat_pixwin(fine, sampling.res_arcmin)
    old_bin = S.flat_pixwin(ell, sampling.res_arcmin)
    out = np.ones((len(footprints), len(S.PROJS), nbins))
    for i, footprint in enumerate(footprints):
        for j, proj in enumerate(S.PROJS):
            spin = 'TT' if spec in ('TT', 'WHITE') or leg == S.EB_LEG else spec
            if spec == 'BB' and proj == 'ProjSTG' and leg == S.QU_LEG:
                spin = 'EE'
            b, w, _ = response(proj, fine, th['ell'], cl, healpix_masks, footprint,
                               sampling, lmax, flattening, spec=spin, nside=32, nchi=64)
            b = b.reshape(nbins, -1).mean(-1)
            # Production divides the bandpower by the top-hat at ell_eff,
            # rather than dividing the integrand before averaging the bin.
            w = (w * old_fine).reshape(nbins, -1).mean(-1) / old_bin
            out[i, j] = np.divide(w, b, out=np.ones_like(w), where=b != 0)
    out[..., ell >= sampling.nyquist] = 1.
    return out
