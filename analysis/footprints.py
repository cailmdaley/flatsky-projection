"""Coarse HEALPix masks of the analysis footprints, and the shear each one sees.

The footprints' shapes and the patch centre are those of
`src/stages/_footprints.py`; the apodization is passed in, read from the
`healpix_masks` index files. Nothing here touches map pixels or spectra -- only
the *distribution of shear over the weighted footprint* is needed, and that
field is smooth on degree scales while the apodization kernel is degrees wide,
so a coarse Nside (1024, 3.4 arcmin) is already deep into convergence.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "stages"))
from _footprints import DEC_DEG, RA_DEG  # noqa: E402


def _hp():
    import healpy
    return healpy


def apodize(mask, apotype, aposize_deg):
    """NaMaster's mask apodization."""
    import pymaster as nmt
    return nmt.mask_apodization(mask, apotype=apotype, aposize=aposize_deg)


def disk_mask(radius_rad, nside, apotype, aposize_deg):
    """An apodized disk about the patch centre."""
    hp = _hp()
    vec = hp.ang2vec(RA_DEG, DEC_DEG, lonlat=True)
    mask = np.zeros(hp.nside2npix(nside))
    mask[hp.query_disc(nside, vec, radius_rad)] = 1.0
    return apodize(mask, apotype, aposize_deg)


def rect_mask(ra_half_deg, dec_min_deg, dec_max_deg, nside, apotype, aposize_deg):
    """An apodized RA/dec rectangle about the patch centre's RA -- a survey field.

    "Rectangle" means constant limits in the *coordinates*, which is how survey
    fields are actually defined and scanned; on the sphere it is wider in
    angular extent at its northern edge than its southern one.
    """
    hp = _hp()
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    ra = np.where(ra > 180.0, ra - 360.0, ra)
    inside = (np.abs(ra - RA_DEG) <= ra_half_deg) & \
             (dec >= dec_min_deg) & (dec <= dec_max_deg)
    mask = np.zeros(npix)
    mask[inside] = 1.0
    return apodize(mask, apotype, aposize_deg)


def f_sky(mask):
    """Effective sky fraction, <mask^2> -- the pipeline's own definition."""
    return float(np.mean(np.asarray(mask) ** 2))


def mask_geometry(mask, proj, nside, threshold=0.0, **jac_kw):
    """(s1, s2, weights) over the pixels a mask actually keeps.

    Weights are mask^2: the pseudo-Cl estimator applies the apodized mask to
    the map, so power is weighted by its square.
    """
    import geometry as G
    hp = _hp()
    mask = np.asarray(mask, dtype=float)
    keep = np.nonzero(mask > threshold)[0]
    ra, dec = hp.pix2ang(nside, keep, lonlat=True)
    ra = np.where(ra > 180.0, ra - 360.0, ra)
    s1, s2 = G.singular_values(G.jacobian(proj, ra, dec, **jac_kw))
    return s1, s2, mask[keep] ** 2


def weighted_quantiles(values, weights, q=(0.05, 0.5, 0.95)):
    """Mask-weighted quantiles -- the honest summary of a shear *distribution*.

    A footprint does not have "a" shear; it has a spread, and the high-l excess
    is an average over that spread. Quoting only the mean hides that 1500d's
    tail reaches values no disk samples.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= np.sum(w)
    return np.interp(np.asarray(q), cdf, v)
