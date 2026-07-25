"""Sky masks, and the projection shear each one actually sees.

The 2021 suite used seven concentric apodized disks about the projection
centre. That choice makes sky area and projection shear degenerate -- a bigger
disk is also a more sheared disk -- so the measured trend "excess grows with
f_sky" cannot distinguish the two. This module rebuilds those disks and puts
SPT's real 1500d footprint beside them, which breaks the degeneracy: 1500d is
wide in RA and narrow in declination, so it carries far more shear than a disk
of the same area.

Masks are built at a coarser Nside than the 2021 pipeline's 4096. Nothing here
touches map pixels or spectra -- we only need the *distribution of shear over
the weighted footprint*, and that field is smooth on degree scales while the
apodization kernel is 2.5 deg wide. Nside 1024 (3.4 arcmin) is already deep
into convergence and keeps this a laptop-scale calculation.
"""

import numpy as np

# The 2021 configuration, verbatim from
# /sptlocal/analysis/pol_project2020/config.yaml
DISK_RADII_RAD = [0.175, 0.2, 0.225, 0.25, 0.35, 0.45, 0.65]
DISK_PERC = [0.75, 1.00, 1.25, 1.5, 3.0, 5.0, 10.0]
APOSIZE_DEG = 2.5
APOTYPE = "C2"

# Projection / field centre. The 2021 sims and SPT's own 1500d maps share it.
ALPHA_CENTER_DEG = 0.0
DELTA_CENTER_DEG = -56.0

NSIDE = 1024


def _hp():
    import healpy
    return healpy


def apodize(mask, aposize_deg=APOSIZE_DEG, apotype=APOTYPE):
    """NaMaster's mask apodization, matching the 2021 pipeline exactly."""
    import pymaster as nmt
    return nmt.mask_apodization(mask, apotype=apotype, aposize=aposize_deg)


def disk_mask(radius_rad, nside=NSIDE, apod=True):
    """One of the 2021 concentric disks, apodized as the pipeline did."""
    hp = _hp()
    vec = hp.ang2vec(ALPHA_CENTER_DEG, DELTA_CENTER_DEG, lonlat=True)
    mask = np.zeros(hp.nside2npix(nside))
    mask[hp.query_disc(nside, vec, radius_rad)] = 1.0
    return apodize(mask) if apod else mask


def rect_mask(ra_half_deg, dec_min_deg, dec_max_deg, nside=NSIDE, apod=True):
    """A RA/dec rectangle -- the shape of a real SPT survey field.

    "Rectangle" means constant limits in the *coordinates*, which is how survey
    fields are actually defined and scanned; on the sphere it is wider in
    angular extent at its northern edge than its southern one.
    """
    hp = _hp()
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    ra = np.where(ra > 180.0, ra - 360.0, ra)
    inside = (np.abs(ra - ALPHA_CENTER_DEG) <= ra_half_deg) & \
             (dec >= dec_min_deg) & (dec <= dec_max_deg)
    mask = np.zeros(npix)
    mask[inside] = 1.0
    return apodize(mask) if apod else mask


def f_sky(mask):
    """Effective sky fraction, <mask^2> -- the pipeline's own definition."""
    return float(np.mean(np.asarray(mask) ** 2))


def mask_geometry(mask, proj, nside=NSIDE, threshold=0.0, **jac_kw):
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
    tail reaches values no 2021 disk ever sampled.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= np.sum(w)
    return np.interp(np.asarray(q), cdf, v)
