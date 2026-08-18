#!/usr/bin/env python3
"""Evaluate the shear-leakage law BB/EE = 2<g^2> on the real BK18 apodization
mask (BICEP/Keck public data release), on BICEP's real map grid.

Data: bk18_mask_largefield_cel_n0512.fits, downloaded from
http://bicepkeck.org/BK18_datarelease/bk18_mask_largefield_cel_n0512.fits
(the public apodization/inverse-variance mask used in the BK18 power-spectrum
analysis for the "large field" BICEP3/WMAP/Planck maps; Healpix, Nside=512,
RING, celestial coordinates). Cached at
/sptlocal/user/cailmdaley/pol_project/external/bk18/.

Grid: BICEP/Keck's standard flat map is the "BICEP projection", an
equirectangular/CAR-like grid with

    x = (alpha - alpha0) * cos(delta0),   y = delta - delta0

centered on their standard deep field "ra0hdec-57.5", i.e. delta0 = -57.5 deg,
alpha0 = 0. Public map products use 0.25 deg pixels on this grid (see e.g.
BICEP2/Keck papers, and the RA/Dec footprint recovered from the mask below:
RA in [-60, 60] deg, Dec in [-71.5, -39.2] deg -- consistent with the standard
BICEP field boundaries).

Closed-form shear of this projection (see analysis/shear_leakage.py and the
felt fiber shear-predicts-leakage.md for the general law; here specialized to
BICEP's CAR grid): the local Jacobian d(x,y)/d(east,north) at declination
delta is diag(cos(delta0)/cos(delta), 1) -- pure axis-aligned magnification
along x, no shear from rotation (this projection has no cross term). Singular
values are s1 = max(r,1), s2 = min(r,1) with r = cos(delta0)/cos(delta), so

    g = |r - 1| / (r + 1),   r = cos(delta0) / cos(delta)

We regrid the Healpix mask onto BICEP's native 0.25 deg CAR grid (nearest
neighbor -- the mask is a smooth apodization taper, not a noisy field, so NN
resampling does not bias its low moments), compute g per grid pixel from the
closed form, and evaluate 2<g^2> under several mask-power weightings.
"""
import numpy as np
import healpy as hp
from astropy.io import fits

MASK_URL = "http://bicepkeck.org/BK18_datarelease/bk18_mask_largefield_cel_n0512.fits"
MASK_PATH = "/sptlocal/user/cailmdaley/pol_project/external/bk18/bk18_mask_largefield_cel_n0512.fits"

DELTA0 = -57.5      # deg, BICEP standard field center declination
ALPHA0 = 0.0         # deg
PIX_DEG = 0.25        # deg, BICEP/Keck public map pixel size
RA_HALF_WIDTH = 60.0   # deg, matches the mask footprint (RA in [-60, 60])
DEC_MIN, DEC_MAX = -71.5, -39.0  # deg, matches the mask footprint


def load_grid(pix_deg=PIX_DEG):
    """Build BICEP's native CAR grid and regrid the public mask onto it."""
    m = hp.read_map(MASK_PATH)
    nside = hp.get_nside(m)
    m = np.nan_to_num(m, nan=0.0)

    dec_centers = np.arange(DEC_MIN + pix_deg / 2, DEC_MAX, pix_deg)
    # x = (alpha - alpha0) * cos(delta0), so alpha spacing in the *sky* is
    # pix_deg / cos(delta0) to keep the pixel a fixed physical x-extent --
    # this is what makes the map square in (x, y), matching BICEP's grid.
    ra_half = RA_HALF_WIDTH
    n_ra = int(round(2 * ra_half * np.cos(np.deg2rad(DELTA0)) / pix_deg))
    ra_centers = ALPHA0 + (np.arange(n_ra) - (n_ra - 1) / 2) * (pix_deg / np.cos(np.deg2rad(DELTA0)))

    RA, DEC = np.meshgrid(ra_centers, dec_centers)
    theta = np.deg2rad(90 - DEC)
    phi = np.deg2rad(RA % 360)
    pix = hp.ang2pix(nside, theta, phi)
    W = m[pix]
    return RA, DEC, W


def shear_g(dec_deg, delta0=DELTA0):
    r = np.cos(np.deg2rad(delta0)) / np.cos(np.deg2rad(dec_deg))
    return np.abs(r - 1) / (r + 1)


def weighted_mean(x, w):
    ws = w.sum()
    return float(np.sum(x * w) / ws) if ws > 0 else np.nan


def main():
    RA, DEC, W = load_grid()
    g = shear_g(DEC)
    g2 = g ** 2

    footprint = W > 0
    print(f"BICEP-grid pixels total: {W.size}, footprint (W>0): {footprint.sum()}")
    print(f"delta0 = {DELTA0} deg, pixel = {PIX_DEG} deg, "
          f"RA range [{RA.min():.2f},{RA.max():.2f}], Dec range [{DEC.min():.2f},{DEC.max():.2f}]")
    print(f"mask value range: [{W.min():.4g}, {W.max():.4g}], W>0 count {footprint.sum()}")

    results = {}
    # unweighted over footprint
    results["footprint (unweighted, W>0)"] = 2 * np.mean(g2[footprint])

    for p in (1, 2, 4):
        Wp = W ** p
        results[f"W^{p}"] = 2 * weighted_mean(g2, Wp)

    # threshold variant: mask^2 but only pixels above a small fraction of peak
    thresh = 0.01 * W.max()
    keep = W > thresh
    results["W^2, W>1% peak"] = 2 * weighted_mean(g2[keep], (W ** 2)[keep])

    print("\n2<g^2> under different weightings:")
    for k, v in results.items():
        print(f"  {k:28s}: {v:.6e}")

    try:
        import theory
        d = theory.generate(lmax=2000)
        ell, BB, EE = d["ell"], d["lensed_BB"], d["lensed_EE"]
        print("\nContext -- lensing BB/EE, l=100..1000:")
        for l0 in (100, 300, 500, 1000):
            print(f"  l={l0:5d}: {BB[l0] / EE[l0]:.4e}")
        print("\nProjection leakage (W^2, real mask) vs lensing BB/EE: "
              f"{results['W^2'] / (BB[100] / EE[100]):.1f}x the l=100 lensing "
              "level; well above BK's quoted r<1e-4 matrix-purification residual.")
    except Exception as e:
        print(f"\n(theory comparison skipped: {e})")

    return results, RA, DEC, W, g


if __name__ == "__main__":
    main()
