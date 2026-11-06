"""Signal-only scalar lognormal control for statistics beyond two-point.

Central angular scale is fixed: Gaussian sigma=0.5 degree, wavelet is the
0.5-minus-1 degree difference of Gaussians. This is NOT a lensing survey
forecast. A spherical harmonic filter of the same nonlinear realization is
the truth. Overscan keeps the flat convolution away from map boundaries.
All statistics use spherical area weights, except peak NUMBER (each maximum
counts once). Truth and planar results use the identical chart pixel grid.

Run through the additive ASTRA beyond_two_point output on OSG.
"""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

PROJECTIONS = ("TAN", "STG", "ZEA")
DIAMETERS = (14.6, 30., 60.)
STATISTICS = ("raw_variance", "raw_third", "smooth_variance", "smooth_third",
              "wavelet_l1", "peaks_gt_2sigma")
SIGMA = np.deg2rad(.5)


def radius(c, chart):
    return {"TAN": np.tan, "STG": lambda x: 2*np.tan(x/2),
            "ZEA": lambda x: 2*np.sin(x/2)}[chart](c)


def chart_grid(chart, diameter, pixel_deg):
    """Inverse azimuthal projection centred on (RA,dec)=(0,0)."""
    pixel = np.deg2rad(pixel_deg)
    edge = radius(np.deg2rad(diameter/2), chart)
    n = int(np.ceil((edge+5*2*SIGMA)/pixel))
    axis = np.arange(-n, n+1)*pixel
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y)
    c = {"TAN": np.arctan, "STG": lambda z: 2*np.arctan(z/2),
         "ZEA": lambda z: 2*np.arcsin(z/2)}[chart](r)
    sinc_r = np.divide(np.sin(c), r, out=np.ones_like(r), where=r != 0)
    theta = np.arccos(y*sinc_r)
    phi = np.mod(np.arctan2(x*sinc_r, np.cos(c)), 2*np.pi)
    if chart == "TAN":
        mu = 1/np.cos(c)**3
        g = (1-np.cos(c))/(1+np.cos(c))
    elif chart == "STG":
        mu = 1/np.cos(c/2)**4
        g = np.zeros_like(c)
    else:
        mu = np.ones_like(c)
        g = (1-np.cos(c/2)**2)/(1+np.cos(c/2)**2)
    mask = c <= np.deg2rad(diameter/2)
    weight = (1/mu)[mask]
    return theta, phi, mask, weight, g[mask], mu[mask]


def moments(v, weight):
    mean = np.average(v, weights=weight)
    return np.average((v-mean)**2, weights=weight), np.average((v-mean)**3, weights=weight)


def statistics(raw, smooth, broad, mask, weight, threshold, pixel_deg=.1):
    wavelet = smooth-broad
    # A fixed physical neighbourhood prevents sub-HEALPix interpolation
    # ripples from becoming extra peaks when chart pixels are refined.
    # Peaks are maxima in a +/-0.3 degree square, not just 8 neighbours.
    half_width = int(round(.3/pixel_deg))
    neighbours = maximum_filter(smooth, size=2*half_width+1, mode="nearest")
    peaks = mask & (smooth == neighbours) & (smooth > threshold)
    return np.array((*moments(raw[mask], weight), *moments(smooth[mask], weight),
                     np.average(abs(wavelet[mask]), weights=weight), peaks.sum()))


def lognormal(seed, nside, lmax):
    ell = np.arange(lmax+1)
    cl = np.exp(-ell*(ell+1)*np.deg2rad(.25)**2)/(ell+20.)**1.5
    cl[:2] = 0
    variance = np.sum((2*ell+1)*cl)/(4*np.pi)
    cl *= .7**2/variance
    np.random.seed(12000+seed)
    alm = hp.synalm(cl, lmax=lmax, new=True)
    gaussian = hp.alm2map(alm, nside, lmax=lmax)
    field = np.expm1(gaussian-.7**2/2)
    nonlinear_alm = hp.map2alm(field, lmax=lmax, iter=1)
    smoothed = [hp.alm2map(hp.almxfl(nonlinear_alm,
                np.exp(-.5*ell*(ell+1)*s**2)), nside, lmax=lmax)
                for s in (SIGMA, 2*SIGMA)]
    return field, smoothed


def run(seed, nside, pixel_deg):
    lmax = 1024  # Fixed across sampling-resolution controls; identical alms.
    field, smoothed = lognormal(seed, nside, lmax)
    threshold = 2*np.std(smoothed[0])
    rows = []
    for diameter in DIAMETERS:
        for chart in PROJECTIONS:
            theta, phi, mask, weight, g, mu = chart_grid(chart, diameter, pixel_deg)
            raw = hp.get_interp_val(field, theta, phi)
            truth = [hp.get_interp_val(m, theta, phi) for m in smoothed]
            planar = [gaussian_filter(raw, s/np.deg2rad(pixel_deg), mode="nearest",
                                      truncate=5.) for s in (SIGMA, 2*SIGMA)]
            # raw field is identical in both arms, providing an exact null.
            reference = statistics(raw, *truth, mask, weight, threshold, pixel_deg)
            measured = statistics(raw, *planar, mask, weight, threshold, pixel_deg)
            rows.append(dict(chart=chart, diameter=diameter,
                g2=float(np.average(g*g, weights=weight)),
                mean_log_mu=float(np.average(np.log(mu), weights=weight)),
                reference=reference.tolist(), measured=measured.tolist()))
    return dict(seed=seed, nside=nside, lmax=lmax, pixel_deg=pixel_deg,
                sigma_deg=.5, lognormal_sigma=.7, statistics=STATISTICS, rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--pixel-deg", type=float, default=.1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.nside, a.pixel_deg)
    name = f"seed{a.seed:03d}_n{a.nside}_p{a.pixel_deg:.3f}.json"
    (a.out/name).write_text(json.dumps(result, indent=2)+"\n")
    print(name, flush=True)
