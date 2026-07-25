"""Exact mode coupling for a flat-sky pseudo-C_l estimator that does not decouple.

Why this exists
---------------
`quick_flatsky_routines.cl_flatsky` -- the estimator behind Crawford's
bandpowers -- multiplies the map by an apodization mask, takes an FFT, and
divides by <mask^2>. That is the whole mask treatment. It is *not* a MASTER
deconvolution: there is no coupling matrix and no inverse.

Dividing by <mask^2> is exact only when the mask's own power is concentrated
at |q| small compared with the scale over which C_l varies. A smooth,
generously apodized mask satisfies that well. A mask with a *hard* edge does
not: its transform falls only as a power law, so the estimator scatters power
across the band, and with a CMB spectrum falling by four decades the leakage
runs downhill from large scales into small ones.

That matters here because Crawford's two mask variants are made by zeroing
pixels outright, which destroys the apodization along the new boundary, while
his full mask keeps it. So the variants and the full mask are not estimated
the same way, and the difference is computable without simulating anything.

The calculation
---------------
For a map g = W f on an N x N grid of pixel size `res`, with the estimator's
own normalisation, the expected binned spectrum is a normalised convolution

    Ctilde(k) = sum_q K(q) C(k - q),      K(q) = |W~(q)|^2 / (N^4 <W^2>)

with W~ the unnormalised DFT of the mask. K sums to one exactly, which is the
statement that <mask^2> gets the *mean* level right and only redistributes.
Everything below evaluates that convolution with FFTs on the grid the
estimator itself uses, so the wraparound in `k - q` is the estimator's own
discrete structure rather than an approximation we impose.

The output to look at is `bias`, Ctilde/C after azimuthal binning: the
multiplicative error the estimator makes on a given input spectrum and mask.
For the projection ratio what matters is the *difference* of the two biases,
since proj0 and proj5 are measured through differently-shaped masks.
"""

import numpy as np

import crawford_masks as CM

DELTA_L = 100.0


def ell_grid(ngrid, res_arcmin):
    """|l| at every FFT cell, in the estimator's own convention l = 2*pi*|u|."""
    res = np.deg2rad(res_arcmin / 60.0)
    u = np.fft.fftfreq(ngrid, d=res)
    return 2 * np.pi * np.hypot(u[:, None], u[None, :])


def kernel(mask):
    """K(q) = |W~(q)|^2 / (N^4 <W^2>), the normalised coupling kernel.

    Returned in FFT order with q = 0 at index [0, 0], and summing to 1 to
    machine precision -- which is worth asserting, because it is exactly the
    statement that this kernel only moves power around.
    """
    mask = np.asarray(mask, dtype=np.float64)
    n = mask.shape[0]
    wt = np.fft.fft2(mask)
    k = (wt.real ** 2 + wt.imag ** 2) / (n ** 4 * np.mean(mask ** 2))
    return k


def couple(cl_2d, kern):
    """The normalised convolution sum_q K(q) C(k-q), by FFT."""
    return np.fft.ifft2(np.fft.fft2(kern) * np.fft.fft2(cl_2d)).real


def bin_annuli(field_2d, ell2d, n_bins, delta_l=DELTA_L):
    """Flat average over 2D cells per band -- exactly what `cl_flatsky` does."""
    idx = np.floor(ell2d / delta_l).astype(int).ravel()
    keep = idx < n_bins
    tot = np.bincount(idx[keep], weights=field_2d.ravel()[keep], minlength=n_bins)
    num = np.bincount(idx[keep], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot[:n_bins] / num[:n_bins]


def bias(mask, cl_of_ell, ngrid, res_arcmin, n_bins=60, delta_l=DELTA_L):
    """Multiplicative error the estimator makes: binned Ctilde / binned C.

    `cl_of_ell` is a callable taking |l| and returning C_l, so the caller
    controls the band limit and the units. Anything the caller sets to zero
    above a band limit stays zero in the input and is *not* forced to stay
    zero in the output -- leakage across the band limit is a real prediction
    of this calculation, and is how power appears where none was put in.
    """
    ell2d = ell_grid(ngrid, res_arcmin)
    cl2d = cl_of_ell(ell2d)
    tilde = couple(cl2d, kernel(mask))
    num = bin_annuli(tilde, ell2d, n_bins, delta_l)
    den = bin_annuli(cl2d, ell2d, n_bins, delta_l)
    return num / den, num, den


def spectrum_interpolator(ell_theory, cl_theory, band_limit=None):
    """A callable C(|l|) built from a tabulated spectrum, zero past the limit."""
    ell_theory = np.asarray(ell_theory, dtype=float)
    cl = np.asarray(cl_theory, dtype=float).copy()
    if band_limit is not None:
        cl = np.where(ell_theory > band_limit, 0.0, cl)

    def f(ell):
        return np.interp(ell, ell_theory, cl, left=0.0, right=0.0)
    return f
