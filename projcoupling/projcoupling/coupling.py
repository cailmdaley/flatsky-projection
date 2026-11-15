"""The projection's coupling matrix, and decoupling bandpowers with it.

A flat-sky pseudo-C_l estimator on a projected grid measures, in each
annulus of plane wavenumber k, the sky spectrum read at the sheared
wavenumber k rho(chi) and mixed between E and B by the misalignment
dpsi(chi), averaged over mode direction chi and over the footprint:

    C^X_b = sum_l' R^{X<-Y}_{b l'} C^Y_l'
    R^{X<-Y}_{b l'} = (1/N) int d^2n W^2 mu^2 < K^{X<-Y}(chi) [l' <- k rho(chi)] >_{chi, k in b}
    N = int d^2n W^2 mu,      rho(chi) = sqrt(s1^2 cos^2 chi + s2^2 sin^2 chi)

with kernels K = 1 (TT<-TT), cos^2 2dpsi (EE<-EE, BB<-BB), sin^2 2dpsi
(BB<-EE, EE<-BB), cos 2dpsi (TE<-TE). On the grid, d^2n = d^2x / mu, so the
footprint weight per grid pixel is W^2 mu and the normalisation sum W^2.

Two conventions fix the map from plane to sphere. A plane mode of wavenumber
q is the spherical multipole l = q - k_offset, k_offset = 1/2 by default (the
large-l correspondence between Legendre and Bessel functions). And a bandpower
is the uniform average of the plane spectrum over k in [lo, hi), which is how
NaMaster bins a theory spectrum. The input spectrum is read between integer
multipoles by linear interpolation, and the average over k within each band is
done exactly for that interpolant, so R needs no sampling in k.

R depends on a pixel only through (s1, s2), so pixels are first collected in a
weighted histogram of (log s1, log s2), each occupied cell placed at its
weighted centroid. The cost is then set by the number of cells rather than of
pixels. This is exact up to the spread of (s1, s2) within a cell, which is
second order in the cell width.

The one approximation is locality: the projection's shear varies on the scale
of the field, multipoles L of a few, so applying it locally to modes at
l >> L errs at order (L / l)^2.
"""

import numpy as np

BLOCKS = ("TT<-TT", "EE<-EE", "BB<-BB", "BB<-EE", "EE<-BB", "TE<-TE")


def _cells(geometry, mask, n_cells):
    """Occupied (s1, s2) cells: centroids and summed weight W^2 mu / sum W^2."""
    w2 = np.asarray(mask, dtype=float) ** 2
    good = (w2 > 0) & np.isfinite(geometry.s1)
    l1, l2 = np.log(geometry.s1[good]), np.log(geometry.s2[good])
    v = w2[good] * geometry.mu[good]

    flat = np.zeros(l1.size, dtype=np.intp)
    for l in (l1, l2):
        lo, span = l.min(), max(l.max() - l.min(), 1e-12)
        flat = flat * n_cells + np.minimum((n_cells * (l - lo) / span).astype(np.intp), n_cells - 1)
    weight = np.bincount(flat, weights=v)
    occ = weight > 0
    s1 = np.exp(np.bincount(flat, weights=v * l1)[occ] / weight[occ])
    s2 = np.exp(np.bincount(flat, weights=v * l2)[occ] / weight[occ])
    return s1, s2, weight[occ] / w2.sum()


def _deposit(rows, x, v, n_rows, n_cols):
    """Accumulate v * Phi(x - j) on columns j, Phi the integral of the unit hat.

    Phi(u) is 1 for u >= 1, 1 - (1-u)^2/2 on [0, 1), u'^2/2 at u = u' - 1 on
    [-1, 0), and 0 below. The constant part is deposited once at its last
    column and spread by a reverse cumulative sum, returned separately so
    every chunk shares one cumsum.
    """
    j0 = np.floor(x).astype(np.intp)
    f = x - j0
    step = np.bincount(rows * (n_cols + 1) + np.clip(j0, 0, n_cols), weights=v,
                       minlength=n_rows * (n_cols + 1))
    edge = np.zeros(n_rows * n_cols)
    for col, t in ((j0, 1 - 0.5 * (1 - f) ** 2), (j0 + 1, 0.5 * f**2)):
        ok = (col >= 0) & (col < n_cols)
        edge += np.bincount(rows[ok] * n_cols + col[ok], weights=(v * t)[ok],
                            minlength=n_rows * n_cols)
    return step, edge


def coupling_matrix(geometry, mask, edges, ell_in, k_offset=0.5, n_chi=64,
                    n_cells=128, chunk=2_000_000):
    """R^{X<-Y}: input C_l' on `ell_in` -> bandpowers on `edges`, per block.

    geometry : `Geometry` of the analysis grid.
    mask     : analysis mask W on the same grid (the one NaMaster sees).
    edges    : bandpower edges in plane wavenumber, bands [edges[i], edges[i+1]).
    ell_in   : consecutive integer multipoles the input spectrum is given on.
    k_offset : plane wavenumber minus spherical multipole.
    n_chi    : mode directions over a quarter turn (the kernels have that period).
    n_cells  : histogram cells per axis in (log s1, log s2).

    Returns {block: array (n_bands, len(ell_in))} for the blocks in `BLOCKS`.
    EE<-EE and BB<-BB are the same array, as are BB<-EE and EE<-BB.
    """
    s1, s2, w = _cells(geometry, mask, n_cells)
    chi = (np.arange(n_chi) + 0.5) * (0.5 * np.pi / n_chi)
    rho = np.sqrt((s1[:, None] * np.cos(chi)) ** 2 + (s2[:, None] * np.sin(chi)) ** 2)
    dpsi = np.arctan2(s2[:, None] * np.sin(chi), s1[:, None] * np.cos(chi)) - chi
    kernels = {"TT<-TT": np.ones_like(dpsi), "BB<-EE": np.sin(2 * dpsi) ** 2,
               "TE<-TE": np.cos(2 * dpsi)}

    edges = np.asarray(edges, dtype=float)
    lo, hi = edges[:-1], edges[1:]
    n_b, n_l, ell0 = lo.size, len(ell_in), float(ell_in[0])
    step = {k: 0.0 for k in kernels}
    edge = {k: 0.0 for k in kernels}

    per_cell = n_chi * n_b
    for c0 in range(0, s1.size, max(1, chunk // per_cell)):
        sl = slice(c0, c0 + max(1, chunk // per_cell))
        r = rho[sl][..., None]
        a = lo * r - k_offset - ell0
        b = hi * r - k_offset - ell0
        # Mean of the interpolant over [a, b] is its integral over the width.
        base = w[sl, None, None] / n_chi / (b - a)
        rows = np.broadcast_to(np.arange(n_b), a.shape).ravel()
        x = np.concatenate([b.ravel(), a.ravel()])
        rows2 = np.concatenate([rows, rows])
        for name, K in kernels.items():
            v = (base * K[sl][..., None]).ravel()
            s, e = _deposit(rows2, x, np.concatenate([v, -v]), n_b, n_l)
            step[name] = step[name] + s
            edge[name] = edge[name] + e

    R = {}
    for name in kernels:
        cum = np.cumsum(step[name].reshape(n_b, n_l + 1)[:, ::-1], axis=1)[:, ::-1]
        R[name] = cum[:, 1:] + edge[name].reshape(n_b, n_l)
    R["EE<-EE"] = R["BB<-BB"] = R["TT<-TT"] - R["BB<-EE"]
    R["EE<-BB"] = R["BB<-EE"]
    return {k: R[k] for k in BLOCKS}


def bin_spectrum(cl, ell_in, edges):
    """Uniform mean of C_l over the integers l in each [edges[i], edges[i+1])."""
    ell_in = np.asarray(ell_in)
    return np.array([np.mean(cl[(ell_in >= a) & (ell_in < b)])
                     for a, b in zip(edges[:-1], edges[1:])])


def decouple(cl, R, fiducial, ell_in, sky_edges):
    """Undo the projection: measured bandpowers -> bandpowers of the sky.

    The sky spectrum Y is modelled as one free amplitude per band of
    `sky_edges` times the shape `fiducial[Y]` (on `ell_in`), the amplitudes
    fitted to `cl` by least squares through R -- a direct solve when the sky
    bands match the measured ones. Spectra coupled by R (EE and BB) are
    solved jointly.

    cl       : {X: measured bandpowers, shape (..., n_bands)}, mask already
               decoupled (e.g. NaMaster's output, see `namaster_spectra`).
    R        : the blocks from `coupling_matrix`.
    fiducial : {Y: model sky C_l on ell_in}, in absolute units: its shape
               fixes each band's profile, and its values are used as they
               stand outside the sky bands.
    Returns {Y: sky bandpowers on sky_edges, shape (..., n_sky_bands)}, the
    fitted amplitudes times the band means of the fiducial.
    """
    outs = [x for x in cl if any(k.startswith(x + "<-") for k in R)]
    ins = [y for y in fiducial if any(f"{x}<-{y}" in R for x in outs)]
    ell_in = np.asarray(ell_in)
    bands = np.array([(ell_in >= a) & (ell_in < b)
                      for a, b in zip(sky_edges[:-1], sky_edges[1:])])
    n_out = R[next(iter(R))].shape[0]
    design = np.block([[R[f"{x}<-{y}"] @ (bands * fiducial[y]).T if f"{x}<-{y}" in R
                        else np.zeros((n_out, len(bands))) for y in ins] for x in outs])
    exterior = ~bands.any(0)
    data = np.concatenate([np.asarray(cl[x]) - sum(R[f"{x}<-{y}"] @ (fiducial[y] * exterior)
                                                    for y in ins if f"{x}<-{y}" in R)
                           for x in outs], axis=-1)
    amp = data @ np.linalg.pinv(design).T
    n = len(bands)
    return {y: amp[..., i * n:(i + 1) * n] * bin_spectrum(fiducial[y], ell_in, sky_edges)
            for i, y in enumerate(ins)}


def namaster_spectra(cl):
    """Name NaMaster's decoupled spectra: 1 row TT, 2 rows TE TB, 4 rows EE EB BE BB."""
    names = {1: ("TT",), 2: ("TE", "TB"), 4: ("EE", "EB", "BE", "BB")}[len(cl)]
    return dict(zip(names, cl))
