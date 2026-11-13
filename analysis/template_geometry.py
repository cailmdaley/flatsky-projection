"""Geometry of the E-mode template: flattening angles and the template operator.

The pieces of `template_subtraction` and `polar_flatten` that need nothing but
a projected map's grid, a flat-sky mask and the theory EE: the polar
flattening angle, the separable 1500d field taper, and `Grid`, which builds
the template operator S_J of one projection. The `masked_template` stage
imports this module alone, so it reads only the files it is handed. The
derivations are in the docstrings of `template_subtraction` and
`polar_flatten`.
"""

import numpy as np

from spt3g import core, maps

DEG = core.G3Units.deg
ARCMIN_RAD = np.deg2rad(1.0 / 60.0)
FIELD = (50.0, -70.0, -42.0)   # 1500d: |RA| < 50 deg, dec -70 to -42
APOD_DEG = 2.5
PAD_DEG = 4.0
N_HARM = 12                 # harmonics |n| <= N_HARM of the pi-periodic symbol
N_PHI = 128                 # samples of phi in [0, pi) for the expansion
# Overall sign of S_J under this module's conventions (dpsi from `symbol`,
# E/B with U in the handedness of (x, y)), fixed by `check`: S_J[E] with
# sign +1 correlates with the grid B at r = 0.999 on SFL's 5% disk, and the
# opposite sign anticorrelates. It is a convention, the same for every
# projection; the subtracted map is B_grid - TEMPLATE_SIGN * S_J[E].
TEMPLATE_SIGN = 1.0


def _wrap(v):
    return (v + np.pi) % (2 * np.pi) - np.pi


def flatten_angles(fsm, x, y, h=1e-3):
    """FlattenPol's rot and the polar rot_R at pixel coordinates (x, y).

    The arithmetic of maputils.cxx: central differences of `xy_to_angle` with
    step h pixels, RA wrapped. Returns (rot, rot_R, alpha, delta) in radians.
    """
    def ang(xx, yy):
        a, d = fsm.xy_to_angle(core.DoubleVector(xx), core.DoubleVector(yy))
        return np.asarray(a), np.asarray(d)
    a1, d1 = ang(x - h, y)
    a2, d2 = ang(x + h, y)
    a3, d3 = ang(x, y - h)
    a4, d4 = ang(x, y + h)
    dax, ddx = _wrap(a2 - a1) / (2 * h), (d2 - d1) / (2 * h)
    day, ddy = _wrap(a4 - a3) / (2 * h), (d4 - d3) / (2 * h)
    alpha, delta = ang(x, y)
    cd = np.cos(delta)
    rot = np.arctan2(-dax, day) + np.arctan2(-ddy, -ddx)
    rot_r = 2 * np.arctan2(-ddx - cd * day, ddy - cd * dax)
    return rot, rot_r, alpha, delta


def _c2_taper(dist_deg, width=APOD_DEG):
    """Taper x - sin(2 pi x)/(2 pi), x = distance / width, of a distance inside the edge.

    This is the functional form of NaMaster's C1 kernel (its C2 kernel is
    (1 - cos pi x)/2, as in `src/stages/make_masks.taper`).
    """
    x = np.clip(dist_deg / width, 0, 1)
    return x - np.sin(2 * np.pi * x) / (2 * np.pi)


def field_masks(alpha, delta):
    """1500d mask and its halves about the central meridian, on the flat grid.

    Separable `_c2_taper` in dec and in RA arc length (RA offset times cos
    dec), 2.5 deg wide: the production field's box, with a taper whose shape
    differs from the production mask's C2 kernel and whose corners are a
    product of two tapers. `alpha`, `delta` in radians.
    """
    ra = np.degrees(_wrap(alpha))
    dec = np.degrees(delta)
    ra_max, dlo, dhi = FIELD
    cd = np.cos(delta)
    t_dec = _c2_taper(np.minimum(dec - dlo, dhi - dec))
    full = _c2_taper((ra_max - np.abs(ra)) * cd) * t_dec
    east = _c2_taper(np.minimum(ra, ra_max - ra) * cd) * t_dec
    west = _c2_taper(np.minimum(-ra, ra_max + ra) * cd) * t_dec
    for m in (full, east, west):
        m[~np.isfinite(m)] = 0.0
    return {"1500d": full, "east": east, "west": west}


def _grid(fsm):
    return np.mgrid[0:fsm.shape[0], 0:fsm.shape[1]].astype(float)


def _metric(fsm, x, y, h=1e-3):
    """Grid metric A^T A at pixel coordinates, in (sky radian / pixel)^2.

    A = d(east arc, north)/d(x, y) by central differences of `xy_to_angle`.
    Returns (G_xx, G_xy, G_yy).
    """
    def ang(xx, yy):
        a, d = fsm.xy_to_angle(core.DoubleVector(xx), core.DoubleVector(yy))
        return np.asarray(a), np.asarray(d)
    a1, d1 = ang(x - h, y)
    a2, d2 = ang(x + h, y)
    a3, d3 = ang(x, y - h)
    a4, d4 = ang(x, y + h)
    _, dc = ang(x, y)
    cd = np.cos(dc)
    ex, nx_ = cd * _wrap(a2 - a1) / (2 * h), (d2 - d1) / (2 * h)
    ey, ny_ = cd * _wrap(a4 - a3) / (2 * h), (d4 - d3) / (2 * h)
    return ex * ex + nx_ * nx_, ex * ey + nx_ * ny_, ey * ey + ny_ * ny_


def symbol(gxx, gxy, gyy, phi):
    """tan 2dpsi at grid mode angles `phi` for metrics (G_xx, G_xy, G_yy).

    dpsi = arg(G^{-1/2} l) - arg l. Shapes: metric (P,), phi (K,) -> (P, K).
    Also returns the shear g = (s1 - s2)/(s1 + s2) per pixel.
    """
    tr, det = gxx + gyy, gxx * gyy - gxy ** 2
    sd = np.sqrt(det)
    st = np.sqrt(tr + 2 * sd)
    # sqrt of a 2x2 SPD matrix: (G + sqrt(det) I) / sqrt(tr + 2 sqrt(det));
    # its inverse is adj / det(sqrt G) with det(sqrt G) = sqrt(det).
    r_xx, r_xy, r_yy = (gxx + sd) / st, gxy / st, (gyy + sd) / st
    i_xx, i_xy, i_yy = r_yy / sd, -r_xy / sd, r_xx / sd
    c, s = np.cos(phi)[None, :], np.sin(phi)[None, :]
    kx = i_xx[:, None] * c + i_xy[:, None] * s
    ky = i_xy[:, None] * c + i_yy[:, None] * s
    dpsi = np.arctan2(ky, kx) - phi[None, :]
    dpsi = (dpsi + np.pi / 2) % np.pi - np.pi / 2
    ev = np.sqrt(np.maximum((gxx - gyy) ** 2 / 4 + gxy ** 2, 0))
    s1, s2 = np.sqrt(tr / 2 + ev), np.sqrt(np.maximum(tr / 2 - ev, 0))
    return np.tan(2 * dpsi), (s1 - s2) / (s1 + s2)


def harmonics(fsm, box, window, chunk=250_000):
    """c_2n(x), n = 0..N_HARM, on `box`, times `window`; per pixel
    <tan^2 2dpsi>_phi and g.

    Evaluated only where `window` > 0 (zero elsewhere). Also returns the
    truncation error max |f - sum_{|n|<=N_HARM}| over those pixels.
    """
    yy, xx = _grid(fsm)
    shape = xx[box].shape
    sel = window.ravel() > 0
    xb, yb = xx[box].ravel()[sel], yy[box].ravel()[sel]
    del xx, yy
    phi = np.arange(N_PHI) * (np.pi / N_PHI)
    basis = np.exp(-2j * np.outer(np.arange(N_HARM + 1), phi)) / N_PHI
    c = np.empty((N_HARM + 1, xb.size), complex)
    tan2, g = np.empty(xb.size), np.empty(xb.size)
    trunc = 0.0
    for a in range(0, xb.size, chunk):
        sl = slice(a, a + chunk)
        fval, g[sl] = symbol(*_metric(fsm, xb[sl], yb[sl]), phi)
        c[:, sl] = basis @ fval.T
        tan2[sl] = np.mean(fval ** 2, axis=1)
        rec = c[0, sl][:, None].real + 2 * np.real(
            c[1:, sl].T @ np.exp(2j * np.outer(np.arange(1, N_HARM + 1), phi)))
        trunc = max(trunc, float(np.abs(rec - fval).max()))
    def full(v, dtype=float):
        o = np.zeros(v.shape[:-1] + (sel.size,), dtype)
        o[..., sel] = v
        return o.reshape(v.shape[:-1] + shape)
    c = full(c * window.ravel()[sel], complex)
    return c, full(tan2), full(g), trunc


TAPER_DEG = 2.0


class Grid:
    """Per-projection geometry: polar reflattening, masks, the template's c_2n."""

    def __init__(self, proj, qu_path, disk_mask_path, theory_path):
        """`qu_path`: any one projected Q/U map of `proj` (only its grid is
        used); `disk_mask_path`: the flatsky_masks npz of the disk the
        template covers besides the field; `theory_path`: theory_spectra."""
        self.proj = proj
        fr = next(core.G3File(str(qu_path)))
        fsm = fr["Q"]
        ny, nx = fsm.shape
        self.res = fsm.res
        yy, xx = _grid(fsm)
        rot, rot_r, alpha, delta = (v.reshape(ny, nx) for v in flatten_angles(
            fsm, xx.ravel(), yy.ravel()))
        del xx, yy
        sign = -1.0 if fsm.pol_conv == maps.MapPolConv.COSMO else 1.0
        # E/B by the FFT need the pol angle in the handedness of (x, y):
        # COSMO maps carry U with the opposite sign.
        self.usign = sign
        self.extra = np.exp(1j * sign * _wrap(rot_r - rot))
        del rot, rot_r
        fm = field_masks(alpha, delta)
        del alpha, delta
        d = np.load(disk_mask_path)
        lo, hi = int(d["lo"]), int(d["hi"])
        self.disk = dict(mask=d["mask"], box=(slice(lo, hi), slice(lo, hi)),
                         lx=float(d["lx_rad"]), ly=float(d["ly_rad"]))
        rows, cols = np.nonzero(fm["1500d"])
        pad = int(round(PAD_DEG * DEG / fsm.res))
        fbox = (slice(rows.min() - pad, rows.max() + pad + 1),
                slice(cols.min() - pad, cols.max() + pad + 1))
        self.field = dict(mask=np.ascontiguousarray(fm["1500d"][fbox]), box=fbox,
                          lx=(fbox[1].stop - fbox[1].start) * fsm.res,
                          ly=(fbox[0].stop - fbox[0].start) * fsm.res)
        del fm
        # The template box: the union of both masks' boxes. The template is
        # applied on the union of the masks' supports and tapered to zero
        # over TAPER_DEG outside it: far outside, the grid's shear grows
        # without bound (tan 2dpsi diverges at g = 1/sqrt 2), and a pure-B
        # template there is irrelevant to the masked spectra anyway.
        r0 = min(lo, fbox[0].start); r1 = max(hi, fbox[0].stop)
        c0 = min(lo, fbox[1].start); c1 = max(hi, fbox[1].stop)
        self.tbox = (slice(r0, r1), slice(c0, c1))
        support = np.zeros(self.shape_of(fsm), bool)
        support[self.disk["box"]] |= self.disk["mask"] > 0
        support[fbox] |= self.field["mask"] > 0
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~support[self.tbox]) * fsm.res / DEG
        window = 1 - _c2_taper(dist, TAPER_DEG)
        self.c, self.tan2, self.g, self.trunc = harmonics(fsm, self.tbox, window)
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.fftfreq(nx)[None, :]
        phi = np.arctan2(ky, kx)
        self.e2 = np.exp(2j * phi)
        self.shape = (ny, nx)
        # Signal power of the grid E per Fourier cell, for the Wiener filter:
        # the input unlensed EE at |l| times the 2' top-hat pixel window.
        # SFL and ZEA are equal-area, so l on the grid is l on the sky up to
        # the shear; the bilinear-interpolation transfer is left out.
        ell = 2 * np.pi * np.hypot(ky, kx) / fsm.res
        th = np.load(theory_path)
        self.c_ee = (np.interp(ell, th["ell"].astype(float), th["unlensed_EE"])
                     * (np.sinc(ky) * np.sinc(kx)) ** 2)

    @staticmethod
    def shape_of(fsm):
        return tuple(fsm.shape)

    def sub(self, arr, which):
        """`arr` on the template box cut to the mask's box."""
        mb = getattr(self, which)["box"]
        return arr[..., mb[0].start - self.tbox[0].start: mb[0].stop - self.tbox[0].start,
                   mb[1].start - self.tbox[1].start: mb[1].stop - self.tbox[1].start]

    def weighted(self, arr, which):
        """W^2-weighted mean of a template-box map over a mask."""
        w = getattr(self, which)["mask"] ** 2
        return float(np.sum(w * self.sub(arr, which)) / np.sum(w))

    # -- the operators --------------------------------------------------
    def eb(self, q, u):
        """E_l, B_l of a grid Q/U (FFT over the whole grid)."""
        qf, uf = np.fft.fft2(q), np.fft.fft2(u)
        c, s = self.e2.real, self.e2.imag
        uf = self.usign * uf
        return c * qf + s * uf, -s * qf + c * uf

    def template(self, e_l):
        """S_J[E] on the template box (zero outside), real space."""
        b = self.c[0].real * np.fft.ifft2(e_l).real[self.tbox]
        en = np.ones(self.shape, complex)
        for n in range(1, N_HARM + 1):
            en = en * self.e2
            b += 2 * np.real(self.c[n] * np.fft.ifft2(en * e_l)[self.tbox])
        out = np.zeros(self.shape)
        out[self.tbox] = b
        return out

    def qu_of_b(self, b):
        """The pure-B Q/U of a real-space B map."""
        bl = np.fft.fft2(b)
        c, s = self.e2.real, self.e2.imag
        return (np.fft.ifft2(-s * bl).real,
                self.usign * np.fft.ifft2(c * bl).real)
