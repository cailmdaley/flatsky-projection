"""HEALPix pixels as parallelograms: their shear and the window it implies.

Each pixel is taken as the parallelogram its corners span on the local tangent
plane. `pixel_shear` reads the shear g of the map from a pixel's two diagonals
to the sky. `sheared_window` averages |W|^2 over mode direction and pixels,
with W the exact two-sinc transform of the parallelogram, and compares it with
`hp.pixwin` and with a square pixel of the same area. Nothing is fitted.

Run in the SPT environment:
    ~/code/python/spt3g_software/build/env-shell.sh python analysis/healpix_sheared_window.py
"""

import healpy as hp
import numpy as np


def _tangent_vectors(nside, ids):
    """Diagonals of the pixels `ids` in the local east/north frame, radians."""
    b = hp.boundaries(nside, ids, step=1, nest=True)   # (n, 3, 4): N, E, S, W
    th, ph = hp.pix2ang(nside, ids, nest=True)
    z = np.cos(th)
    n = np.stack([-z * np.cos(ph), -z * np.sin(ph), np.sqrt(1 - z**2)], 1)
    e = np.stack([-np.sin(ph), np.cos(ph), 0 * ph], 1)
    project = lambda d: np.stack([(d * e).sum(1), (d * n).sum(1)], 1)
    return project(b[:, :, 0] - b[:, :, 2]), project(b[:, :, 1] - b[:, :, 3]), z


def pixel_shear(nside=1024, stride=211):
    """g per sampled pixel, with the pixel's |z|-zone label, and the area check."""
    d0, d1, z = _tangent_vectors(nside, np.arange(0, hp.nside2npix(nside), stride))
    s = np.linalg.svd(np.stack([d0, d1], 2), compute_uv=False)
    area = s[:, 0] * s[:, 1] / 2                       # rhombus: |d0 x d1| / 2
    return dict(g=(s[:, 0] - s[:, 1]) / (s[:, 0] + s[:, 1]), z=z,
                area_ratio=area / (4 * np.pi / hp.nside2npix(nside)))


def sheared_window(nside, x, npix_sample=6000, nchi=90):
    """Window at l = x Nside: `hp.pixwin`, parallelogram pixels, square pixels."""
    npix = hp.nside2npix(nside)
    ids = np.arange(0, npix, npix // npix_sample)
    d0, d1, _ = _tangent_vectors(nside, ids)
    s1, s2 = (d0 + d1) / 2, (d0 - d1) / 2              # the two edge vectors
    chi = (np.arange(nchi) + .5) * np.pi / nchi
    u = np.stack([np.cos(chi), np.sin(chi)], 1)
    side = np.sqrt(4 * np.pi / npix)
    w = hp.pixwin(nside, lmax=3 * nside - 1)
    ell = int(x * nside)
    W = (np.sinc(ell * (s1 @ u.T) / 2 / np.pi)
         * np.sinc(ell * (s2 @ u.T) / 2 / np.pi))
    square = (np.sinc(ell * side * np.cos(chi) / 2 / np.pi)
              * np.sinc(ell * side * np.sin(chi) / 2 / np.pi))
    return w[ell], np.sqrt((W**2).mean()), np.sqrt((square**2).mean())


if __name__ == "__main__":
    px = pixel_shear()
    g, z = px["g"], px["z"]
    print(f"area/expected: mean {px['area_ratio'].mean():.5f} std {px['area_ratio'].std():.1e}")
    print(f"g: mean {g.mean():.4f} max {g.max():.4f}; <g^2> all sky {(g**2).mean():.5f}")
    print(f"  eq zone <g^2> {(g[abs(z) < 2/3]**2).mean():.5f}, "
          f"polar caps <g^2> {(g[abs(z) >= 2/3]**2).mean():.5f}")
    F = ((1 + g**2) / (1 - g**2)).mean()
    print(f"predicted <(1+g^2)/(1-g^2)> = {F:.5f}")
    for nn in (1024, 2048, 4096):
        w = hp.pixwin(nn, lmax=2 * nn)
        l = np.arange(len(w))
        for hi in (0.2, 0.5):
            m = (l / nn > 0.05) & (l / nn < hi)
            c1 = np.polyfit((l[m] / nn)**2, -np.log(w[m]) / (l[m] / nn)**2, 1)[1]
            print(f"Nside {nn} x<{hi}: measured c={c1:.5f}  square pixel "
                  f"pi/72={np.pi/72:.5f}  sheared {np.pi/72*F:.5f}")
    print("full window, parallelogram pixels against hp.pixwin")
    for nn in (1024, 4096):
        print(f"Nside {nn}:  l/Nside  pixwin  sheared  square")
        for x in (0.5, 1, 1.5, 2, 2.5, 3 - 1 / nn):
            p, s, q = sheared_window(nn, x)
            print(f"   {x:.2f}  {p:.5f}  {s:.5f}  {q:.5f}")
