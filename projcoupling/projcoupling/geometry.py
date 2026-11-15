"""Local geometry of a map projection, from the sky coordinates of its pixels.

Over a patch small compared with the field, a projection is a linear map J
from the sky's tangent plane to the map plane. Everything the coupling matrix
needs is the pair of singular values s1 >= s2 of J at each pixel:

    magnification  mu = s1 s2
    shear          g  = (s1 - s2) / (s1 + s2)
    misalignment   dpsi(chi) = atan2(s2 sin chi, s1 cos chi) - chi

The rotation part of J is assumed removed by flattening the polarization with
the polar angle of J, so it does not appear.

The singular values come from the metric the sphere induces on the grid: with
n(x, y) the unit vector of each pixel and (x, y) the plane coordinates in
radians, G = [[n_x.n_x, n_x.n_y], [n_x.n_y, n_y.n_y]] = J^-T J^-1, whose
eigenvalues are 1/s1^2 and 1/s2^2. Working with unit vectors needs no sky
basis, so it has no RA wrap and no coordinate singularity at the poles.
Derivatives are second-order central differences on the grid.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Geometry:
    """Singular values of the projection's Jacobian per grid pixel.

    `s1`, `s2` have the shape of the analysis grid; NaN marks pixels that no
    point of the sphere projects onto.
    """

    s1: np.ndarray
    s2: np.ndarray

    @property
    def mu(self):
        return self.s1 * self.s2

    @property
    def g(self):
        return (self.s1 - self.s2) / (self.s1 + self.s2)

    def dpsi(self, chi):
        """Angle from a plane mode to its sky mode, for mode angle `chi` to the major axis."""
        s1, s2 = self.s1[..., None], self.s2[..., None]
        return np.arctan2(s2 * np.sin(chi), s1 * np.cos(chi)) - chi

    @classmethod
    def from_radec(cls, ra_deg, dec_deg, res):
        """From per-pixel sky coordinates on the analysis grid.

        `ra_deg`, `dec_deg`: arrays of shape (ny, nx), NaN off the sphere.
        `res`: pixel size of the flat grid in radians, a scalar or
        (res_y, res_x) -- the spacing the flat-sky estimator assumes, i.e.
        Ly / ny and Lx / nx for NaMaster.
        """
        ra, dec = np.radians(ra_deg), np.radians(dec_deg)
        n = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        res_y, res_x = np.broadcast_to(np.asarray(res, dtype=float), (2,))
        n_y = np.gradient(n, axis=1, edge_order=2) / res_y
        n_x = np.gradient(n, axis=2, edge_order=2) / res_x
        a, b, d = (n_x * n_x).sum(0), (n_x * n_y).sum(0), (n_y * n_y).sum(0)
        half_tr = 0.5 * (a + d)
        disc = np.sqrt(np.maximum(half_tr**2 - (a * d - b * b), 0.0))
        return cls(s1=1 / np.sqrt(half_tr - disc), s2=1 / np.sqrt(half_tr + disc))

    @classmethod
    def from_wcs(cls, wcs, shape):
        """From an astropy celestial WCS and the grid shape (ny, nx).

        The pixel size is the WCS's projection-plane scale, which is what a
        flat-sky estimator on that grid assumes.
        """
        from astropy.wcs.utils import proj_plane_pixel_scales

        y, x = np.indices(shape)
        ra, dec = wcs.celestial.pixel_to_world_values(x, y)
        res_x, res_y = np.radians(proj_plane_pixel_scales(wcs.celestial))
        return cls.from_radec(ra, dec, (res_y, res_x))


def gsq(geometry, mask):
    """<g^2> weighted by mask^2 per unit sky area: the leakage is BB/EE = 2 <g^2>.

    A grid pixel covers sky area proportional to 1/mu.
    """
    w = mask**2 / geometry.mu
    return float(np.nansum(w * geometry.g**2) / np.nansum(w))
