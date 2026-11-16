import numpy as np
import pytest

from projcoupling import Geometry, bin_spectrum, coupling_matrix, decouple, gsq

ELL = np.arange(0, 12001)
EDGES = np.arange(1000, 4001, 50)


def constant_shear(g, shape=(40, 40)):
    """Equal-area geometry with shear g everywhere: s1 = sqrt((1+g)/(1-g)), s2 = 1/s1."""
    s1 = np.sqrt((1 + g) / (1 - g))
    return Geometry(np.full(shape, s1), np.full(shape, 1 / s1))


def wcs_grid(proj, n=600, res_arcmin=2.0, dec0=-56.0):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
    w.wcs.crval = [0.0, dec0]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-res_arcmin / 60, res_arcmin / 60]
    return w, (n, n)


@pytest.mark.parametrize("g", [0.05, 0.2, 0.4])
def test_leakage_law_constant_shear(g):
    """A white sky leaks <sin^2 2dpsi> = 2g^2 (1 - 3g^2/4) of E into B, exactly."""
    R = coupling_matrix(constant_shear(g), np.ones((40, 40)), EDGES, ELL)
    one = np.ones(ELL.size)
    leak = 2 * g**2 * (1 - 0.75 * g**2)
    np.testing.assert_allclose(R["BB<-EE"] @ one, leak, rtol=1e-10)
    np.testing.assert_allclose(R["EE<-EE"] @ one, 1 - leak, rtol=1e-10)
    np.testing.assert_allclose(R["TT<-TT"] @ one, 1, rtol=1e-12)
    assert abs(gsq(constant_shear(g), np.ones((40, 40))) - g**2) < 1e-14


@pytest.mark.parametrize("g", [0.1, 0.3])
def test_minus_two_power_law_invariant(g):
    """For mu = 1, <rho^-2>_chi = 1/(s1 s2) = 1, so C ~ l^-2 is unchanged at every order in g."""
    ell = np.arange(1, 12001)
    R = coupling_matrix(constant_shear(g), np.ones((40, 40)), EDGES, ell, k_offset=0.0)
    lo, hi = EDGES[:-1], EDGES[1:]
    np.testing.assert_allclose(R["TT<-TT"] @ ell**-2.0, 1 / (lo * hi), rtol=2e-6)


def test_te_suppression():
    g = 0.05
    R = coupling_matrix(constant_shear(g), np.ones((40, 40)), EDGES, ELL)
    te = R["TE<-TE"] @ np.ones(ELL.size)
    assert np.all(abs(te - (1 - g**2)) < 2 * g**4)


def test_conformal_does_not_leak():
    w, shape = wcs_grid("STG")
    geo = Geometry.from_wcs(w, shape)
    mask = np.ones(shape)
    assert gsq(geo, mask) < 1e-12
    R = coupling_matrix(geo, mask, EDGES, ELL)
    cl = (ELL + 1.0) ** -2.5
    assert np.all(R["BB<-EE"] @ cl < 1e-10 * (R["EE<-EE"] @ cl))
    assert geo.mu.max() > 1.005  # magnified, yet no leakage


def test_zea_geometry_closed_form():
    """ZEA from a WCS: mu = 1 and g = (1 - cos r) / (3 + cos r) at angular radius r."""
    w, shape = wcs_grid("ZEA")
    geo = Geometry.from_wcs(w, shape)
    y, x = np.indices(shape)
    ra, dec = (np.radians(c) for c in w.pixel_to_world_values(x, y))
    cos_r = np.sin(dec) * np.sin(np.radians(-56)) + np.cos(dec) * np.cos(np.radians(-56)) * np.cos(ra)
    np.testing.assert_allclose(geo.mu, 1, atol=1e-6)
    np.testing.assert_allclose(geo.g, (1 - cos_r) / (3 + cos_r), atol=1e-7)


@pytest.mark.parametrize("sky_step", [50, 500])
def test_decouple_recovers_sky(sky_step):
    """Bandpowers forward-modelled through R come back as the sky's band means."""
    w, shape = wcs_grid("ZEA", n=300, res_arcmin=4.0)
    geo = Geometry.from_wcs(w, shape)
    mask = np.ones(shape)
    R = coupling_matrix(geo, mask, EDGES, ELL)
    ell = np.maximum(ELL, 1.0)
    sky = {"EE": ell**-2.5 * (1 + 0.3 * np.sin(ell / 300)), "BB": 0.05 * ell**-2.0}
    measured = {"EE": R["EE<-EE"] @ sky["EE"] + R["EE<-BB"] @ sky["BB"],
                "BB": R["BB<-EE"] @ sky["EE"] + R["BB<-BB"] @ sky["BB"]}
    sky_edges = np.arange(1000, 4001, sky_step)
    out = decouple(measured, R, sky, ELL, sky_edges)
    for y in sky:
        np.testing.assert_allclose(out[y], bin_spectrum(sky[y], ELL, sky_edges), rtol=1e-9)
