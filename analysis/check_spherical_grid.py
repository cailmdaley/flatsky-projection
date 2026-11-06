"""Small regression gates for spherical-grid tests; optional pixell API check.

Run: bash env.sh python3 analysis/check_spherical_grid.py
With pixell 0.32.9 on PYTHONPATH, add --pixell to check its actual pixel window
and spherical synthesis/analysis (pixell is not required by production stages).
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/stages'))
from spherical_grid_test import np, hp, transport, synth, analyse, hp_inverse, car_average

p = argparse.ArgumentParser()
p.add_argument('--pixell', action='store_true')
args = p.parse_args()
t, phi, t0, p0 = .8, .4, .9, .6
np.testing.assert_allclose(transport(t, phi, t0, phi), 1, atol=1e-15)
np.testing.assert_allclose(transport(t, phi, t0, p0)*transport(t0, p0, t, phi), 1, atol=1e-15)
# Independent bearing formula: angles of the connecting great circle in
# the two local tangent frames determine their relative parallel rotation.
def bearing(t, p, u, q):
    return np.arctan2(np.sin(u)*np.sin(q-p), np.cos(t)*np.sin(u)*np.cos(q-p)-np.sin(t)*np.cos(u))
phase = np.exp(2j*(bearing(t0, p0, t, phi)-bearing(t, phi, t0, p0)))
np.testing.assert_allclose(transport(t, phi, t0, p0), phase, atol=1e-15)
np.random.seed(10)
e = hp.synalm(np.r_[0., 0., np.ones(31)], lmax=32)
a = np.array([e, e*0])
back = analyse(synth(a, 64), 32)
np.testing.assert_allclose(back, a, atol=3e-14)
back = hp_inverse(np.asarray(hp.alm2map_spin(a, 32, 2, 32)), 32)
np.testing.assert_allclose(back, a, atol=1e-12)
# Constant longitude phase must not be confused with a half-pixel shift.
# A narrow equatorial row approaches a separable, real top-hat response.
for factor in (2, 4):
    averaged, raw = car_average(a, 64, factor)
    assert np.isfinite(averaged).all() and np.isfinite(raw).all()

if args.pixell:
    from pixell import enmap, curvedsky
    shape, wcs = enmap.fullsky_geometry(shape=(64, 128))
    m = curvedsky.alm2map(a, enmap.zeros((2,)+shape, wcs), spin=2, nthread=2)
    recovered = curvedsky.map2alm(m, lmax=32, spin=2, nthread=2)
    np.testing.assert_allclose(recovered, a, atol=3e-14)
    wy, wx = enmap.calc_window(shape, order=0)
    np.testing.assert_allclose(wy, np.sinc(np.fft.fftfreq(shape[0])), atol=1e-15)
    np.testing.assert_allclose(wx, np.sinc(np.fft.fftfreq(shape[1])), atol=1e-15)
    # Verify chart Fourier wavevectors give physical widths
    # delta_theta and sin(theta)*delta_phi in a local tangent plane.
    delta = np.pi/shape[0]
    theta, ell, chi = 2.5, 20., .7
    ky, kx = ell*np.cos(chi), ell*np.sin(chi)*np.sin(theta)
    chart = np.sinc(ky*delta/(2*np.pi))*np.sinc(kx*delta/(2*np.pi))
    sky = np.sinc(ell*np.cos(chi)*delta/(2*np.pi))*np.sinc(ell*np.sin(chi)*np.sin(theta)*delta/(2*np.pi))
    np.testing.assert_allclose(chart, sky)
    print('pixell 0.32.9: spherical point roundtrip and chart-sinc window agree')
print('spherical-grid regression gates pass')
