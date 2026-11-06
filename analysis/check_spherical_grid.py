"""Small regression gates for spherical-grid tests; optional pixell API check.

Run: bash env.sh python3 analysis/check_spherical_grid.py
With pixell 0.32.9 on PYTHONPATH, add --pixell to check its actual pixel window
and spherical synthesis/analysis (pixell is not required by production stages).
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/stages'))
from spherical_grid_test import np, hp, transport, synth, analyse, hp_inverse, car_average, regions

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
# The footprint uses production C2, not the distinct C1 sin(2*pi*x) taper.
ntheta = 384
theta = (np.arange(ntheta)+.5)*np.pi/ntheta
dec = 90-np.degrees(theta)
row = np.flatnonzero((dec < -42)&(dec > -44.5))[0]
distance = np.radians(-42-dec[row])
x = np.sqrt((1-np.cos(distance))/(1-np.cos(np.radians(2.5))))
expected = (.5*(1-np.cos(np.pi*x)))**2*np.sin(theta[row])
np.testing.assert_allclose(regions(ntheta)['field1500'][row, 0], expected, atol=1e-14)
np.random.seed(10)
e = hp.synalm(np.r_[0., 0., np.ones(31)], lmax=32)
a = np.array([e, e*0])
back = analyse(synth(a, 64), 32)
np.testing.assert_allclose(back, a, atol=3e-14)
# Constant chart-longitude averaging has a declination-dependent physical
# width, yet is exactly an m-filter and cannot mix E/B.
m = synth(a, 64)
phi_filter = np.sinc(np.fft.rfftfreq(128))
m = np.fft.irfft(np.fft.rfft(m, axis=-1)*phi_filter, n=128, axis=-1)
filtered = analyse(m, 32)
_, mm = hp.Alm.getlm(32)
np.testing.assert_allclose(filtered, a*np.sinc(mm/128), atol=3e-14)
back = hp_inverse(np.asarray(hp.alm2map_spin(a, 32, 2, 32)), 32)
np.testing.assert_allclose(back, a, atol=1e-12)
# Test the transport sign against a spin-2 spherical eigenfunction, not
# only a second expression for the same geometric angle.
from ducc0 import sht
from scipy.special import eval_jacobi
rng = np.random.default_rng(1)
single = np.zeros_like(a)
ids = hp.Alm.getidx(32, 32, np.arange(33))
single[0, ids] = rng.normal(size=33)+1j*rng.normal(size=33)
single[0, ids[0]] = single[0, ids[0]].real
parent_t = np.arccos(rng.uniform(-.98, .98, 128))
parent_p = rng.uniform(0, 2*np.pi, 128)
radius = .1
bearing = (np.arange(64)+.5)*2*np.pi/64
radial = hp.ang2vec(parent_t, parent_p)
et = np.column_stack([np.cos(parent_t)*np.cos(parent_p), np.cos(parent_t)*np.sin(parent_p), -np.sin(parent_t)])
ep = np.column_stack([-np.sin(parent_p), np.cos(parent_p), np.zeros(128)])
children = np.cos(radius)*radial[:, None]+np.sin(radius)*(np.cos(bearing)[None, :, None]*et[:, None]+np.sin(bearing)[None, :, None]*ep[:, None])
ct, cp = hp.vec2ang(children.reshape(-1, 3))
def at(t, p):
    qu = sht.synthesis_general(alm=single, spin=2, lmax=32,
        loc=np.column_stack([t, p]), epsilon=1e-12, nthreads=2)
    return qu[0]+1j*qu[1]
child_pol = at(ct, cp).reshape(128, 64)
parent_pol = at(parent_t, parent_p)
phase = transport(ct.reshape(128, 64), cp.reshape(128, 64), parent_t[:, None], parent_p[:, None])
d22 = np.cos(radius/2)**4*eval_jacobi(30, 0, 4, np.cos(radius))
np.testing.assert_allclose(np.mean(child_pol*phase, axis=1), d22*parent_pol, atol=2e-11)
# Cross-check the actual healpy and ducc spin conventions.
t, phi = hp.pix2ang(16, np.arange(hp.nside2npix(16)))
hqu = hp.alm2map_spin(single, 16, 2, 32)
np.testing.assert_allclose(hqu[0]+1j*hqu[1], at(t, phi), atol=2e-11)
# A single longitudinal harmonic must retain its phase under centred
# symmetric subpixel averaging; a half-pixel centring error fails this.
mode = np.zeros_like(a)
mode[0, hp.Alm.getidx(32, 12, 7)] = 1
point = synth(mode, 64)
for factor in (2, 4):
    averaged, raw = car_average(mode, 64, factor)
    assert np.isfinite(averaged).all()
    f0, f1 = np.fft.rfft(point[0], axis=-1)[:, 7], np.fft.rfft(raw[0], axis=-1)[:, 7]
    row = np.argmax(abs(f0))
    ratio = f1[row]/f0[row]
    assert ratio.real > 0
    assert abs(ratio.imag) < 1e-12

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
