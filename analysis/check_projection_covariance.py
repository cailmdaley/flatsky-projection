"""Executable gates for the covariance experiment, under bash env.sh python3."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/stages'))
from bin_covariance_sims import mask_stub, project_array
from namaster_spectra import load_mask
from projection_covariance import counterexample, quadratic_moments


def test_counterexample():
    result = counterexample()
    assert result['permutation_covariance'][0][1] == 0
    np.testing.assert_allclose(result['mixing_covariance'][0][1], 2.25)
    # Independent Gaussian draws must reproduce the Wick covariance.
    rng = np.random.default_rng(5748)
    a = np.array([[1., 2., 0.], [0., 3., 1.]])
    variance = np.array([1., 3., 2.])
    q = np.array([np.diag([1., 0.]), np.diag([0., 1.])])
    mean, cov = quadratic_moments(a, variance, q)
    y = (rng.normal(size=(300000, 3))*np.sqrt(variance)) @ a.T
    values = y**2
    np.testing.assert_allclose(values.mean(0), mean, rtol=.012)
    np.testing.assert_allclose(np.cov(values, rowvar=False), cov, rtol=.025)
    print('counterexample and Wick/Monte Carlo gate pass')


def test_cropped_projection():
    from spt3g import core, maps
    hmap = maps.HealpixSkyMap(np.random.default_rng(42).normal(size=12*32**2),
                              pol_conv=maps.MapPolConv.COSMO)
    for proj in ('ProjZEA', 'ProjSFL', 'ProjSTG'):
        _, window, _, _ = load_mask(ROOT/'results/highell/field_flatsky_masks', proj, 0)
        stub = mask_stub(proj, window)
        # Compare exact coordinate correspondence, then the actual interpolation.
        full = maps.FlatSkyMap(3300, 3300, 2*core.G3Units.arcmin,
                              proj=maps.MapProjection.names[proj],
                              delta_center=-56*core.G3Units.deg,
                              pol_conv=maps.MapPolConv.COSMO)
        rows, cols = window
        for x, y in ((0, 0), (11, 57), (stub.shape[1]-1, stub.shape[0]-1)):
            np.testing.assert_allclose(stub.xy_to_angle(x, y),
                                       full.xy_to_angle(x+cols.start, y+rows.start), atol=1e-14)
        settings = dict(interp=True, rebin=1)
        actual = np.asarray(project_array(hmap, stub, settings))
        expected = np.asarray(project_array(hmap, full, settings))[window]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
        print(proj, 'cropped projection gate pass', float(abs(actual-expected).max()))


if __name__ == '__main__':
    test_counterexample()
    if '--projection' in sys.argv:
        test_cropped_projection()
