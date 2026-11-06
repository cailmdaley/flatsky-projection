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


def test_cropped_rebin8():
    """Exercise production oversampling and FlattenPol at sampled mask pixels."""
    from spt3g import core, maps
    rng = np.random.default_rng(421)
    sky = [maps.HealpixSkyMap(rng.normal(size=12*32**2),
                              pol_conv=maps.MapPolConv.COSMO) for _ in range(2)]
    for proj in ('ProjZEA', 'ProjSFL', 'ProjSTG'):
        mask, window, _, _ = load_mask(ROOT/'results/highell/field_flatsky_masks', proj, 0)
        stub = mask_stub(proj, window)
        full = maps.FlatSkyMap(3300, 3300, 2*core.G3Units.arcmin,
                              proj=maps.MapProjection.names[proj],
                              delta_center=-56*core.G3Units.deg,
                              pol_conv=maps.MapPolConv.COSMO)
        yy, xx = np.nonzero(mask > 0)
        select = rng.choice(len(yy), 512, replace=False)
        yy, xx = yy[select], xx[select]
        outputs = []
        for template, oy, ox in ((full, window[0].start, window[1].start), (stub, 0, 0)):
            qu = []
            for hmap in sky:
                m = template.clone(False)
                selected = maps.G3SkyMapMask(m)
                for pixel in np.ravel_multi_index((yy+oy, xx+ox), m.shape):
                    selected[int(pixel)] = True
                maps.reproj_map(hmap, m, rebin=8, interp=True, mask=selected)
                qu.append(m)
            maps.flatten_pol(*qu)
            outputs.append(np.array([np.asarray(m)[yy+oy, xx+ox] for m in qu]))
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-8)
        print(proj, 'sampled rebin8 + FlattenPol gate pass',
              float(abs(outputs[0]-outputs[1]).max()))


def test_control_random_streams():
    import pymaster as nmt
    from bin_covariance_controls import control_seed
    fields = [nmt.synfast_flat(256, 256, 1., 1., [np.ones(6001)], [0],
                               seed=control_seed(0, seed))[0] for seed in (0, 1)]
    a, b = [np.fft.rfft2(field) for field in fields]
    overlap = abs(a @ b.conj().T)/np.outer(np.linalg.norm(a, axis=1),
                                          np.linalg.norm(b, axis=1))
    assert overlap.max() < 1-1e-12, 'distinct skies reused Fourier-row random streams'
    print('NaMaster independent-seed gate pass; maximum Fourier-row overlap', overlap.max())


if __name__ == '__main__':
    test_counterexample()
    if '--projection' in sys.argv:
        test_cropped_projection()
    if '--rebin8' in sys.argv:
        test_cropped_rebin8()
    if '--random' in sys.argv:
        test_control_random_streams()
