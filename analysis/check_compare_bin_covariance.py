"""Small statistical gates for the projected/control covariance comparison."""
import numpy as np
from compare_bin_covariance import (jackknife, lag_statistic, relative_variance,
                                    binwise_lag, binwise_comparison)
from projection_covariance import jackknife_lag


def main():
    rng = np.random.default_rng(20261106)
    n, bands, target = 1000, 16, .2
    covariance = np.eye(bands)+target*(np.eye(bands, k=1)+np.eye(bands, k=-1))
    x = rng.normal(size=(n, bands)) @ np.linalg.cholesky(covariance).T
    y = rng.normal(size=(n, bands))
    ell = np.arange(25, 50*bands, 50)
    # Match the existing slow full-matrix implementation, including its error.
    for lag in (1, 2, 4):
        value, se = jackknife(x, lambda z: lag_statistic(z, lag))
        old = jackknife_lag(x, ell, 0, 50*bands, lag)
        np.testing.assert_allclose([value, se], [old['rho'], old['jackknife_se']], atol=1e-14)
        expected = target if lag == 1 else 0
        assert abs(value-expected) < 4*se, (lag, value, se)
    value, se = jackknife(y, lambda z: lag_statistic(z, 1))
    assert abs(value) < 4*se
    # Multiplying a band by a positive deterministic window changes neither
    # correlation nor fractional variance, as used for the native pseudos.
    window = np.linspace(.3, 1, bands)
    np.testing.assert_allclose(lag_statistic(x*window, 1), lag_statistic(x, 1))
    positive_x, positive_y = x+10, y+10
    np.testing.assert_allclose(relative_variance(positive_x*window, positive_y*window),
                               relative_variance(positive_x, positive_y))
    np.testing.assert_allclose(relative_variance(positive_x, positive_x), 1.)
    # The pointwise diagnostic must agree with scalar jackknifes on each pair.
    keys = ('ProjZEA', 50, 'tt', 'coupled')
    row = binwise_comparison({keys: positive_x[:40]}, {keys: positive_y[:40]})[0]
    selected = positive_x[:40, 10:]  # bin centres >=500, as in the diagnostic
    for i in range(selected.shape[1]-1):
        value, se = jackknife(selected[:, i:i+2], lambda z: lag_statistic(z, 1))
        np.testing.assert_allclose(value, row['projected_rho'][i], atol=1e-14)
        np.testing.assert_allclose(se, row['projected_se'][i], atol=1e-14)
    np.testing.assert_allclose(binwise_lag(x, 1).mean(), lag_statistic(x, 1))
    print('Lag correlation, whole-sky jackknife and window-invariance gates pass')


if __name__ == '__main__':
    main()
