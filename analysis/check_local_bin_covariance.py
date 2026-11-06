"""Check finite-bin overlap against exact discrete Gaussian Wick pair counts."""
import numpy as np
from local_bin_covariance import overlap_kernel


def exact_pairs(amplitude_squared, mask, ell, width):
    n = len(mask)
    ky, kx = np.meshgrid(2*np.pi*np.fft.fftfreq(n),
                         2*np.pi*np.fft.fftfreq(n), indexing='ij')
    radius = np.hypot(kx, ky)
    rings = [((radius >= ell+(lag-.5)*width) &
              (radius < ell+(lag+.5)*width)).astype(float) for lag in (0, 1, 2)]
    kernel = np.abs(np.fft.fft2(mask**2*amplitude_squared))**2
    # Real maps require both Wick pairings. Rings are centrosymmetric, making
    # them equal; their common factor of two cancels from these ratios.
    values = []
    for ring in rings:
        pairs = np.fft.ifft2(np.conj(np.fft.fft2(rings[0]))*np.fft.fft2(ring)).real
        values.append(np.sum(pairs*kernel)/(rings[0].sum()*ring.sum()))
    diagonal = []
    for ring in rings:
        pairs = np.fft.ifft2(np.abs(np.fft.fft2(ring))**2).real
        diagonal.append(np.sum(pairs*kernel)/ring.sum()**2)
    return values[0], np.array(values[1:])/np.sqrt(diagonal[0]*np.array(diagonal[1:]))


def main():
    n = 256
    y, x = np.meshgrid(np.arange(n)-n/2, np.arange(n)-n/2, indexing='ij')
    mask = np.exp(-((x/80)**8+(y/65)**8))
    power = 1+.8*np.cos(y/10)
    angles = (np.arange(128)+.5)*2*np.pi/128
    one = np.ones(mask.shape+(len(angles),))
    mu = np.sum(mask**2*power)/np.sum(mask**2)
    for width in (.15, .3, .6):
        v = overlap_kernel(power[..., None]*one, mask, 1, angles, width,
                           width*np.arange(3))
        c = overlap_kernel(one, mask, 1, angles, width, width*np.arange(3))
        va, ra = exact_pairs(power, mask, 1.8, width)
        vc, rc = exact_pairs(np.ones_like(power), mask, 1.8, width)
        predicted = v[0]/c[0]/mu**2
        exact = va/vc/mu**2
        print(width, 'variance ratios', predicted, exact,
              'correlations', v[1:]/v[0], ra, 'control', c[1:]/c[0], rc)
        np.testing.assert_allclose(predicted, exact, rtol=.015)
        np.testing.assert_allclose(v[1:]/v[0], ra, atol=.006)
        np.testing.assert_allclose(c[1:]/c[0], rc, atol=.006)
        # Spatially constant power must cancel exactly from the fractional
        # variance ratio and every correlation, for any direction sampling.
        scaled = overlap_kernel(3*one, mask, 1, angles, width, width*np.arange(3))
        np.testing.assert_allclose(scaled, 9*c, rtol=1e-10)
    # The translated triangles partition unity, so summing all radial lags
    # must recover the wide-bin Parseval value, even for finite bin widths.
    width = .3
    separations = width*np.arange(-16, 17)
    v = overlap_kernel(power[..., None]*one, mask, 1, angles, width, separations)
    c = overlap_kernel(one, mask, 1, angles, width, separations)
    wide = np.sum(mask**4*power**2)/np.sum(mask**4)/mu**2
    np.testing.assert_allclose(v.sum()/c.sum()/mu**2, wide, rtol=1e-12)
    # A constant-shear homogeneous plane can have strongly directional power
    # and enlarged diagonal variance without any correlation of disjoint rings.
    angular = 1+.8*np.cos(2*angles)
    uniform = np.ones((16, 16))
    directional = np.broadcast_to(angular, uniform.shape+(len(angles),))
    v = overlap_kernel(directional, uniform, 1, angles, .3, np.array([0, .3, .6]))
    c = overlap_kernel(np.ones_like(directional), uniform, 1, angles, .3,
                       np.array([0, .3, .6]))
    np.testing.assert_allclose(v[0]/c[0], np.mean(angular**2), atol=1e-13)
    np.testing.assert_allclose(v[1:], 0, atol=1e-13)
    print('Finite-bin local Wick checks passed')


if __name__ == '__main__':
    main()
