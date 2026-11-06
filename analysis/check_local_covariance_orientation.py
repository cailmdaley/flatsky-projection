"""Detect a west/east reflection with an orientation-sensitive exact Wick gate.

Map (x,y) samples the flat-sky chart (x*(1+y/L),y), retaining spatial AND
angular power structure. Direction-independent or spatially uniform power
cannot detect this reflection. Uses ~1GB memory with 96x96 pixels.
"""
import numpy as np
from local_bin_covariance import overlap_kernel


def main():
    n, sigma, ell, width, scale = 96, .5, 1.2, .08, 120.
    y, x = np.meshgrid(np.arange(n)-n//2, np.arange(n)-n//2, indexing='ij')
    mask = np.exp(-(x/(.36*n))**8-(y/(.36*n))**8)
    east, north = x*(1+y/scale), y
    keep = np.flatnonzero(mask.ravel() > 1e-3)
    e, no = east.ravel()[keep], north.ravel()[keep]
    distance = (e[:, None]-e[None, :])**2+(no[:, None]-no[None, :])**2
    covariance = np.exp(-sigma**2*distance/2)
    del distance
    k = 2*np.pi*np.fft.fftfreq(n)
    ky, kx = np.meshgrid(k, k, indexing='ij')
    radius = np.hypot(kx, ky)
    rings = [np.flatnonzero((radius >= ell+(lag-.5)*width) &
                            (radius < ell+(lag+.5)*width)) for lag in (0, 1)]
    modes = np.concatenate(rings)
    nleft, nright = map(len, rings)
    fourier = np.exp(-1j*(np.outer(kx.ravel()[modes], x.ravel()[keep])+
                          np.outer(ky.ravel()[modes], y.ravel()[keep])))*mask.ravel()[keep]
    # Contiguous real arrays enable BLAS rather than strided-view multiplication.
    projected = ((np.ascontiguousarray(fourier.real)@covariance+
                  1j*(np.ascontiguousarray(fourier.imag)@covariance))@fourier.conj().T)
    del covariance
    control = fourier@fourier.conj().T

    def exact(matrix):
        diagonal = np.diag(matrix).real
        vl = 2*np.sum(np.abs(matrix[:nleft, :nleft])**2)/nleft**2
        vr = 2*np.sum(np.abs(matrix[nleft:, nleft:])**2)/nright**2
        cross = 2*np.sum(np.abs(matrix[:nleft, nleft:])**2)/(nleft*nright)
        return vl/diagonal[:nleft].mean()**2, cross/np.sqrt(vl*vr)

    vp, rp = exact(projected)
    vc, rc = exact(control)
    angles = (np.arange(64)+.5)*2*np.pi/64
    ax, off = (1+y/scale)[..., None], (x/scale)[..., None]
    qeast = np.cos(angles)/ax
    qnorth = np.sin(angles)-off*np.cos(angles)/ax
    power = (2*np.pi/sigma**2)*np.exp(-ell**2*(qeast**2+qnorth**2)/(2*sigma**2))/ax

    def predict(p, w):
        sep = np.array([0, width])
        v = overlap_kernel(p, w, 1, angles, width, sep)
        c = overlap_kernel(np.ones_like(p), w, 1, angles, width, sep)
        mean = np.sum(w**2*p.mean(-1))/np.sum(w**2)
        return v[0]/c[0]/mean**2, (v[1]/v[0])/(c[1]/c[0])

    expected = np.array([vp/vc, rp/rc])
    correct = np.array(predict(power, mask))
    mirrored = np.array(predict(power[:, ::-1], mask[:, ::-1]))
    print('Exact variance ratio / correlation-to-control:', expected)
    print('Local:', correct, 'wrong x reflection:', mirrored)
    np.testing.assert_allclose(correct, expected, atol=.008, rtol=0)
    # The negative control must be detected, not merely accidentally accurate.
    assert np.all(np.abs(mirrored-expected) > .012)
    print('Orientation-sensitive exact Wick gate passes')


if __name__ == '__main__':
    main()
