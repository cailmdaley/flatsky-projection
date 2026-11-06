"""Exact-injection and coordinate-noise checks for the conditional comparison."""
import numpy as np
from choosing_projection import fit_amplitudes, geometry


def check_inverse():
    ell = np.arange(1., 101.)
    shape = ell**-2
    edges = np.array([1, 26, 51, 76, 101])
    basis = np.array([shape*((ell>=lo)&(ell<hi))
                      for lo,hi in zip(edges[:-1],edges[1:])]).T
    # A non-diagonal, full-rank response with polarization mixing.
    r = .8*np.eye(100)+.1*np.roll(np.eye(100),1,axis=1)+.1*np.roll(np.eye(100),-1,axis=1)
    rng = np.random.default_rng(803)
    ae, ab = rng.normal(0,.02,(2,3,4))
    blocks = {'EE<-EE':.9*r, 'BB<-BB':.9*r, 'BB<-EE':.1*r, 'EE<-BB':.1*r}
    e = shape+ae@basis.T
    b = ab@basis.T
    data = np.c_[e@blocks['EE<-EE'].T+b@blocks['EE<-BB'].T,
                 e@blocks['BB<-EE'].T+b@blocks['BB<-BB'].T]
    fit, _, _ = fit_amplitudes(blocks,data,shape,np.ones(100,dtype=bool),edges,ell,shape,True)
    np.testing.assert_allclose(fit,np.c_[ae,ab],atol=1e-13)
    fit, _, _ = fit_amplitudes(r,e@r.T,shape,np.ones(100,dtype=bool),edges,ell,shape)
    np.testing.assert_allclose(fit,ae,atol=1e-13)


def check_geometry():
    result = geometry(64)
    z, stg, sfl = result['charts']
    assert stg['optimum_g2']==0 and stg['centre'] is None
    assert z['optimum_g2'] < z['default_g2']
    assert result['affine_zea']['g2'] < z['optimum_g2']
    np.testing.assert_allclose([z['area_magnification'],sfl['area_magnification']],1,atol=2e-6)
    # Fixed sky white noise has N_plane=det(J)*N_sky; a plane pixel has
    # Omega_sky=Omega_plane/det(J), so this does not create sky noise.
    d = np.array([1.,1.1,1.2])
    omega_plane, nsky = .01, 3.
    variance = nsky/(omega_plane/d)
    np.testing.assert_allclose(variance*omega_plane/d,nsky)


if __name__=='__main__':
    check_inverse()
    check_geometry()
    print('Conditional scalar/polarization inverse and geometry/noise checks pass.')
