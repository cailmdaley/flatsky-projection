"""Finite-difference and coordinate gates for the shared spherical lens."""
import numpy as np
import geometry as G
from lensing_sky_injection import evaluate,potential,remap


def run():
    rng=np.random.default_rng(12)
    ra=np.deg2rad(rng.uniform(-45,45,40))
    dec=np.deg2rad(rng.uniform(-68,-44,40))
    modes=potential(0)
    phi,de,dn,kap=evaluate(ra,dec,modes)
    h=2e-6
    rp=evaluate(ra+h,dec,modes)[0]; rm=evaluate(ra-h,dec,modes)[0]
    dp=evaluate(ra,dec+h,modes)[0]; dm=evaluate(ra,dec-h,modes)[0]
    np.testing.assert_allclose((rp-rm)/(2*h*np.cos(dec)),de,rtol=1e-6,atol=1e-11)
    np.testing.assert_allclose((dp-dm)/(2*h),dn,rtol=1e-6,atol=1e-11)
    lap=(dp+dm-2*phi)/h**2-np.tan(dec)*(dp-dm)/(2*h)+(rp+rm-2*phi)/(h*np.cos(dec))**2
    np.testing.assert_allclose(-lap/2,kap,rtol=1e-4,atol=1e-7)
    np.testing.assert_allclose(np.array([v[0] for v in modes]),np.array([v[0] for v in potential(0)]))
    assert not np.array_equal(np.array([v[0] for v in modes]),np.array([v[0] for v in potential(1)]))
    for proj in ('ProjSFL','ProjSTG','ProjZEA','ProjBICEP'):
        fsm=G._flat_map(proj)
        jac=G.jacobian(proj,np.rad2deg(ra),np.rad2deg(dec))
        for e,n in ((de,dn),(-dn,de)):
            eps=.01
            r1,d1=remap(ra,dec,eps*e,eps*n)
            r2,d2=remap(ra,dec,-eps*e,-eps*n)
            observed=(G._xy(fsm,np.rad2deg(r1),np.rad2deg(d1))-
                      G._xy(fsm,np.rad2deg(r2),np.rad2deg(d2))).T*np.pi/180/(2*eps)
            expected=np.einsum('...ij,...j->...i',jac,np.stack([e,n],axis=-1))
            np.testing.assert_allclose(observed,expected,rtol=2e-5,atol=1e-10)
    print('Spherical potential derivatives, exponential remapping and all-chart J*d_sky agree.')


if __name__=='__main__':
    run()
