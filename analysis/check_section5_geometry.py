"""Physical invariants and quadrature convergence for Section 5 geometry.

Run:  bash env.sh python3 analysis/check_section5_geometry.py
          --healpix-masks <dir> --theory-spectra <npz>
"""

import argparse

import numpy as np

import section5_geometry as G5
import spectra as S
import theory as T


def run(masks, theory):
    for g in (.001, .05, .3):
        s1 = np.sqrt((1+g)/(1-g))
        jac = np.diag((s1,1/s1))[None]
        # d(theta_sky)/d(chi_plane)=det(J)/|J.T k|² makes the n=-2
        # quadrupole integrate to zero at every shear.
        np.testing.assert_allclose(G5.anisotropic_response(jac,[1.],n=-2,a2=1),
                                   0., atol=1e-12)
        np.testing.assert_allclose(G5.anisotropic_response(jac,[1.],n=0,a2=1),
                                   g, atol=1e-12)
        np.testing.assert_allclose(G5.anisotropic_response(jac,[1.],n=-3,a2=1,
                                                          chi0=np.pi/4),
                                   0., atol=1e-12)
    direct, mean, low, high = G5.beam_response(np.eye(2)[None],[1.])
    np.testing.assert_allclose([mean,low,high],direct,atol=1e-14)
    ls=np.array([1000.,3000.])
    chi=np.arange(8)*np.pi/8
    same_grid=G5.beam_target_prediction(ls,chi,theory,masks,target='ProjSFL',nside=32)
    sigma=np.deg2rad(1/60)/np.sqrt(8*np.log(2))
    for ns,ds,_,_ in same_grid.values():
        source=ns/ds
        np.testing.assert_allclose(source,np.broadcast_to(np.exp(-(sigma*ls[:,None])**2),source.shape),
                                   atol=1e-12)
    g=.0001
    s1=np.sqrt((1+g)/(1-g))
    jac=np.diag((s1,1/s1))[None]
    # A scale-dependent quadrupole also contributes its radial derivative.
    measured=G5.anisotropic_response(jac,[1.],n=-3,
                                     a2=lambda ell:.3*(ell/1500.)**2)
    np.testing.assert_allclose(measured/g,.15,rtol=1e-6)
    measured=G5.anisotropic_response(jac,[1.],n=-3,a2=.3,attachment='grid')
    np.testing.assert_allclose(measured/g,-.45,rtol=1e-6)
    low = G5.footprint_response(masks,nside=128)
    high = G5.footprint_response(masks,nside=256)
    for key in low:
        a,b=np.array(low[key]),np.array(high[key])
        print(key, 'Nside256', b, 'change from128', b-a)
        np.testing.assert_allclose(a[0],b[0],rtol=.003)
        np.testing.assert_allclose(a[1:],b[1:],rtol=.003,atol=2e-6)
    for face in (0,4):
        a=G5.healpix_face_g2(256,4,face)
        b=G5.healpix_face_g2(512,4,face)
        print('HEALPix face',face,a,b)
        np.testing.assert_allclose(a,b,rtol=.02)
    np.testing.assert_allclose(G5.healpix_face_g2(512,4,0),
                               G5.healpix_face_g2(512,4,8),rtol=1e-10)
    print('Physical invariants and geometry convergence pass.')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    S.add_inputs(ap,'healpix_masks','theory_spectra')
    args=ap.parse_args()
    run(args.healpix_masks,T.load(args.theory_spectra))
