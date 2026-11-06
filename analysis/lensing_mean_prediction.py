"""Zero-fit squeezed TT-QE mean-field prediction from the intrinsic metric.

Keep the exact nonlinear affine power det(J) C(|J.T l|), then apply the
squeezed TT score and the estimator's exact finite-grid normalization.
This tests the L << ell approximation, not a TAN-relative displacement.
The mask enters as W²; nonlocal filtering at its boundary is approximate.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy import fft
from lensing_qe import TTQE
from lensing_geometry import geometry
from paper_style import PROJS
import geometry as G
from spt3g import core


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    theory=np.load(Path(__file__).parent/'theory_spectra.npz')
    ell,cl=theory['ell'],theory['unlensed_TT']
    pixel=np.deg2rad(4/60)
    qe=TTQE((900,1350),pixel,ell,cl,300,1200)
    iy=np.flatnonzero(abs(qe.y[:,0])<=300)
    ix=np.flatnonzero(abs(qe.x[0,:])<=300)
    lx,ly=np.meshgrid(qe.x[0,ix],qe.y[iy,0])
    ls2=lx*lx+ly*ly
    c=np.divide(lx*lx-ly*ly,ls2,out=np.zeros_like(ls2),where=ls2>0)
    s=np.divide(2*lx*ly,ls2,out=np.zeros_like(ls2),where=ls2>0)
    radial=np.arange(305.,1200.,10.)
    theta=(np.arange(64)+.5)*2*np.pi/64
    dirs=np.stack([np.cos(theta),np.sin(theta)])
    fid=np.interp(radial,ell,cl)
    n=np.interp(radial,ell[2:],np.gradient(np.log(cl[2:]),np.log(ell[2:])))
    quadrature=radial[:,None]*10/(2*np.pi*64)
    score=np.stack([np.broadcast_to(2+n[:,None],(len(radial),len(theta))),
                    n[:,None]*np.cos(2*theta),n[:,None]*np.sin(2*theta)])*.25*quadrature
    arrays={}
    for proj in PROJS:
        dp,k,g1,g2,w,_=geometry(proj,res_arcmin=15.)
        shape=w.shape
        y,x=np.meshgrid((np.arange(shape[0])-(shape[0]-1)/2)*dp,
                        (np.arange(shape[1])-(shape[1]-1)/2)*dp,indexing='ij')
        fsm=G._flat_map(proj,res_arcmin=15.)
        ra,dec=fsm.xy_to_angle(core.DoubleVector((31.5-x/dp).ravel()),
                              core.DoubleVector((31.5+y/dp).ravel()))
        keep=np.flatnonzero(w.ravel()>0)
        jac=G.jacobian(proj,np.rad2deg(np.asarray(ra)[keep]),np.rad2deg(np.asarray(dec)[keep]))
        local=np.zeros((3,w.size))
        for start in range(0,len(keep),128):
            jj=jac[start:start+128]
            # Grid direction is transformed by J.T into sky wavevector.
            stretched=np.einsum('pji,ja->pia',jj,dirs)
            stretch=np.sqrt(np.sum(stretched*stretched,axis=1))
            sampled=np.interp(radial[None,:,None]*stretch[:,None,:],ell,cl)
            ratio=np.linalg.det(jj)[:,None,None]*sampled/fid[None,:,None]-1
            local[:,keep[start:start+128]]=np.einsum('pla,cla->cp',ratio,score)
        local=local.reshape((3,)+shape)*w*w
        # Convert the coarse pixel-centred Fourier phase to the 4' grid's.
        fftlocal=fft.fft2(local,axes=(-2,-1))/w.size
        yind=np.rint(ly[:,0]*shape[0]*dp/(2*np.pi)).astype(int)%shape[0]
        xind=np.rint(lx[0,:]*shape[1]*dp/(2*np.pi)).astype(int)%shape[1]
        coeff=fftlocal[:,yind[:,None],xind[None,:]]*np.exp(-.5j*(lx+ly)*(dp-pixel))
        ag=coeff[0]+c*coeff[1]+s*coeff[2]
        ac=-s*coeff[1]+c*coeff[2]
        rgg=qe.rgg[iy[:,None],ix[None,:]]
        rcc=qe.rcc[iy[:,None],ix[None,:]]
        rgc=qe.rgc[iy[:,None],ix[None,:]]
        det=rgg*rcc-rgc*rgc
        good=(ls2>0)&(det>0)
        predg=np.divide(.5*ls2**2*(rcc*ag-rgc*ac),det,out=np.zeros_like(ag),where=good)
        predc=np.divide(.5*ls2**2*(rgg*ac-rgc*ag),det,out=np.zeros_like(ac),where=good)
        arrays[proj]=np.stack([predg,predc])
        print(proj,'predicted masked RMS below L30',np.sqrt(np.sum(abs(arrays[proj][:,(ls2>0)&(ls2<900)])**2,axis=-1)),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**arrays,lx=qe.x[0,ix],ly=qe.y[iy,0],pixel_rad=pixel)


if __name__=='__main__':
    main()
