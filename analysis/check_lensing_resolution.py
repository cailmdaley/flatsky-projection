"""Gate that changing map resolution preserves the continuous test inputs."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from scipy import fft
from lensing_qe import TTQE
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/stages'))
import lensing_qe_resolution as R


def grid(shape,pixel):
    x,y=np.meshgrid(2*np.pi*fft.fftfreq(shape[1],pixel),2*np.pi*fft.fftfreq(shape[0],pixel))
    return SimpleNamespace(shape=shape,pixel_rad=pixel,x=x,y=y,l2=x*x+y*y,l=np.sqrt(x*x+y*y),n=np.prod(shape))


def run():
    base=grid(R.BASE_SHAPE,R.BASE_PIXEL)
    fine=grid((1800,2700),R.BASE_PIXEL/2)
    coarse_k,_=TTQE.injection(base,np.random.default_rng(904))
    fine_k,_=R.injection(fine,np.random.default_rng(904))
    iy=np.flatnonzero(abs(base.y[:,0])<=240)
    ix=np.flatnonzero(abs(base.x[0,:])<=240)
    fy=np.rint(fft.fftfreq(base.shape[0])[iy]*base.shape[0]).astype(int)%fine.shape[0]
    fx=np.rint(fft.fftfreq(base.shape[1])[ix]*base.shape[1]).astype(int)%fine.shape[1]
    phase=np.exp(.5j*(base.x[iy[:,None],ix[None,:]]+base.y[iy[:,None],ix[None,:]])*(fine.pixel_rad-base.pixel_rad))
    np.testing.assert_allclose(fine_k[fy[:,None],fx[None,:]]/fine.n,
                               coarse_k[iy[:,None],ix[None,:]]*phase/base.n,rtol=2e-14,atol=1e-18)
    theory=np.load(Path(__file__).parent/'theory_spectra.npz')
    base.cl=np.interp(base.l,theory['ell'],theory['unlensed_TT'])
    flat=TTQE.gaussian(base,np.random.default_rng(2000))
    refined=R.gaussian(fine,np.random.default_rng(2000))
    coarse_t=fft.fft2(flat)/base.n
    fine_t=fft.fft2(refined)/fine.n
    np.testing.assert_allclose(fine_t[fy[:,None],fx[None,:]],
                               coarse_t[iy[:,None],ix[None,:]]*phase,rtol=1e-12,atol=1e-13)
    print('Continuous lens potential and flat-control Fourier phases match across 4/2 arcmin grids.')


if __name__=='__main__':
    run()
