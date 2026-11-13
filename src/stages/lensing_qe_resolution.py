#!/usr/bin/env python3
"""Resolution control with the SAME continuous injection and flat sky as 4'.

Drawing a new white array at a new resolution changes the injection pattern,
which changes a mask-weighted response on an anisotropic background. Embed
4' Fourier coefficients, with their pixel-centre phase, on the 2' grid.
The spherical harmonic realization already shares its seed across Nside.
"""
import argparse
from pathlib import Path
import os
import sys
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
import numpy as np
from scipy import fft
from lensing_qe import TTQE
import lensing_qe_sims

BASE_SHAPE=(900,1350)
BASE_PIXEL=np.deg2rad(4/60)
BX,BY=np.meshgrid(2*np.pi*fft.fftfreq(BASE_SHAPE[1],BASE_PIXEL),
                  2*np.pi*fft.fftfreq(BASE_SHAPE[0],BASE_PIXEL))
BL=np.sqrt(BX*BX+BY*BY)


def embed(qe,coeff):
    # Nyquist modes of the coarse array are excluded. They cannot reach the
    # <=1200 estimator legs through the <=240 lens; their mask leakage is
    # negligible. Avoiding them keeps the embedding exactly Hermitian.
    iy=np.flatnonzero(abs(BY[:,0])<2600)
    ix=np.flatnonzero(abs(BX[0,:])<2600)
    dy=np.rint(fft.fftfreq(BASE_SHAPE[0])[iy]*BASE_SHAPE[0]).astype(int)%qe.shape[0]
    dx=np.rint(fft.fftfreq(BASE_SHAPE[1])[ix]*BASE_SHAPE[1]).astype(int)%qe.shape[1]
    phase=np.exp(.5j*(BX[iy[:,None],ix[None,:]]+BY[iy[:,None],ix[None,:]])*(qe.pixel_rad-BASE_PIXEL))
    out=np.zeros(qe.shape,dtype=complex)
    out[dy[:,None],dx[None,:]]=coeff[iy[:,None],ix[None,:]]*phase*qe.n/np.prod(BASE_SHAPE)
    return out


def injection(qe,rng,bins=(30,60,120,240),rms=.003):
    white=fft.fft2(rng.normal(size=BASE_SHAPE))
    k=np.zeros(BASE_SHAPE,dtype=complex)
    for lo,hi in zip(bins[:-1],bins[1:]):
        b=white*((BL>=lo)&(BL<hi))
        b*=rms/np.std(fft.ifft2(b).real)
        k+=b
    k=embed(qe,k)
    phi=np.divide(2*k,qe.l2,out=np.zeros_like(k),where=qe.l2>0)
    return k,np.stack([fft.ifft2(1j*qe.x*phi).real,fft.ifft2(1j*qe.y*phi).real])


def gaussian(qe,rng):
    cl=np.interp(BL,THEORY['ell'],THEORY['unlensed_TT'])
    t=fft.fft2(rng.normal(size=BASE_SHAPE))*np.sqrt(cl/BASE_PIXEL**2)
    return fft.ifft2(embed(qe,t)).real


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--theory',type=Path,required=True)
    THEORY=np.load(p.parse_known_args()[0].theory)
    TTQE.injection=injection
    TTQE.gaussian=gaussian
    lensing_qe_sims.main()
