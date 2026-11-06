#!/usr/bin/env python3
"""TT-QE response to shared gradient/curl potentials on the sphere.

Each seed varies both the isotropic TT sky and the physical lens; all charts
share both. The lens is a smooth analytic scalar on the unit sphere, applied
with its exponential map before projection. This is a response experiment,
not a cosmological lensing-power bias estimate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
import numpy as np
from scipy import fft
import healpy as hp
from spt3g import core
import geometry as G
from lensing_qe import TTQE
from lensing_sky_injection import evaluate,potential,remap
from make_masks import taper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--projection',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--theory',type=Path,default=Path('analysis/theory_spectra.npz'))
    p.add_argument('--res-arcmin',type=float,default=4.)
    p.add_argument('--nside',type=int,default=2048)
    p.add_argument('--sky-lmax',type=int,default=4000)
    p.add_argument('--lmin',type=float,default=300.)
    p.add_argument('--lmax',type=float,default=1200.)
    a=p.parse_args()
    pixel=np.deg2rad(a.res_arcmin/60)
    assert a.lmax<np.pi/pixel,'TT legs must lie below the axis Nyquist frequency'
    shape=tuple(int(round(s*60/a.res_arcmin)) for s in (60,90))
    y,x=np.meshgrid((np.arange(shape[0])-(shape[0]-1)/2)*pixel,
                    (np.arange(shape[1])-(shape[1]-1)/2)*pixel,indexing='ij')
    fsm=G._flat_map(a.projection,res_arcmin=a.res_arcmin)
    ra,dec=fsm.xy_to_angle(core.DoubleVector((31.5-x/pixel).ravel()),
                          core.DoubleVector((31.5+y/pixel).ravel()))
    ra=np.asarray(ra).reshape(shape); dec=np.asarray(dec).reshape(shape)
    ra=(ra+np.pi)%(2*np.pi)-np.pi
    dmin,dmax,half=np.deg2rad([-70,-42,50])
    inside=(abs(ra)<half)&(dec>dmin)&(dec<dmax)
    distance=np.minimum(dec-dmin,dmax-dec)
    for edge in (-half,half):
        distance=np.minimum(distance,np.arcsin(np.clip(np.cos(dec)*abs(np.sin(ra-edge)),0,1)))
    w=np.where(inside,taper(np.maximum(distance,0),'C2',2.5),0.)
    assert not np.any(w[[0,-1],:]) and not np.any(w[:,[0,-1]])
    theory=np.load(a.theory)
    qe=TTQE(shape,pixel,theory['ell'],theory['unlensed_TT'],a.lmin,a.lmax)
    iy=np.flatnonzero(abs(qe.y[:,0])<=300)
    ix=np.flatnonzero(abs(qe.x[0,:])<=300)
    def low(v):
        return v[...,iy[:,None],ix[None,:]]/qe.n
    modes=potential(a.seed)
    _,de,dn,kappa=evaluate(ra,dec,modes)
    jac=G.jacobian(a.projection,np.rad2deg(ra).ravel(),np.rad2deg(dec).ravel()).reshape(shape+(2,2))
    np.random.seed(1000+a.seed)
    sky=hp.synfast(theory['unlensed_TT'][:a.sky_lmax+1],a.nside,lmax=a.sky_lmax,new=True)
    flat=qe.gaussian(np.random.default_rng(2000+a.seed))
    ft=fft.fft2(flat)
    derivatives=np.stack([fft.ifft2(1j*k*ft).real for k in (qe.x,qe.y)],axis=-1)
    result={'lx':qe.x[0,ix],'ly':qe.y[iy,0],
            'input_template':low(fft.fft2(w*w*kappa)),
            'pixel_rad':pixel,'shape':shape,
            'lens_k':np.array([v[0] for v in modes]),
            'lens_direction':np.array([v[1] for v in modes]),
            'lens_phase':np.array([v[2] for v in modes]),
            'lens_amplitude':np.array([v[3] for v in modes])}
    for label,(e,n) in zip(('gradient','curl'),((de,dn),(-dn,de))):
        qs=[]
        for sign in (1,-1):
            r,d=remap(ra,dec,sign*e,sign*n)
            qs.append(qe.reconstruct(w*hp.get_interp_val(sky,np.pi/2-d,r)))
        result[label]=low((qs[0]-qs[1])/2)
        dgrid=np.einsum('...ij,...j->...i',jac,np.stack([e,n],axis=-1))
        perturb=np.sum(dgrid*derivatives,axis=-1)
        result['flat_'+label]=low((qe.reconstruct(w*(flat+perturb))-
                                    qe.reconstruct(w*(flat-perturb)))/2)
    a.out.mkdir(parents=True,exist_ok=True)
    name=f'sky_qe_{a.projection}_{a.seed:04d}'
    np.savez_compressed(a.out/(name+'.npz'),**result)
    (a.out/(name+'.json')).write_text(json.dumps({
        'seed':a.seed,'projection':a.projection,'nside':a.nside,
        'res_arcmin':a.res_arcmin,'sky_lmax':a.sky_lmax,
        'lmin':a.lmin,'lmax':a.lmax,
        'injection':'Shared spherical scalar potential; gradient/curl exponential remapping; potential varies by seed',
        'template':'W^2 times spherical -laplacian(phi)/2, projected as a scalar; not grid convergence',
        'characteristic_wave_number_ranges':[[30,60],[60,120],[120,240]],
        'kappa_rms_per_range_on_common_sky_quadrature':.003},indent=2)+'\n')
    print(a.out/(name+'.npz'),flush=True)


if __name__=='__main__':
    main()
