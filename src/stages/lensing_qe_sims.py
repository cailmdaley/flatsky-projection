#!/usr/bin/env python3
"""TT-QE mean fields and paired signed gradient/curl remapping responses.

An isotropic Gaussian TT sky is synthesized on the sphere. All four charts
sample that same sky. The controlled deflections are gradients/curls of a
fixed potential on each analysis grid. On the sphere the lens is defined by
P^-1 o (I+d) o P, and evaluated BEFORE projection by sampling the spherical
sky at the displaced direction. These are grid-potential response injections,
not realizations of cosmological full-sky lensing. Their purpose is to test
the normalization/gradient-curl mixing with the SAME known grid input.

No polarization, beam, instrumental noise, or mean-field subtraction learned
from the data is involved. The flat Gaussian control uses the identical mask
and filters. Finite leg cuts and finite-L normalization are exact.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--projection',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nside',type=int,default=2048)
    p.add_argument('--sky-lmax',type=int,default=4000)
    p.add_argument('--res-arcmin',type=float,default=4.)
    p.add_argument('--lmin',type=float,default=300.)
    p.add_argument('--lmax',type=float,default=1200.)
    p.add_argument('--theory',type=Path,default=Path('analysis/theory_spectra.npz'))
    a=p.parse_args()
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
    import numpy as np
    import healpy as hp
    from scipy import fft
    from spt3g import core
    import geometry as G
    from lensing_qe import TTQE,metric_strain,shear_eb
    from make_masks import taper
    pixel=np.deg2rad(a.res_arcmin/60)
    shape=tuple(int(round(s*60/a.res_arcmin)) for s in (60,90))
    y,x=np.meshgrid((np.arange(shape[0])-(shape[0]-1)/2)*pixel,
                    (np.arange(shape[1])-(shape[1]-1)/2)*pixel,indexing='ij')
    fsm=G._flat_map(a.projection,res_arcmin=a.res_arcmin)
    def angles(x,y):
        ra,dec=fsm.xy_to_angle(core.DoubleVector((31.5-x/pixel).ravel()),
                              core.DoubleVector((31.5+y/pixel).ravel()))
        return np.asarray(ra).reshape(shape),np.asarray(dec).reshape(shape)
    ra,dec=angles(x,y)
    ra=(ra+np.pi)%(2*np.pi)-np.pi
    dmin,dmax,half=np.deg2rad([-70,-42,50])
    inside=(np.abs(ra)<half)&(dec>dmin)&(dec<dmax)
    distance=np.minimum(dec-dmin,dmax-dec)
    for edge in (-half,half):
        distance=np.minimum(distance,np.arcsin(np.clip(np.cos(dec)*np.abs(np.sin(ra-edge)),0,1)))
    w=np.where(inside,taper(np.maximum(distance,0),'C2',2.5),0.)
    assert not np.any(w[[0,-1],:]) and not np.any(w[:,[0,-1]])
    theory=np.load(a.theory)
    ell,cl=theory['ell'],theory['unlensed_TT']
    qe=TTQE(shape,pixel,ell,cl,a.lmin,a.lmax)
    # Same random potential in each chart and each seed.
    kinj,displacement=qe.injection(np.random.default_rng(904),rms=.003)
    np.random.seed(1000+a.seed)
    sky=hp.synfast(cl[:a.sky_lmax+1],a.nside,lmax=a.sky_lmax,new=True)
    def sample(dx=0.,dy=0.):
        r,d=angles(x+dx,y+dy)
        return hp.get_interp_val(sky,np.pi/2-d,r)
    # Keep rectangular low-L Fourier subgrid; mean fields are smooth, and
    # storing full high-L reconstructions would waste transfer and disk.
    iy=np.flatnonzero(np.abs(qe.y[:,0])<=300)
    ix=np.flatnonzero(np.abs(qe.x[0,:])<=300)
    def low(v):
        return v[...,iy[:,None],ix[None,:]]/qe.n
    spherical=qe.reconstruct(w*sample())
    flat=qe.gaussian(np.random.default_rng(2000+a.seed))
    flat_q=qe.reconstruct(w*flat)
    flat_t=fft.fft2(flat)
    derivatives=[fft.ifft2(1j*k*flat_t).real for k in (qe.x,qe.y)]
    output={'spherical':low(spherical),'flat':low(flat_q),
            'lx':qe.x[0,ix],'ly':qe.y[iy,0], 'mask_w2':low(fft.fft2(w*w)),
            'input_kappa':low(kinj),'pixel_rad':pixel,'shape':shape}
    for label,(dx,dy) in zip(('gradient','curl'),
                         ((displacement[0],displacement[1]),
                          (-displacement[1],displacement[0]))):
        plus=qe.reconstruct(w*sample(dx,dy))
        minus=qe.reconstruct(w*sample(-dx,-dy))
        output[label]=low((plus-minus)/2)
        perturb=dx*derivatives[0]+dy*derivatives[1]
        output['flat_'+label]=low((qe.reconstruct(w*(flat+perturb))-
                                   qe.reconstruct(w*(flat-perturb)))/2)
    del sky
    # The chart-free positive stretch on the GRID, then its pseudo E/B.
    jac=G.jacobian(a.projection,np.rad2deg(ra).ravel(),np.rad2deg(dec).ravel())
    k,g1,g2=metric_strain(jac)
    k,g1,g2=(v.reshape(shape) for v in (k,g1,g2))
    for label,weight in (('W',w),('W2',w*w)):
        e,b=shear_eb(g1,g2,pixel,weight)
        output['metric_'+label]=low(np.stack([fft.fft2(weight*k),e,b]))
    output['metric_rms']=np.sqrt([np.sum(w*w*v*v)/np.sum(w*w) for v in (k,np.hypot(g1,g2))])
    a.out.mkdir(parents=True,exist_ok=True)
    filename=a.out/f'qe_{a.projection}_{a.seed:04d}.npz'
    np.savez_compressed(filename,**output)
    (a.out/f'index_{a.projection}_{a.seed:04d}.json').write_text(json.dumps(
        {'seed':a.seed,'projection':a.projection,'nside':a.nside,'sky_lmax':a.sky_lmax,
         'res_arcmin':a.res_arcmin,'lmin':a.lmin,'lmax':a.lmax,
         'injection':'grid gradient/curl applied before projection; rms kappa .003 per bin',
         'injection_L_edges':[30,60,120,240], 'file':filename.name},indent=2)+'\n')
    print(filename,flush=True)


if __name__=='__main__':
    main()
