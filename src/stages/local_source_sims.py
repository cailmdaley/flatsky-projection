#!/usr/bin/env python3
"""Spherical CMB plus exact radial sources through the production chart transforms.

Compact objects are evaluated analytically on the sphere, not pixelized on
HEALPix before projection. Only the common CMB uses HEALPix interpolation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
import numpy as np
from scipy import fft
import healpy as hp
from spt3g import core
import geometry as G
from local_measurements import cutout, profiles, fit, ARCMIN


def run(seed, out, nside=2048):
    n, res = 256, .2
    pixel = res*ARCMIN
    theory=np.load('analysis/theory_spectra.npz')
    cl=theory['lensed_TT'][:6001].copy()
    ell=np.arange(len(cl))
    sigma=1.2/np.sqrt(8*np.log(2))*ARCMIN
    cl*=np.exp(-ell*(ell+1)*sigma**2)
    np.random.seed(7700+seed)
    sky=hp.synfast(cl,nside,lmax=len(cl)-1,new=True)
    rng=np.random.default_rng(8800+seed)
    k=2*np.pi*fft.fftfreq(n,pixel)
    ky,kx=np.meshgrid(k,k,indexing='ij')
    ell=np.hypot(kx,ky)
    power=np.interp(ell,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(ell*sigma)**2)
    power+=(18*ARCMIN)**2
    power[ell<300]=np.inf
    rows=[]
    examples={}
    for proj in ('ProjSFL','ProjZEA','ProjBICEP'):
        fsm=G._flat_map(proj,res_arcmin=res)
        for ra in (-45.,-22.5,0.,22.5,45.):
            for dec in (-68.,-56.,-44.):
                x,y,radius=cutout(proj,ra,dec,n,res)
                xc,yc=fsm.angle_to_xy(ra*G.DEG,dec*G.DEG)
                rr,dd=fsm.xy_to_angle(core.DoubleVector((xc-x/pixel).ravel()),
                                     core.DoubleVector((yc+y/pixel).ravel()))
                cmb=hp.get_interp_val(sky,np.pi/2-np.asarray(dd),np.asarray(rr)).reshape(x.shape)
                jac=G.jacobian(proj,[ra],[dec])[0]
                mu=abs(np.linalg.det(jac))
                noise=rng.normal(size=x.shape)*18/res*np.sqrt(mu)
                background=cmb+noise
                for kind, amplitude in (('point',100.),('cluster',300.)):
                    source=profiles(radius,kind)
                    template=profiles(np.hypot(x,y)/ARCMIN,kind)
                    tangent=np.linalg.solve(jac,np.stack([x.ravel(),y.ravel()]))
                    metric=profiles(np.linalg.norm(tangent,axis=0).reshape(x.shape)/ARCMIN,kind)
                    image=amplitude*source+background
                    fixed,peak,pos,_=fit(image,template,power,pixel)
                    corrected=fit(image,metric,power,pixel)[0]
                    blank=fit(background,template,power,pixel)[0]
                    rows.append(dict(projection=proj,ra=ra,dec=dec,profile=kind,
                        input_uK=amplitude,fixed_uK=fixed,peak_uK=peak,
                        blank_uK=blank,metric_uK=corrected,position_arcmin=pos.tolist()))
                    if seed==0 and ra==45. and dec==-68.:
                        examples[proj+'_'+kind+'_source']=source
                        examples[proj+'_'+kind+'_image']=image
        print(proj,seed,'complete',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/f'sources_{seed:03d}.json').write_text(json.dumps(dict(seed=seed,nside=nside,
        res_arcmin=res,n=n,rows=rows),indent=2)+'\n')
    if examples:
        np.savez_compressed(out/'source_examples.npz',**examples)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nside',type=int,default=2048)
    a=p.parse_args()
    run(a.seed,a.out,a.nside)
