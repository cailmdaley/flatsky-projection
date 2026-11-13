"""Intrinsic metric shear E/B on the 1500d field and a round GRID taper.

These are pseudo E/B: a Fourier decomposition of W gamma, not a unique
E/B decomposition on a cut sky and not a prediction that the QE equals it.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy import fft
from spt3g import core
import geometry as G
from lensing_qe import metric_strain,shear_eb
from paper_style import PROJS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/stages'))
from make_masks import taper
from _decisions import parse_apodization
from _footprints import FIELD,FOOTPRINTS


def geometry(proj,apodization,res_arcmin=8.):
    """Metric convergence and shear on a 60 x 90 deg grid of `res_arcmin`
    pixels, the 1500d field mask with the `apodization` taper (a
    `parse_apodization` dict), and a round 22-degree grid taper."""
    pixel=np.deg2rad(res_arcmin/60)
    shape=tuple(int(round(s*60/res_arcmin)) for s in (60,90))
    y,x=np.meshgrid((np.arange(shape[0])-(shape[0]-1)/2)*pixel,
                    (np.arange(shape[1])-(shape[1]-1)/2)*pixel,indexing='ij')
    fsm=G._flat_map(proj,res_arcmin=res_arcmin)
    ra,dec=fsm.xy_to_angle(core.DoubleVector((31.5-x/pixel).ravel()),
                          core.DoubleVector((31.5+y/pixel).ravel()))
    ra=(np.asarray(ra).reshape(shape)+np.pi)%(2*np.pi)-np.pi
    dec=np.asarray(dec).reshape(shape)
    half,lo,hi=np.deg2rad(FOOTPRINTS[FIELD].box)
    inside=(abs(ra)<half)&(dec>lo)&(dec<hi)
    distance=np.minimum(dec-lo,hi-dec)
    for edge in (-half,half):
        distance=np.minimum(distance,np.arcsin(np.clip(np.cos(dec)*abs(np.sin(ra-edge)),0,1)))
    field=np.where(inside,taper(np.maximum(distance,0),apodization['apotype'],apodization['aposize']),0.)
    dist=np.deg2rad(22)-np.hypot(x,y)
    circular=np.where(dist>0,taper(np.maximum(dist,0),'C2',2.5),0.)
    jac=G.jacobian(proj,np.rad2deg(ra).ravel(),np.rad2deg(dec).ravel())
    k,g1,g2=(v.reshape(shape) for v in metric_strain(jac))
    return pixel,k,g1,g2,field,circular


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--res-arcmin',type=float,default=8.)
    p.add_argument('--apodization',type=parse_apodization,required=True)
    a=p.parse_args()
    result={}; arrays={}
    for proj in PROJS:
        pixel,k,g1,g2,field,circular=geometry(proj,a.apodization,a.res_arcmin)
        rows={}
        for label,w in (('field',field),('round_grid',circular)):
            e,b=shear_eb(g1,g2,pixel,w)
            pe,pb=np.sum(abs(e)**2),np.sum(abs(b)**2)
            rows[label]={'kappa_rms':float(np.sqrt(np.sum(w*w*k*k)/np.sum(w*w))),
                         'gamma_rms':float(np.sqrt(np.sum(w*w*(g1*g1+g2*g2))/np.sum(w*w))),
                         'pseudo_B_fraction':float(pb/(pe+pb))}
            arrays[proj+'_'+label+'_B']=fft.ifft2(b).real
        result[proj]=rows
        arrays[proj+'_gamma1']=field*g1
        arrays[proj+'_gamma2']=field*g2
        arrays[proj+'_mask']=field
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.out/'geometry.npz',**arrays,pixel_rad=pixel)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
