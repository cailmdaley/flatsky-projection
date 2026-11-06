"""Paired spherical QU rerun with sky cutoffs 5000 and 6000, one seed.

The 6000 leg is the already materialized production map; the 5000 leg uses
its own alms truncated before synthesis, retaining Nside8192, 2' pixels,
8x interpolation/rebin and FlattenPol. Only field EE/BB bandpowers return.
"""
from pathlib import Path
import argparse
from _decisions import set_threads


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--alms',required=True)
    p.add_argument('--production-maps',required=True)
    p.add_argument('--masks',required=True)
    p.add_argument('--workspaces',required=True)
    p.add_argument('--projection',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    set_threads(4)
    import numpy as np
    import healpy as hp
    import pymaster as nmt
    from spt3g import core, maps
    from _project import project_frame
    from namaster_spectra import (load_mask, finite_in_mask, compute_qu_bandpowers,
                                  load_workspace_manifest, compute_pixel_window)
    almfile=Path(args.alms)/f'alms_seed{args.seed:04d}.fits'
    ae,ab=hp.read_alm(str(almfile),hdu=(2,3))
    lmax=hp.Alm.getlmax(ae.size)
    ell,m=hp.Alm.getlm(5000)
    keep=hp.Alm.getidx(lmax,ell,m)
    ae,ab=ae[keep],ab[keep]
    del ell,m,keep
    q,u=hp.alm2map_spin([ae,ab],8192,2,5000)
    del ae,ab
    frame=core.G3Frame()
    frame['Q']=maps.HealpixSkyMap(q,pol_conv=maps.MapPolConv.COSMO)
    del q
    frame['U']=maps.HealpixSkyMap(u,pol_conv=maps.MapPolConv.COSMO)
    del u
    flat=project_frame(frame,args.projection,res_arcmin=2,dec_deg=-56,
                       size_deg=110,interp=True,rebin=8)
    del frame
    maps.flatten_pol(flat['Q'],flat['U'])
    mask,window,lx,ly=load_mask(Path(args.masks),args.projection,0)
    _,ws,_=load_workspace_manifest(Path(args.workspaces),[args.projection])
    workspace=nmt.NmtWorkspaceFlat()
    workspace.read_from(str(Path(args.workspaces)/ws[args.projection][0]['w22']))
    edges=np.arange(0,6050,50,dtype=float)
    bins=nmt.NmtBinFlat(edges[:-1],edges[1:])
    ell_eff=bins.get_effective_ells()
    pix=compute_pixel_window('flat_only',bins,ell_eff,6000)
    result={}
    for cutoff in (5000,6000):
        if cutoff==6000:
            flat=next(core.G3File(str(Path(args.production_maps)/f'qu_{args.projection}_seed{args.seed:04d}.g3')))
        q=finite_in_mask(np.asarray(flat['Q'],dtype=float)[window],mask)
        u=finite_in_mask(np.asarray(flat['U'],dtype=float)[window],mask)
        ee,bb=compute_qu_bandpowers(q,u,mask,lx,ly,True,bins,workspace)
        result[f'ee_{cutoff}']=ee/pix
        result[f'bb_{cutoff}']=bb/pix
        del flat,q,u
        print(f'{args.projection} seed{args.seed} cutoff{cutoff}: spectra done',flush=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    np.savez(out/f'bandlimit_{args.projection}_seed{args.seed:04d}.npz',
             ell=ell_eff,projection=args.projection,seed=args.seed,**result)


if __name__=='__main__': main()
