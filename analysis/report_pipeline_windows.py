"""Compute sky-frame window corrections and paired-seed residuals."""
import argparse
from pathlib import Path
import numpy as np
import spectra as S
import theory as T
import shear_leakage as SL
import field_1500d as FD
from pipeline_windows import response


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nside',type=int,default=32)
    p.add_argument('--nchi',type=int,default=64)
    p.add_argument('--aliases',action='store_true')
    p.add_argument('--out',default='/sptlocal/user/cailmdaley/pol_project/pipeline_windows.npz')
    args=p.parse_args()
    th=T.load()
    outputs={'ell':S.ELL,'projs':S.PROJS,'nside':args.nside,'nchi':args.nchi,
             'aliases_enabled':args.aliases}
    # Five samples per original bin. Only a ratio of smooth window integrals
    # is used as a correction; no acoustic structure is approximated away.
    ell=np.arange(5,6000,10,dtype=float)
    for field in (True,False):
        name='field' if field else 'disk5'
        mom=FD.pipeline_moments() if field else SL.load_moments(verbose=False)
        reference={}
        for j,proj in enumerate(S.PROJS):
            R=SL.coupling_matrix(mom,0 if field else 5,j,np.arange(S.LMAX,dtype=float),
                                 th['ell'],band_limit=S.LMAX)
            for spec,block in (('TT','TT<-TT'),('white','TT<-TT'),('EE','EE<-EE'),('BB','BB<-EE')):
                cl=np.where(th['ell']>=2,1.,0.) if spec=='white' else th['unlensed_TT' if spec=='TT' else 'unlensed_EE']
                reference[(proj,spec)]=(R[block]@cl).reshape(S.NBINS,S.DELTA_L).mean(-1)
            del R
        for spec in ('TT','white','EE','BB'):
            cl=th['unlensed_TT' if spec=='TT' else 'unlensed_EE'] if spec!='white' else np.where(th['ell']>=2,1.,0.)
            measured = (S.load_scalar(spec,field=field,correct_windows=False) if spec in ('TT','white') else S.load_field(spec,correct_windows=False) if field else S.load(spec,correct_windows=False))
            if not field: measured=measured[5]
            correction=[]
            for j,proj in enumerate(S.PROJS):
                spin='TT' if spec=='white' else spec
                b,w,a=response(proj,ell,th['ell'],cl,spec=spin,
                               field=field,nside=args.nside,nchi=args.nchi,aliases=args.aliases)
                old_fine,old_bin=S.flat_pixwin(ell),S.flat_pixwin(S.ELL)
                b=b.reshape(120,5).mean(-1)
                w=(w*old_fine).reshape(120,5).mean(-1)/old_bin
                a=(a*old_fine).reshape(120,5).mean(-1)/old_bin
                stg_floor=spec=='BB' and proj=='ProjSTG'
                if stg_floor:
                    # STG has no geometric B. Use the EE normalization for
                    # the BB alias, and EE's window only to display the floor.
                    eb,ew,_=response(proj,ell,th['ell'],cl,spec='EE',field=field,
                                     nside=args.nside,nchi=args.nchi)
                    eb=eb.reshape(120,5).mean(-1)
                    ew=(ew*old_fine).reshape(120,5).mean(-1)/old_bin
                    factor=ew/eb
                    norm=reference[(proj,'EE')]/eb
                else:
                    factor=w/b
                    norm=reference[(proj,spec)]/b
                correction.append(factor)
                # All three terms must share the same R-reference normalization.
                a=a*norm
                b=np.zeros_like(b) if stg_floor else reference[(proj,spec)]
                w=b*factor
                total=w+a
                outputs[f'{name}_{spec}_{proj}_base']=b
                outputs[f'{name}_{spec}_{proj}_windowed']=w
                outputs[f'{name}_{spec}_{proj}_alias']=a
                outputs[f'{name}_{spec}_{proj}_raw']=measured[j]
                ratio=(measured[j]/factor/S.theory('unlensed_EE') if stg_floor
                       else measured[j]/w)
                ratio_total=np.divide(measured[j],total,out=np.full_like(measured[j],np.nan),
                                      where=total>0)
                print(name,spec,proj,'floor/input EE' if stg_floor else 'residual',flush=True)
                for lo,hi in ((1000,2000),(2000,3000),(3000,4000),(4000,4500),(4500,4800),(4800,5400)):
                    sel=(S.ELL>=lo)&(S.ELL<hi)
                    r=ratio[:,sel].mean(-1)
                    rt=ratio_total[:,sel].mean(-1)
                    denom=reference[(proj,'EE')] if stg_floor else b
                    print(f' {lo}-{hi}: window-only {r.mean():.6f} +/- {r.std(ddof=1)/np.sqrt(len(r)):.6f} window+alias {rt.mean():.6f} +/- {rt.std(ddof=1)/np.sqrt(len(rt)):.6f} window {factor[sel].mean():.6f} alias/{"EE" if stg_floor else "base"} {np.mean(a[sel]/denom[sel]):.6f}',flush=True)
                outputs[f'{name}_{spec}_{proj}_ratio']=ratio
                outputs[f'{name}_{spec}_{proj}_ratio_total']=ratio_total
                if spec=='BB' and not stg_floor:
                    floor=measured[S.PROJS.index('ProjSTG')]
                    # Above 4800 STG itself contains aliases, not a shared floor.
                    sub=np.where(S.ELL[None,:]<4800,(measured[j]-floor)/total,np.nan)
                    outputs[f'{name}_{spec}_{proj}_ratio_floor_subtracted']=sub
                    for lo,hi in ((3000,4000),(4000,4500),(4500,4800)):
                        r=sub[:,(S.ELL>=lo)&(S.ELL<hi)].mean(-1)
                        print(f' paired STG subtraction {lo}-{hi}: {r.mean():.6f} +/- {r.std(ddof=1)/np.sqrt(len(r)):.6f}',flush=True)
            outputs[f'{name}_{spec}_correction']=np.array(correction)
        np.savez(args.out,**outputs)
    for name in ('field','disk5'):
        for spec in ('TT','white','EE','BB'):
            for proj in S.PROJS:
                if spec=='BB' and proj=='ProjSTG': continue
                for kind in ('ratio','ratio_total'):
                    r=outputs[f'{name}_{spec}_{proj}_{kind}']-outputs[f'{name}_{spec}_ProjZEA_{kind}']
                    for lo,hi in ((1000,3000),(3000,4000),(4000,4800)):
                        sel=(S.ELL>=lo)&(S.ELL<hi)
                        d=r[:,sel].mean(-1)
                        print(f'paired {name} {spec} {kind} {proj}-ZEA {lo}-{hi}: {d.mean():.6f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.6f}',flush=True)

if __name__=='__main__': main()
