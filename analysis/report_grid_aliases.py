"""Local-affine reciprocal-grid prediction, including spin rotation on folding.

This is an analytic band-limit intervention, not a rerun of spherical maps.
The independent real-projector tests are check_projected_windows.py. The grid
sampling theorem predicts copies at 10800 along each map axis; a sky cutoff
maps to an ellipse, not a circle, on a sheared projection.
"""
import numpy as np
import spectra as S
import theory as T
from pipeline_windows import response

if __name__=='__main__':
    th=T.load()
    ell=np.arange(3000.,6001.,100.)
    out={'ell':ell,'projs':S.PROJS}
    for cutoff in (6000,5000):
        for proj in S.PROJS:
            for spec in ('TT','white','EE','BB'):
                cl=np.where(th['ell']>=2,1.,0.) if spec=='white' else th['unlensed_TT' if spec=='TT' else 'unlensed_EE']
                b,w,a=response(proj,ell,th['ell'],cl,spec='TT' if spec=='white' else spec,
                               nside=32,nchi=128,band_limit=cutoff,aliases=True)
                key=f'{cutoff}_{proj}_{spec}'
                out[key+'_base']=b; out[key+'_windowed']=w; out[key+'_alias']=a
                print(key,flush=True)
                for l in (3500,4000,4500,4800,5000,5400,5800):
                    k=np.argmin(abs(ell-l))
                    print(f' {l}: base {b[k]:.8g} window {w[k]:.8g} alias {a[k]:.8g} relative {a[k]/w[k] if w[k]>1e-30 else np.nan:.6g}',flush=True)
            np.savez('/sptlocal/user/cailmdaley/pol_project/pipeline_aliases.npz',**out)
