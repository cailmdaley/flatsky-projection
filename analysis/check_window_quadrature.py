"""Convergence of effective windows, not of the separately tabulated R model."""
import numpy as np
import spectra as S
import theory as T
from pipeline_windows import response

if __name__=='__main__':
    th=T.load(); ell=np.array([1025.,2025.,3025.,4025.,4525.,4775.])
    for field in (True,False):
        for proj in S.PROJS:
            for spec in ('TT','EE','BB'):
                if spec=='BB' and proj=='ProjSTG': continue
                corrections=[]
                for nside,nchi in ((32,64),(64,128),(128,128)):
                    b,w,_=response(proj,ell,th['ell'],th['unlensed_TT' if spec=='TT' else 'unlensed_EE'],
                                   spec=spec,field=field,nside=nside,nchi=nchi)
                    corrections.append(w/b)
                c=np.array(corrections)
                print(field,proj,spec,'32/128-1',c[0]/c[2]-1,'64/128-1',c[1]/c[2]-1,flush=True)
                assert np.max(abs(c[1]/c[2]-1))<4e-4
