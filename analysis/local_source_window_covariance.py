"""Nonperiodic stationary-sky covariance of the finite-patch linear filter.

Zero-padding the actual real-space estimator kernel samples its continuous
Fourier transform; it does not change the filter or taper any data.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import fft
import geometry as G
from local_measurements import profiles,ARCMIN


@lru_cache()
def kernel(kind,n=256,res=.2,pad=4):
    pix=res*ARCMIN
    theory=np.load(Path(__file__).with_name('theory_spectra.npz'))
    k=2*np.pi*fft.fftfreq(n,pix)
    ell=np.hypot(k[:,None],k[None,:])
    sig=1.2/np.sqrt(8*np.log(2))*ARCMIN
    power=np.interp(ell,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(ell*sig)**2)+(18*ARCMIN)**2
    power[ell<300]=np.inf
    a=(np.arange(n)-n//2)*pix
    tf=fft.fft2(profiles(np.hypot(a[:,None],a[None,:])/ARCMIN,kind))
    norm=np.sum(abs(tf)**2/power).real
    weights=fft.ifft2(tf/power).real*n*n/norm
    squared=abs(fft.fft2(weights,s=(n*pad,n*pad)))**2
    # Independent Parseval normalization for pixel-white noise.
    np.testing.assert_allclose(squared.sum()/(n*pad)**2,np.sum(weights**2),rtol=1e-12)
    return squared,theory


@lru_cache(maxsize=256)
def covariance(proj,ra,dec,kind,pad=4):
    n,res=256,.2
    pix=res*ARCMIN
    squared,theory=kernel(kind,n,res,pad)
    m=n*pad
    k=2*np.pi*fft.fftfreq(m,pix)
    jac=G.jacobian(proj,[ra],[dec])[0]
    qe=jac[0,0]*k[None,:]+jac[1,0]*k[:,None]
    qn=jac[0,1]*k[None,:]+jac[1,1]*k[:,None]
    q=np.hypot(qe,qn)
    mu=abs(np.linalg.det(jac))
    sig=1.2/np.sqrt(8*np.log(2))*ARCMIN
    cmb=mu*np.interp(q,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(q*sig)**2)
    cmb[q>6000]=0.
    var_cmb=np.sum(squared*cmb)/(m*m*pix*pix)
    var_white=np.sum(squared)*mu*(18*ARCMIN)**2/(m*m*pix*pix)
    return float(var_cmb),float(var_white)


def models(rows):
    """Nonperiodic background forecasts for the actual finite-patch weights."""
    out={}
    for row in rows:
        if row['noise']!='cmb_white':
            continue
        kind=row['profile']
        c,w=covariance(row['projection'],row['ra'],row['dec'],kind)
        c0,w0=covariance('ProjSFL',0.,-56.,kind)
        out[row['projection'],row['ra'],row['dec'],kind]=dict(
            finite_patch_sigma_cmb_uK=np.sqrt(c),finite_patch_sigma_white_uK=np.sqrt(w),
            snr_relative_center_finite_patch=float(row['amplitude']*np.sqrt((c0+w0)/(c+w))))
    return out


if __name__=='__main__':
    import json
    rows=[]
    for proj in ('ProjSFL','ProjZEA','ProjBICEP'):
        for ra,dec in ((0.,-56.),(45.,-68.)):
            for kind in ('point','cluster'):
                for pad in (1,2,4,8):
                    c,w=covariance(proj,ra,dec,kind,pad)
                    rows.append(dict(projection=proj,ra=ra,dec=dec,profile=kind,pad=pad,
                                     cmb_sigma_uK=np.sqrt(c),white_sigma_uK=np.sqrt(w)))
    for proj in ('ProjSFL','ProjZEA','ProjBICEP'):
        for ra in (0.,45.):
            for kind in ('point','cluster'):
                selected={r['pad']:r for r in rows if r['projection']==proj and r['ra']==ra and r['profile']==kind}
                assert abs(selected[8]['cmb_sigma_uK']**2/selected[4]['cmb_sigma_uK']**2-1)<1e-4
    print(json.dumps(rows,indent=2))
