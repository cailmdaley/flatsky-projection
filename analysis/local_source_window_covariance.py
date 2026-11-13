"""Nonperiodic stationary-sky covariance of the finite-patch linear filter.

Zero-padding the actual real-space estimator kernel samples its continuous
Fourier transform; it does not change the filter or taper any data. The
filter's cutout (`n` pixels of `res` arcmin), beam and white-noise depth are
those of the local_source_geometry campaign, passed in from its header; the
CMB is the lensed TT of `theory`, band-limited at the sims' `lmax`.
"""
from functools import lru_cache
import numpy as np
from scipy import fft
import geometry as G
from local_measurements import profiles,ARCMIN


@lru_cache()
def kernel(kind,theory,n,res,beam,noise,pad=4):
    pix=res*ARCMIN
    theory=np.load(theory)
    k=2*np.pi*fft.fftfreq(n,pix)
    ell=np.hypot(k[:,None],k[None,:])
    sig=beam/np.sqrt(8*np.log(2))*ARCMIN
    power=np.interp(ell,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(ell*sig)**2)+(noise*ARCMIN)**2
    power[ell<300]=np.inf
    a=(np.arange(n)-n//2)*pix
    tf=fft.fft2(profiles(np.hypot(a[:,None],a[None,:])/ARCMIN,kind,beam))
    norm=np.sum(abs(tf)**2/power).real
    weights=fft.ifft2(tf/power).real*n*n/norm
    squared=abs(fft.fft2(weights,s=(n*pad,n*pad)))**2
    # Independent Parseval normalization for pixel-white noise.
    np.testing.assert_allclose(squared.sum()/(n*pad)**2,np.sum(weights**2),rtol=1e-12)
    return squared,theory


@lru_cache(maxsize=256)
def covariance(proj,ra,dec,kind,theory,n,res,beam,noise,lmax,pad=4):
    pix=res*ARCMIN
    squared,theory=kernel(kind,theory,n,res,beam,noise,pad)
    m=n*pad
    k=2*np.pi*fft.fftfreq(m,pix)
    jac=G.jacobian(proj,[ra],[dec])[0]
    qe=jac[0,0]*k[None,:]+jac[1,0]*k[:,None]
    qn=jac[0,1]*k[None,:]+jac[1,1]*k[:,None]
    q=np.hypot(qe,qn)
    mu=abs(np.linalg.det(jac))
    sig=beam/np.sqrt(8*np.log(2))*ARCMIN
    cmb=mu*np.interp(q,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(q*sig)**2)
    cmb[q>lmax]=0.
    var_cmb=np.sum(squared*cmb)/(m*m*pix*pix)
    var_white=np.sum(squared)*mu*(noise*ARCMIN)**2/(m*m*pix*pix)
    return float(var_cmb),float(var_white)


def models(geometry,theory,lmax):
    """Nonperiodic background forecasts for the actual finite-patch weights.

    `geometry` is local_source_geometry's geometry.json; `theory` the
    theory_spectra path; `lmax` the band limit of the sims' CMB. Each
    position's S/N is relative to the filter at the patch centre.
    """
    setup=(str(theory),geometry['n'],geometry['res_arcmin'],
           geometry['beam_fwhm_arcmin'],geometry['noise_uK_arcmin'],lmax)
    out={}
    for row in geometry['rows']:
        if row['noise']!='cmb_white':
            continue
        kind=row['profile']
        c,w=covariance(row['projection'],row['ra'],row['dec'],kind,*setup)
        c0,w0=covariance('ProjSFL',G.RA_DEG,G.DEC_DEG,kind,*setup)
        out[row['projection'],row['ra'],row['dec'],kind]=dict(
            finite_patch_sigma_cmb_uK=np.sqrt(c),finite_patch_sigma_white_uK=np.sqrt(w),
            snr_relative_center_finite_patch=float(row['amplitude']*np.sqrt((c0+w0)/(c+w))))
    return out

