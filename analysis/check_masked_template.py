"""Fast pure-E product-rule, support and sign gates."""
import numpy as np
from masked_template import pure_e


def main():
    n=128; y,x=np.mgrid[:n,:n]; res=1.
    phi=np.arctan2(np.fft.fftfreq(n)[:,None],np.fft.fftfreq(n)[None,:])
    c,s=np.cos(2*phi),np.sin(2*phi)
    # A band-limited periodic window admits a machine-precision product test.
    w=(1-np.cos(2*np.pi*x/n))**3*(1-np.cos(2*np.pi*y/n))**3/64
    e=np.cos(2*np.pi*(11*x+7*y)/n)
    ef=np.fft.fft2(e)
    q=np.fft.ifft2(c*ef).real;u=np.fft.ifft2(s*ef).real
    np.testing.assert_allclose(pure_e(q,u,np.ones_like(w),res),ef,atol=1e-9)
    # A pure B field is annihilated, including the mask derivative terms.
    qb=np.fft.ifft2(-s*ef).real;ub=np.fft.ifft2(c*ef).real
    assert np.max(np.abs(pure_e(qb,ub,w,res)))<1e-9
    # Compact C2 window and arbitrary edits outside observed data.
    r=np.hypot(x-n/2,y-n/2)/(n/3)
    wc=np.where(r<1,(1-r*r)**3,0)
    q2=q.copy();u2=u.copy();q2[wc==0]=1e9;u2[wc==0]=-1e9
    np.testing.assert_array_equal(pure_e(q,u,wc,res),pure_e(q2,u2,wc,res))
    print('pure E sign, pure B annihilation, observed-support gates pass')


if __name__=='__main__':main()
