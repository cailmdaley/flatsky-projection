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
    # Integration by parts gives E_pure(k)=k^-2 FFT[W IFFT(k'^2 E(k'))].
    freq=2*np.pi*np.fft.fftfreq(n)
    k2=freq[:,None]**2+freq[None,:]**2
    expected=np.zeros_like(ef)
    np.divide(np.fft.fft2(w*np.fft.ifft2(k2*ef).real),k2,
              out=expected,where=k2>0)
    np.testing.assert_allclose(pure_e(q,u,w,res),expected,atol=1e-9)
    # Compact C2 window and arbitrary edits outside observed data.
    r=np.hypot(x-n/2,y-n/2)/(n/3)
    wc=np.where(r<1,(1-r*r)**3,0)
    q2=q.copy();u2=u.copy();q2[wc==0]=1e9;u2[wc==0]=-1e9
    np.testing.assert_array_equal(pure_e(q,u,wc,res),pure_e(q2,u2,wc,res))
    # Even an ideal local E template changes the response to spherical B.
    angle=np.linspace(-.4,.4,29)
    eg,bg=2*np.cos(angle)-3*np.sin(angle),2*np.sin(angle)+3*np.cos(angle)
    for f in (0.,.5,1.):
        residual=bg-f*np.tan(angle)*eg
        expected=2*np.sin(angle)*(1-f)+3*(np.cos(angle)**2+f*np.sin(angle)**2)/np.cos(angle)
        np.testing.assert_allclose(residual,expected,atol=1e-14)
    print('pure E sign, pure B annihilation, product-rule response, observed support, spherical B response gates pass')
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--oracle-regression',nargs=2,metavar=('NEW_NPZ','OLD_NPZ'))
    a=p.parse_args()
    if a.oracle_regression:
        new,old=(np.load(f) for f in a.oracle_regression)
        seed=int(new['seed'])
        idx=list(old['seeds']).index(seed)
        for source,target in [('field_BB_raw','bb_signal'),('field_BB_sub','bb_signal_oracle')]:
            np.testing.assert_allclose(new[target],old[source][idx]/old['field_w2'],
                                       rtol=2e-10,atol=1e-18)
        print('raw and oracle purified spectra reproduce the original operator test')


if __name__=='__main__':main()
