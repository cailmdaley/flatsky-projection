"""Deterministic metric checks and Monte Carlo normalization of the TT QE."""
import numpy as np
from scipy import fft
from lensing_qe import TTQE, metric_strain, shear_eb


def run():
    k,g1,g2=metric_strain(np.array([[[2.,0],[0,2.]],[[1.2,.3],[0,1/1.2]]]))
    np.testing.assert_allclose([k[0],g1[0],g2[0]],[.5,0,0],atol=1e-14)
    angle=.7
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    j=np.array([[[1.1,.2],[.1,.9]]])
    for a,b in zip(metric_strain(j),metric_strain(j@rot)):
        np.testing.assert_allclose(a,b,atol=1e-14)
    # A radial zenithal shear is E-only before an asymmetric window;
    # a linearly varying bending shear has equal pseudo E/B on a round one.
    u=np.linspace(-8,8,256,endpoint=False)
    x,y=np.meshgrid(u,u)
    radius2=x*x+y*y
    radial=np.exp(-radius2/2)
    g1,g2=(x*x-y*y)*radial,2*x*y*radial
    e,b=shear_eb(g1,g2,u[1]-u[0],np.ones_like(x))
    np.testing.assert_allclose(np.sum(abs(b)**2)/np.sum(abs(e)**2),0,atol=1e-12)
    e,b=shear_eb(g1,g2,u[1]-u[0],np.exp(-x*x/8-y*y/2))
    assert np.sum(abs(b)**2)/np.sum(abs(e)**2)>.01
    e,b=shear_eb(y,np.zeros_like(y),u[1]-u[0],radial)
    # Finite-square Fourier quadrature approaches the continuum 1:1 split.
    np.testing.assert_allclose(np.sum(abs(b)**2)/np.sum(abs(e)**2),1,rtol=.02)
    shape=(192,256)
    pixel=np.deg2rad(6/60)
    ell=np.arange(4000)
    cl=1/(ell+50.)**2*np.exp(-(ell/1300.)**2)
    qe=TTQE(shape,pixel,ell,cl,lmin=200,lmax=900)
    # Independently sum the symmetrized f_a f_b response, including wrap,
    # both leg cuts and the finite-grid off-diagonal. This guards factors
    # of two and signs in the FFT-convolution response implementation.
    iy,ix=np.indices(shape)
    for i,j in ((3,4),(9,5),(15,20)):
        other=((i-iy)%shape[0],(j-ix)%shape[1])
        xp,yp=qe.x[other],qe.y[other]
        cp,ip=qe.cl[other],qe.inv[other]
        lx,ly=qe.x[i,j],qe.y[i,j]
        fg=(lx*qe.x+ly*qe.y)*qe.cl+(lx*xp+ly*yp)*cp
        fc=(-ly*qe.x+lx*qe.y)*qe.cl+(-ly*xp+lx*yp)*cp
        direct=np.array([[np.sum(a*b*qe.inv*ip)/(2*qe.area)
                          for b in (fg,fc)] for a in (fg,fc)])
        convolution=np.array([[qe.rgg[i,j],qe.rgc[i,j]],
                              [qe.rgc[i,j],qe.rcc[i,j]]])
        assert np.max(abs(direct-convolution))/np.max(abs(direct))<1e-11
    rng=np.random.default_rng(41)
    kinj,d=qe.injection(rng,bins=(40,90,180),rms=.003)
    cross=[]
    keep=(qe.l>=40)&(qe.l<180)
    for seed in range(64):
        sky=qe.gaussian(rng)
        t=fft.fft2(sky)
        dx=fft.ifft2(1j*qe.x*t).real
        dy=fft.ifft2(1j*qe.y*t).real
        rows=[]
        for a,b in ((d[0],d[1]),(-d[1],d[0])):
            change=a*dx+b*dy
            delta=(qe.reconstruct(sky+change)-qe.reconstruct(sky-change))/2
            rows.append([np.vdot(kinj[keep],v[keep]).real/np.vdot(kinj[keep],kinj[keep]).real for v in delta])
        cross.append(rows)
    cross=np.array(cross)
    mean=cross.mean(axis=0); error=cross.std(axis=0,ddof=1)/np.sqrt(len(cross))
    print('gradient/curl response rows=injection, columns=estimator',mean,'SE',error)
    np.testing.assert_array_less(np.abs(mean-np.eye(2)),5*error+.025)
    print('Metric rotation invariance and exact finite-grid TT normalization pass.')


if __name__=='__main__':
    run()
