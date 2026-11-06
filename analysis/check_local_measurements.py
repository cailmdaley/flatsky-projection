"""Independent regression gates for the local compact-source estimator."""
import numpy as np
from scipy import fft
import geometry as G
from local_measurements import cutout, profiles, fit, ARCMIN


def run():
    n, res = 128, .2
    pix = res*ARCMIN
    x = (np.arange(n)-n//2)*pix
    y,x = np.meshgrid(x,x,indexing='ij')
    template=profiles(np.hypot(x,y)/ARCMIN,'point')
    power=np.full((n,n),(18*ARCMIN)**2)
    sf=fft.fft2(template)
    norm=np.sum(abs(sf)**2/power)
    sigma=np.sqrt(n*n/(pix*pix*norm))
    rng=np.random.default_rng(71)
    weights=fft.ifft2(sf/power).real*n*n/norm
    amps=np.array([np.sum(weights*rng.normal(size=(n,n))*18/res) for _ in range(2000)])
    assert abs(amps.std(ddof=1)/sigma-1)<.05
    assert abs(amps.mean())<3*sigma/np.sqrt(len(amps))
    for proj in ('ProjSFL','ProjZEA','ProjBICEP'):
        for ra,dec in ((0.,-56.),(45.,-68.),(-45.,-44.)):
            x,y,r=cutout(proj,ra,dec,n,res)
            source=profiles(r,'point')
            template=profiles(np.hypot(x,y)/ARCMIN,'point')
            jac=G.jacobian(proj,[ra],[dec])[0]
            s1,s2=G.singular_values(jac)
            prediction=2*s1*s2/np.sqrt((1+s1*s1)*(1+s2*s2))
            amp,_,pos,_=fit(source,template,power,pix)
            np.testing.assert_allclose(amp,prediction,atol=1e-6,rtol=0)
            assert np.linalg.norm(pos)<.001
            tangent=np.linalg.solve(jac,np.stack([x.ravel(),y.ravel()]))
            metric=profiles(np.linalg.norm(tangent,axis=0).reshape(x.shape)/ARCMIN,'point')
            corrected=fit(source,metric,power,pix)[0]
            np.testing.assert_allclose(corrected,1,atol=1e-6,rtol=0)
    # A known subpixel displacement must be recovered with the right sign.
    displaced=fft.ifft2(fft.fft2(template)*np.exp(-1j*(
        2*np.pi*fft.fftfreq(n,pix)[None,:]*.07*ARCMIN+
        2*np.pi*fft.fftfreq(n,pix)[:,None]*(-.11)*ARCMIN))).real
    _,peak,pos,_=fit(displaced,template,power,pix)
    np.testing.assert_allclose(pos,[.07,-.11],atol=1e-5)
    np.testing.assert_allclose(peak,1,atol=1e-8)
    print('Gaussian magnification/shear identity, metric correction, noise normalization and subpixel position gates pass.')


if __name__=='__main__':
    run()
