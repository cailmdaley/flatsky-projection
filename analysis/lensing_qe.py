"""Noise-free TT gradient/curl quadratic estimator on a periodic flat grid.

Fourier convention: T(x) = sum_l T_l exp(i l.x)/N; <|T_l|²> = N C_l / a,
where a is the pixel area. Deflections are d=grad(phi)+(-d_y,d_x)Omega.
Both reconstructed fields are L²/2 times their potential, i.e. minus half
its divergence (gradient) or minus half its scalar curl (curl).
The response includes the exact finite Fourier grid, both leg cuts, and the
small gradient/curl mixing caused by that grid. A mask is NOT deconvolved.
"""
import numpy as np
from scipy import fft


class TTQE:
    def __init__(self, shape, pixel_rad, ell, cl, lmin=300, lmax=1200):
        self.shape, self.pixel_rad = tuple(shape), pixel_rad
        self.n = np.prod(shape)
        self.area = self.n * pixel_rad**2
        lx = 2*np.pi*fft.fftfreq(shape[1], pixel_rad)
        ly = 2*np.pi*fft.fftfreq(shape[0], pixel_rad)
        self.x, self.y = np.broadcast_arrays(lx[None, :], ly[:, None])
        self.l2 = self.x**2+self.y**2
        self.l = np.sqrt(self.l2)
        self.cl = np.interp(self.l, ell, cl)
        self.cut = (self.l >= lmin) & (self.l <= lmax)
        inv = np.zeros(shape)
        inv[self.cut] = 1/self.cl[self.cut]
        self.inv = inv
        # R_ab = v_a,i v_b,j [conv(l_i l_j C²/Ct,1/Ct)
        #                       +conv(l_i C/Ct,l_j C/Ct)] / area.
        # Here 1/Ct includes the hard leg cut (zero inverse weight outside).
        moment = []
        for i, j in ((self.x,self.x),(self.x,self.y),(self.y,self.y)):
            a = self.conv(i*j*self.cl**2*inv, inv)
            b = self.conv(i*self.cl*inv, j*self.cl*inv)
            moment.append((a+b)/self.area)
        xx, xy, yy = moment
        self.rgg = self.x**2*xx+2*self.x*self.y*xy+self.y**2*yy
        self.rcc = self.y**2*xx-2*self.x*self.y*xy+self.x**2*yy
        self.rgc = self.x*self.y*(yy-xx)+(self.x**2-self.y**2)*xy
        det = self.rgg*self.rcc-self.rgc**2
        self.valid = (self.l > 0) & (self.l < 2*lmax) & (det > 0)
        self.det = np.where(self.valid,det,1.)

    @staticmethod
    def conv(a,b):
        return fft.ifft2(fft.fft2(a)*fft.fft2(b)).real

    def reconstruct(self, sky):
        t = fft.fft2(sky)
        x = fft.ifft2(t*self.inv).real
        y = t*self.cl*self.inv
        vx = fft.fft2(x*fft.ifft2(1j*self.x*y).real)
        vy = fft.fft2(x*fft.ifft2(1j*self.y*y).real)
        qg = -1j*(self.x*vx+self.y*vy)
        qc = -1j*(-self.y*vx+self.x*vy)
        g = (self.rcc*qg-self.rgc*qc)/self.det
        c = (self.rgg*qc-self.rgc*qg)/self.det
        g = np.where(self.valid,.5*self.l2*g,0.)
        c = np.where(self.valid,.5*self.l2*c,0.)
        return np.stack([g,c])

    def gaussian(self, rng):
        return fft.ifft2(fft.fft2(rng.normal(size=self.shape))*
                         np.sqrt(self.cl/self.pixel_rad**2)).real

    def injection(self, rng, bins=(30,60,120,240), rms=.003):
        """A fixed, band-limited potential, equal kappa RMS in each L bin."""
        white = fft.fft2(rng.normal(size=self.shape))
        k = np.zeros(self.shape,dtype=complex)
        for lo,hi in zip(bins[:-1],bins[1:]):
            b = white*((self.l >= lo) & (self.l < hi))
            b *= rms/np.std(fft.ifft2(b).real)
            k += b
        potential = np.divide(2*k,self.l2,out=np.zeros_like(k),where=self.l2 > 0)
        return k, np.stack([fft.ifft2(1j*self.x*potential).real,
                            fft.ifft2(1j*self.y*potential).real])


def metric_strain(jac):
    """Grid-aligned positive stretch Sigma=sqrt(J J.T), no reference chart.

    A local isotropic sky has P_grid(l)=det(J) C(|J.T l|). This depends on
    J J.T, not on the sky-frame rotation in J and not on P_TAN.
    The returned kappa/gamma are defined by Sigma^-1=(1-kappa)I-gamma.
    """
    b = jac @ np.swapaxes(jac,-1,-2)
    rootdet = np.sqrt(np.linalg.det(b))
    sigma = (b+rootdet[...,None,None]*np.eye(2))/np.sqrt(
        np.trace(b,axis1=-2,axis2=-1)+2*rootdet)[...,None,None]
    inv = np.linalg.inv(sigma)
    return (1-.5*np.trace(inv,axis1=-2,axis2=-1),
            .5*(inv[...,1,1]-inv[...,0,0]),-inv[...,0,1])


def shear_eb(g1,g2,pixel_rad,weight):
    """Pseudo E/B of W*gamma on the actual grid; no pure-mode claim."""
    lx = 2*np.pi*fft.fftfreq(g1.shape[1],pixel_rad)[None,:]
    ly = 2*np.pi*fft.fftfreq(g1.shape[0],pixel_rad)[:,None]
    l2 = lx**2+ly**2
    c = np.divide(lx**2-ly**2,l2,out=np.zeros_like(l2),where=l2>0)
    s = np.divide(2*lx*ly,l2,out=np.zeros_like(l2),where=l2>0)
    a,b=fft.fft2(weight*g1),fft.fft2(weight*g2)
    return c*a+s*b,-s*a+c*b
