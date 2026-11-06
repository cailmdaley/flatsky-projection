"""Shared smooth spherical scalar lenses, with analytic derivatives.

Each potential is a sum of cos(k u.n + phase)/k^2, where u is tangent to
field centre and n is a sky direction. These are globally defined scalar
potentials, not cosmological lens realizations or spherical harmonic bands.
"""
import numpy as np


def frame(ra,dec):
    n=np.stack([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)],axis=-1)
    east=np.stack([-np.sin(ra),np.cos(ra),np.zeros_like(ra)],axis=-1)
    north=np.stack([-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)],axis=-1)
    return n,east,north


def evaluate(ra,dec,modes):
    """Return phi, east/north grad(phi), and -laplacian(phi)/2."""
    n,east,north=frame(ra,dec)
    phi=np.zeros_like(ra); de=phi.copy(); dn=phi.copy(); kappa=phi.copy()
    for k,u,phase,amplitude in modes:
        s=n@u
        z=k*s+phase
        sn,cs=np.sin(z),np.cos(z)
        phi+=amplitude*cs/k**2
        de-=amplitude*sn*(east@u)/k
        dn-=amplitude*sn*(north@u)/k
        kappa+=amplitude*((1-s*s)*cs-2*s*sn/k)/2
    return phi,de,dn,kappa


def potential(seed,rms=.003):
    """Three characteristic-k ranges, each normalized on one common sky patch.

    The fixed RA/Dec quadrature is chart independent, weighted by solid angle.
    Vary the potential with seed, sharing it between all projections.
    """
    rng=np.random.default_rng(9000+seed)
    ra,dec=np.meshgrid(np.deg2rad(np.linspace(-50,50,128)),np.deg2rad(np.linspace(-70,-42,64)))
    weight=np.cos(dec)
    _,e0,n0=frame(np.array(0.),np.deg2rad(np.array(-56.)))
    modes=[]
    for lo,hi in ((30,60),(60,120),(120,240)):
        part=[]
        for _ in range(12):
            theta=rng.uniform(0,2*np.pi)
            part.append((rng.uniform(lo,hi),np.cos(theta)*e0+np.sin(theta)*n0,
                         rng.uniform(0,2*np.pi),1.))
        kap=evaluate(ra,dec,part)[3]
        scale=rms/np.sqrt(np.sum(weight*kap*kap)/np.sum(weight))
        modes.extend((k,u,p,scale) for k,u,p,_ in part)
    return modes


def remap(ra,dec,east_displacement,north_displacement):
    """Exponential map on the unit sphere, not a chart-dependent shift."""
    n,east,north=frame(ra,dec)
    radius=np.hypot(east_displacement,north_displacement)
    tangent=east_displacement[...,None]*east+north_displacement[...,None]*north
    moved=np.cos(radius)[...,None]*n+np.sinc(radius/np.pi)[...,None]*tangent
    return np.arctan2(moved[...,1],moved[...,0]),np.arcsin(np.clip(moved[...,2],-1,1))
