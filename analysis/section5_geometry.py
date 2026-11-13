"""Small reproducible geometry calculations for the paper's Section 5.

Run under ``bash env.sh python3 analysis/section5_geometry.py --healpix-masks DIR``
to print them. The synthetic anisotropic sky is specified in the local
east/north Fourier frame.
"""

import numpy as np
from pathlib import Path
import sys

import geometry as G
import healpix_sheared_window as HW
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'stages'))
import make_masks as MM


def tan_g2(x, y):
    """Gnomonic shear squared for tangent-plane coordinates in radians."""
    r = np.arctan(np.hypot(x, y))
    return np.tan(r / 2) ** 4


def literature_footprints():
    axis = (np.arange(800) + .5) / 800 * 2 - 1
    xx, yy = np.meshgrid(axis, axis)
    # Gatti: a gnomonic square, 14.6 degrees on a side.
    gatti = np.mean(tan_g2(np.tan(np.deg2rad(14.6 / 2)) * xx,
                           np.tan(np.deg2rad(14.6 / 2)) * yy))
    # Pires: a 10 by 10 degree tangent-plane square.
    half = np.tan(np.deg2rad(5))
    pires = np.mean(tan_g2(half * xx, half * yy))
    hp = np.mean(HW.pixel_shear()["g"] ** 2)
    return {"GattiTAN": gatti, "PiresTAN": pires, "HEALPixPlane": hp}


def healpix_face_g2(nside=2048, step=16, face=0):
    """Mean shear of the nested (x,y) chart of one Nside=1 face.

    Central differences of the sky position measure the native face chart's
    metric. Inverting the chart-to-sky Jacobian leaves its shear unchanged.
    The one-pixel boundary is omitted. A full face is an explicit estimate,
    rather than a reconstruction of the mask of Jeffrey et al.'s CNN inputs.
    """
    import healpy as hp
    count = 2*((nside-2)//(2*step))
    half = np.floor(1+(np.arange(count//2)+.5)*(nside-2)/count).astype(int)
    axis = np.concatenate((half,nside-1-half[::-1]))
    x, y = np.meshgrid(axis,axis)
    x, y = x.ravel(), y.ravel()

    def vec(dx, dy):
        ids = hp.xyf2pix(nside, x+dx, y+dy, face, nest=True)
        return np.array(hp.pix2vec(nside, ids, nest=True)).T

    th, ph = hp.vec2ang(vec(0, 0))
    east = np.stack((-np.sin(ph), np.cos(ph), np.zeros_like(ph)), axis=1)
    north = np.stack((-np.cos(ph)*np.cos(th), -np.sin(ph)*np.cos(th),
                      np.sin(th)), axis=1)
    dx, dy = (vec(1,0)-vec(-1,0))/2, (vec(0,1)-vec(0,-1))/2
    jac = np.stack((np.stack(((dx*east).sum(1), (dy*east).sum(1)), axis=1),
                    np.stack(((dx*north).sum(1), (dy*north).sum(1)), axis=1)), axis=1)
    s1, s2 = G.singular_values(jac)
    return np.mean(((s1-s2)/(s1+s2))**2)


def anisotropic_response(jac, weight, n=-3., a2=.3, chi0=0., nchi=256,
                         ell=1500., attachment='sky'):
    """Azimuthally binned response of C(l,chi)=l^n[1+a2 cos 2(chi-chi0)].

    For a plane Fourier vector k, the corresponding local sky vector is
    J.T @ k. Input weights are W² dOmega; the two factors of |det J| below
    account for the local power Jacobian and the plane's area measure.
    SFL and ZEA have unit determinant. ``sky`` describes a local scan/TOD
    quadrupole, while ``grid`` describes a Fourier filter on the output grid.
    """
    chi = (np.arange(nchi) + .5) * 2 * np.pi / nchi
    k = np.stack((np.cos(chi), np.sin(chi)))
    q = np.einsum("...ij,ik->...jk", jac, k)
    theta = np.arctan2(q[:, 1], q[:, 0])
    radius = np.linalg.norm(q, axis=1)
    base = radius**n
    if attachment=='sky':
        amplitude = a2(ell*radius) if callable(a2) else a2
        angle=theta
    else:
        assert attachment=='grid'
        amplitude = a2(ell) if callable(a2) else a2
        angle=chi
    signal = (base * amplitude * np.cos(2 * (angle - chi0))).mean(1)
    isotropic = base.mean(1)
    power_weight = np.asarray(weight)*np.linalg.det(jac)**2
    return (np.average(signal, weights=power_weight)
            / np.average(isotropic, weights=power_weight))


def scan_filter_a2(ell, cutoff=300.):
    """Quadrupole of the toy power filter |ell_x| > cutoff.

    SPT-3G's scan high pass motivates the cutoff scale; the sharp step is an
    illustration rather than a calibrated SPT transfer function.
    """
    theta = np.arccos(cutoff/np.asarray(ell))
    return np.sin(2*theta)/theta


def local_check():
    """Constant equal-area stretch tests the claimed leading coefficient."""
    for g in (.001, .01, .05):
        j = np.diag([np.sqrt((1 + g) / (1 - g)),
                     np.sqrt((1 - g) / (1 + g))])[None]
        exact = anisotropic_response(j, np.ones(1), n=-3., a2=1.)
        np.testing.assert_allclose(exact/g, -.5, atol=.001)
        print(f"local g={g:.3f}: response/a2={exact:.8f}, per g={exact/g:.5f}; "
              f"claimed n/2={-1.5:.5f}")


def footprint_samples(healpix_masks, nside=128):
    """Sky points with the production mask's analytic taper squared.

    The 5% disk and the 1500d field, shaped by `_footprints.FOOTPRINTS` and
    tapered with the apodization recorded in the index files of the
    `healpix_masks` directory, evaluated at a quadrature Nside rather than
    read from the Nside-8192 maps.
    """
    import json
    import healpy as hp
    from _footprints import DEC_DEG, FOOTPRINTS, RA_DEG
    masks = Path(healpix_masks)
    out = {}
    for key, name in (('disk5', 'disk_5'), ('field1500d', 'field_1500d')):
        index = json.loads((masks/f'index_{name}.json').read_text())
        apodization = index['apotype'], index['aposize_deg']
        fp = FOOTPRINTS[name]
        if fp.shape == 'disk':
            centre = hp.ang2vec(RA_DEG, DEC_DEG, lonlat=True)
            disk = MM.apodize_disk_analytic(nside, centre, fp.radius_rad, *apodization)
            ids = np.flatnonzero(disk)
            w = disk[ids]
        else:
            ids, w = MM.rect_mask_analytic(nside, RA_DEG, *fp.box, *apodization)
        ra, dec = hp.pix2ang(nside, ids, lonlat=True)
        out[key] = (ra, dec, w**2)
    ra, dec, w2 = out['field1500d']
    east = (ra > 0) & (ra < 180)
    out['fieldEast'] = (ra[east], dec[east], w2[east])
    return out


def footprint_response(healpix_masks, nside=128, attachment='sky', verbose=False):
    out = {}
    for name,(ra_,dec_,w) in footprint_samples(healpix_masks, nside).items():
        for projection in ("ProjSFL", "ProjZEA"):
            jac = G.jacobian(projection, ra_, dec_)
            s1, s2 = G.singular_values(jac)
            g2 = np.average(((s1-s2)/(s1+s2))**2, weights=w)
            responses = [anisotropic_response(jac, w, chi0=chi0,attachment=attachment)
                         for chi0 in (0., np.pi/4)]
            out[(name, projection)] = (g2, *responses)
            if verbose:
                print(f"{attachment} {name} {projection}: <g²>={g2:.7g}, "
                      f"deltaC/C (a2=.3,n=-3,chi0=0/45)="
                      f"{responses[0]:+.7g}/{responses[1]:+.7g}")
    return out


def beam_response(jac, weight, ell=3000., fwhm_arcmin=1., nchi=180):
    """Power response of a source-grid Gaussian expressed in sky modes."""
    chi = (np.arange(nchi) + .5) * 2*np.pi/nchi
    ksky = np.stack((np.cos(chi), np.sin(chi)))
    inv_t = np.swapaxes(np.linalg.inv(jac), 1, 2)
    kplane = np.einsum("nij,jk->nik", inv_t, ksky)
    sigma = np.deg2rad(fwhm_arcmin/60) / np.sqrt(8*np.log(2))
    response = np.exp(-(sigma*ell)**2 * np.sum(kplane**2, axis=1))
    mean = np.average(response, axis=0, weights=weight)
    direct = np.exp(-(sigma*ell)**2)
    return direct, mean.mean(), mean.min(), mean.max()


def beam_footprints(healpix_masks, nside=128, verbose=False):
    out = {}
    for name,(ra,dec,w) in footprint_samples(healpix_masks, nside).items():
        if name=='fieldEast':
            continue
        for projection in ("ProjSFL", "ProjZEA"):
            vals = beam_response(G.jacobian(projection, ra, dec), w)
            out[(name, projection)] = vals
            if verbose:
                print(f"beam {name} {projection}: sphere={vals[0]:.6f}, "
                      f"azimuth={vals[1]:.6f}, 2D range={vals[2]:.6f}-{vals[3]:.6f}")
    return out


def beam_target_prediction(ell, chi, theory, healpix_masks, target='ProjZEA', nside=128,
                            fwhm_arcmin=1., source_pixel_arcmin=2.):
    """Numerator/denominator integrals in target-plane Fourier modes.

    Sky-area samples carry W² det(J_target)² C(|J_target.T k|). The source
    SFL window includes its top-hat pixel and the subsequent bilinear
    interpolation, whose coherent amplitude is sinc³ per axis. This local
    model omits aliasing, mask convolution, and gradients of the Jacobian.
    `theory` is the dict `theory.load` returns.
    """
    sigma=np.deg2rad(fwhm_arcmin/60)/np.sqrt(8*np.log(2))
    pixel=np.deg2rad(source_pixel_arcmin/60)
    direction=np.stack((np.cos(chi),np.sin(chi)))
    out={}
    for name,(ra,dec,w) in footprint_samples(healpix_masks, nside).items():
        if name=='fieldEast':
            continue
        jt=G.jacobian(target,ra,dec)
        js=G.jacobian('ProjSFL',ra,dec)
        q=np.einsum('nij,ik->njk',jt,direction)
        ks=np.einsum('nji,njk->nik',np.linalg.inv(js),q)
        rt=np.linalg.norm(q,axis=1)
        rs=np.linalg.norm(ks,axis=1)
        weight=w[:,None]*np.linalg.det(jt)[:,None]**2
        ns,ds,nr,dr=[],[],[],[]
        for l in np.atleast_1d(ell):
            c=np.interp(l*rt,theory['ell'],theory['unlensed_EE'])
            norm=weight*c
            source_norm=norm*np.prod(np.sinc(l*pixel*ks/(2*np.pi))**6,axis=1)
            ns.append(np.sum(source_norm*np.exp(-(sigma*l*rs)**2),axis=0))
            ds.append(np.sum(source_norm,axis=0))
            ql=l*rt
            nr.append(np.sum(norm*np.exp(-sigma**2*(ql*(ql+1)-4)),axis=0))
            dr.append(np.sum(norm,axis=0))
        out[name]=tuple(np.array(x) for x in (ns,ds,nr,dr))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--healpix-masks', type=Path, required=True)
    masks = ap.parse_args().healpix_masks
    for name, g2 in literature_footprints().items():
        print(f"{name}: <g²>={g2:.8g}, 2<g²>={2*g2:.8g}")
    g2 = healpix_face_g2()
    print(f"HEALPixFaceNested: <g²>={g2:.8g}, 2<g²>={2*g2:.8g}")
    local_check()
    footprint_response(masks, verbose=True)
    beam_footprints(masks, verbose=True)
