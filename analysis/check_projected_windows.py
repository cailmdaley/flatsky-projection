"""Monochromatic skies through the actual spt3g projector, with controls.

The source is a unit complex wave in known projected coordinates. On the sky
this is locally a wave with q=J^T k. Finite sample averaging is exact in the
map frame. Sparse HEALPix maps carry only the required source pixels.
"""
import json
import numpy as np
import healpy as hp
from spt3g import core, maps
import geometry as G
import spectra as S
from pipeline_windows import interpolation_power, grid_power, flattened_rotation


def check(proj, ell=4800., angle=.37, n=32, ra0=35., dec0=-60.):
    res=np.deg2rad(2/60)
    out=G._flat_map(proj,n=n)
    x,y=out.angle_to_xy(np.deg2rad(ra0),np.deg2rad(dec0))
    out.x_center += (n-1)/2-x
    out.y_center += (n-1)/2-y
    # All true sample locations in the projector's 8x8 average.
    yy,xx=np.mgrid[:n*8,:n*8]/8+.5/8-.5
    ra,dec=out.xy_to_angle(core.DoubleVector(xx.ravel()),core.DoubleVector(yy.ravel()))
    ra,dec=np.asarray(ra),np.asarray(dec)
    ids=np.unique(hp.get_interp_weights(8192,np.pi/2-dec,ra)[0])
    th,ph=hp.pix2ang(8192,ids)
    sx,sy=out.angle_to_xy(core.DoubleVector(ph),core.DoubleVector(np.pi/2-th))
    kx,ky=ell*np.cos(angle),ell*np.sin(angle)
    source_phase=-kx*np.asarray(sx)*res+ky*np.asarray(sy)*res
    src=[maps.HealpixSkyMap(ids,v,8192,False,False) for v in (np.cos(source_phase),np.sin(source_phase))]
    y,x=np.mgrid[:n,:n]
    phase=-kx*x*res+ky*y*res
    a,d=out.xy_to_angle(core.DoubleVector(x.ravel().astype(float)),core.DoubleVector(y.ravel().astype(float)))
    jac=G.jacobian(proj,np.rad2deg(a),np.rad2deg(d))
    q=np.einsum('pji,j->pi',jac,np.array([kx,ky]))
    expected_interp=np.sqrt(interpolation_power(q[:,0],q[:,1],np.asarray(d))).mean()**2
    rows=[]
    for interp,rebin in ((True,1),(True,8),(False,8)):
        values=[]
        for m in src:
            dest=out.clone(False)
            maps.reproj_map(m,dest,interp=interp,rebin=rebin)
            values.append(np.asarray(dest).copy())
        amp=((values[0]+1j*values[1])*np.exp(-1j*phase)).mean()
        predicted=expected_interp*(float(grid_power(kx,ky,rebin=rebin)) if rebin>1 else 1.)
        row=dict(proj=proj,ra=ra0,dec=dec0,ell=ell,interp=interp,rebin=rebin,power=float(abs(amp)**2),prediction=predicted)
        row['ratio']=row['power']/predicted
        print(row,flush=True)
        rows.append(row)
        if interp: assert abs(row['ratio']-1)<.001
    return rows

def check_spin(proj='ProjSFL', ra0=38., dec0=-62., angle=.37, ell=3000., n=24):
    """Gate the signed COSMO spin angle through real Q/U reprojection."""
    import shear_leakage as SL
    res=np.deg2rad(2/60)
    out=G._flat_map(proj,n=n)
    out.pol_conv=maps.MapPolConv.COSMO
    x,y=out.angle_to_xy(np.deg2rad(ra0),np.deg2rad(dec0))
    out.x_center+=(n-1)/2-x
    out.y_center+=(n-1)/2-y
    yy,xx=np.mgrid[:n*8,:n*8]/8+.5/8-.5
    ra,dec=out.xy_to_angle(core.DoubleVector(xx.ravel()),core.DoubleVector(yy.ravel()))
    ids=np.unique(hp.get_interp_weights(8192,np.pi/2-np.asarray(dec),np.asarray(ra))[0])
    th,ph=hp.pix2ang(8192,ids)
    sx,sy=out.angle_to_xy(core.DoubleVector(ph),core.DoubleVector(np.pi/2-th))
    k=ell*np.array([np.cos(angle),np.sin(angle)])
    phase=(-k[0]*np.asarray(sx)+k[1]*np.asarray(sy))*res
    jac=G.jacobian(proj,np.rad2deg(ph),90-np.rad2deg(th))
    q=np.einsum('pji,j->pi',jac,k)
    aq=np.arctan2(q[:,1],q[:,0])
    yy,xx=np.mgrid[:n,:n]
    demod=np.exp(-1j*(-k[0]*xx+k[1]*yy)*res)
    amps=[]
    for carrier in (np.cos(phase),np.sin(phase)):
        dests=[]
        for stokes,pol in ((np.cos(2*aq),maps.MapPolType.Q),
                           (np.sin(2*aq),maps.MapPolType.U)):
            src=maps.HealpixSkyMap(ids,carrier*stokes,8192,False,False)
            src.pol_conv=maps.MapPolConv.COSMO
            src.pol_type=pol
            dst=out.clone(False)
            dst.pol_type=pol
            maps.reproj_map(src,dst,interp=True,rebin=8)
            dests.append(dst)
        maps.flatten_pol(*dests)
        amps.append([np.asarray(dst).copy() for dst in dests])
    qa=((amps[0][0]+1j*amps[1][0])*demod).mean()
    ua=((amps[0][1]+1j*amps[1][1])*demod).mean()
    # Q and U share the scalar complex wave amplitude.
    measured=.5*np.arctan2((ua/qa).real,1.)
    jc=G.jacobian(proj,[ra0],[dec0])
    qc=jc[0].T@k
    delta=SL.flattening_residual(jc)[0]
    predicted=np.arctan2(qc[1],qc[0])+flattened_rotation(jc)[0]
    # Polarization axes are modulo pi/2 when inferred from U/Q.
    error=(measured-predicted+np.pi/4)%(np.pi/2)-np.pi/4
    row=dict(proj=proj,ra=ra0,dec=dec0,ell=ell,angle=angle,
             spin_error_deg=float(np.rad2deg(error)),delta_deg=float(np.rad2deg(delta)))
    print(row,flush=True)
    assert abs(row['spin_error_deg'])<.01
    return row


if __name__=='__main__':
    rows=[r for point in ((35.,-60.),(-35.,-45.)) for p in S.PROJS
          for r in check(p,ra0=point[0],dec0=point[1])]
    # Reciprocal-grid images and the HEALPix equatorial-belt branch.
    rows += [r for p in S.PROJS for ell,angle in ((5800.,0.),(6500.,1.57),(7000.,.8))
             for r in check(p,ell=ell,angle=angle,n=64)]
    rows += check('ProjZEA',ra0=20.,dec0=-35.,n=64)
    spin=[check_spin(ra0=ra,dec0=dec,angle=angle)
          for ra,dec in ((38.,-62.),(45.,-48.)) for angle in (.37,.8)]
    spin += [check_spin('ProjBICEP')]
    directory='/sptlocal/user/cailmdaley/pol_project/'
    with open(directory+'pipeline_projector_control.json','w') as f:
        json.dump(rows,f,indent=2)
    with open(directory+'pipeline_spin_control.json','w') as f:
        json.dump(spin,f,indent=2)
