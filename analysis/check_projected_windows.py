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
from pipeline_windows import interpolation_power, grid_power


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

if __name__=='__main__':
    rows=[r for point in ((35.,-60.),(-35.,-45.)) for p in S.PROJS
          for r in check(p,ra0=point[0],dec0=point[1])]
    with open('/sptlocal/user/cailmdaley/pol_project/pipeline_projector_control.json','w') as f:
        json.dump(rows,f,indent=2)
