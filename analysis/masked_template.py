"""Data-only pure-E template on the apodized 1500d field.

The pure-E multipoles integrate Q/U against derivatives of W exp(-ik.x).
No Q/U outside W>0 enter. They lose ambiguous boundary modes, so applying
C/(C+N) is a simple isotropic Wiener weight, not the optimal cut-sky posterior.
The oracle uses exactly the same noisy Q/U but takes E before masking.
All output B spectra use NaMaster purification. Independent noise splits
allow noise-unbiased cross-spectra, retaining signal and noise sample variance.
A 0.5-degree E taper shares the 2.5-degree analysis mask's support; using
that wider analysis taper for E is retained as a control, not an edge limit.
"""
import numpy as np
from scipy import fft


def pure_e(q, u, w, res):
    """Pure-E Fourier coefficients; U uses grid-handedness convention.

    Product-rule form of -k^-2 integral (Q D+ + U Dx)[W exp(-ik.x)].
    Zero mode is set to zero. No division by the mask is performed.
    """
    ky = 2*np.pi*fft.fftfreq(w.shape[0])[:, None]/res
    kx = 2*np.pi*fft.fftfreq(w.shape[1])[None, :]/res
    k2 = kx*kx+ky*ky
    inv = np.zeros_like(k2); np.divide(1, k2, out=inv, where=k2>0)
    wf = fft.fft2(w)
    wx = fft.ifft2(1j*kx*wf).real
    wy = fft.ifft2(1j*ky*wf).real
    wplus = fft.ifft2(-(kx*kx-ky*ky)*wf).real
    wcross = fft.ifft2(-2*kx*ky*wf).real
    # Explicitly zero outside the observed support; spectral derivatives of W
    # have tiny ringing there, which must not grant access to unobserved data.
    q = np.where(w>0, q, 0); u = np.where(w>0, u, 0)
    return inv*((kx*kx-ky*ky)*fft.fft2(w*q)+2*kx*ky*fft.fft2(w*u)
                +2j*kx*fft.fft2(wx*q+wy*u)
                +2j*ky*fft.fft2(wx*u-wy*q)
                -fft.fft2(wplus*q+wcross*u))


def run(maps_path, masks_path, theory_path, seed, out, lmax, delta_ell, purify_b,
        noise=5.):
    """Template comparison for one seed of the SFL Q/U leg.

    `maps_path`, `masks_path`: projected_qu and flatsky_masks directories;
    `theory_path`: theory_spectra. Bandpowers of width `delta_ell` to `lmax`;
    `noise` is the white Q/U depth of each split in uK-arcmin.
    """
    from pathlib import Path
    import template_geometry as TG
    import pymaster as nmt
    from spt3g import core
    qu_path=Path(maps_path)/f'qu_ProjSFL_seed{seed:04d}.g3'
    print(f'seed{seed}: constructing template geometry',flush=True)
    gr=TG.Grid('ProjSFL',qu_path,Path(masks_path)/'mask_ProjSFL_disk_5.npz',theory_path)
    print(f'seed{seed}: geometry ready; symbol truncation {gr.trunc:.3g}',flush=True)
    fr=next(core.G3File(str(qu_path)))
    z=(np.asarray(fr['Q'],float)+1j*np.asarray(fr['U'],float))*gr.extra
    q,u=z.real.copy(),z.imag.copy(); del z
    w=np.zeros(gr.shape); w[gr.field['box']]=gr.field['mask']
    yy,xx=np.nonzero(w>0)
    alpha,delta=fr['Q'].xy_to_angle(core.DoubleVector(xx.astype(float)),core.DoubleVector(yy.astype(float)))
    alpha,delta=np.asarray(alpha),np.asarray(delta)
    ra_max,dlo,dhi=TG.FIELD
    ra=np.degrees(TG._wrap(alpha));dec=np.degrees(delta)
    we=np.zeros(gr.shape)
    we[yy,xx]=(TG._c2_taper(np.minimum(dec-dlo,dhi-dec),.5)
               *TG._c2_taper((ra_max-np.abs(ra))*np.cos(delta),.5))
    del fr,yy,xx,alpha,delta,ra,dec
    bins=nmt.NmtBinFlat(np.arange(0,lmax,delta_ell,dtype=float),
                        np.arange(delta_ell,lmax+delta_ell,delta_ell,dtype=float))
    def field(qq,uu):
        m=gr.field; box=m['box']
        return nmt.NmtFieldFlat(m['lx'],m['ly'],m['mask'],
            [np.ascontiguousarray(qq[box]),np.ascontiguousarray(uu[box])],purify_b=purify_b)
    def cell(a,b):
        return nmt.compute_coupled_cell_flat(a,b,bins)[3]/np.mean(gr.field['mask']**2)
    result={'ell':bins.get_effective_ells(), 'seed':seed,
            'tan2':gr.weighted(gr.tan2,'field'), 'level':noise,
            'trunc':gr.trunc, 'template_apod_deg':.5}
    for name,window in [('masked',we),('wide',w)]:
        result['local_taper_cost_'+name]=(gr.weighted(gr.tan2*(1-window[gr.tbox])**2,'field')
                                         /result['tan2'])
    # Noiseless controls isolate this estimator's mask/taper cost.
    wf=gr.c_ee/(gr.c_ee+(noise*TG.ARCMIN_RAD)**2)
    def subtract(qq,uu,ee):
        tq,tu=gr.qu_of_b(TG.TEMPLATE_SIGN*gr.template(ee))
        return field(qq-tq,uu-tu)
    raw=field(q,u)
    result['bb_signal']=cell(raw,raw)
    result['ee_signal']=nmt.compute_coupled_cell_flat(raw,raw,bins)[0]/np.mean(gr.field['mask']**2)
    estimators={'oracle':lambda qq,uu:gr.eb(qq,uu)[0],
                'wide':lambda qq,uu:pure_e(qq,gr.usign*uu,w,gr.res),
                'masked':lambda qq,uu:pure_e(qq,gr.usign*uu,we,gr.res),
                'pseudo':lambda qq,uu:gr.eb(we*qq,we*uu)[0]}
    for name,estimate in estimators.items():
        ee=estimate(q,u)
        f=subtract(q,u,ee)
        result['bb_signal_'+name]=cell(f,f)
        print(f'seed{seed}: noiseless {name}',flush=True)
    del raw,f,ee
    # Each split has Delta_P=noise uK arcmin, isotropic independent Q/U noise.
    split_fields={name:[] for name in ['raw']+list(estimators)}
    isolated={name:[] for name in estimators}
    for split in range(2):
        rng=np.random.default_rng(np.random.SeedSequence([827346,seed,split]))
        sigma=noise*TG.ARCMIN_RAD/gr.res
        nq=rng.standard_normal(gr.shape)*sigma
        nu=rng.standard_normal(gr.shape)*sigma
        qq,uu=q+nq,u+nu
        split_fields['raw'].append(field(qq,uu))
        for name,estimate in estimators.items():
            ee=estimate(qq,uu)
            split_fields[name].append(subtract(qq,uu,wf*ee))
            isolated[name].append(subtract(q,u,wf*ee))
        del ee,nq,nu,qq,uu
        print(f'seed{seed}: split{split}',flush=True)
    for name,fields in split_fields.items():
        result['bb_cross_'+name]=cell(*fields)
        result['bb_auto_'+name]=.5*(cell(fields[0],fields[0])+cell(fields[1],fields[1]))
    result['ee_cross_raw']=nmt.compute_coupled_cell_flat(*split_fields['raw'],bins)[0]/np.mean(gr.field['mask']**2)
    for name,fields in isolated.items():
        result['bb_isolated_'+name]=.5*(cell(fields[0],fields[0])+cell(fields[1],fields[1]))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez(out/f'masked_template_seed{seed:04d}.npz',**result)
