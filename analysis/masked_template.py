"""Data-only pure-E template on the apodized 1500d field.

The pure-E multipoles integrate Q/U against derivatives of W exp(-ik.x).
No Q/U outside W>0 enter. They lose ambiguous boundary modes, so applying
C/(C+N) is a simple isotropic Wiener weight, not the optimal cut-sky posterior.
The oracle uses exactly the same noisy Q/U but takes E before masking.
All output B spectra use NaMaster purification. Independent noise splits
allow unbiased cross-spectra, retaining signal and noise sample variance.
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


def run(maps_path, masks_path, seed, out):
    from pathlib import Path
    import tempfile
    import template_subtraction as T
    import pymaster as nmt
    from spt3g import core
    fft.set_workers(4)
    # Grid uses seed zero only for geometry; it is identical for every seed.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        (root/'projected_qu').mkdir()
        (root/'projected_qu'/'qu_ProjSFL_seed0000.g3').symlink_to(
            Path(maps_path).resolve()/f'qu_ProjSFL_seed{seed:04d}.g3')
        (root/'flatsky_masks').symlink_to(Path(masks_path).resolve())
        T.RUN=root
        gr=T.Grid('ProjSFL')
    fr=next(core.G3File(str(Path(maps_path)/f'qu_ProjSFL_seed{seed:04d}.g3')))
    z=(np.asarray(fr['Q'],float)+1j*np.asarray(fr['U'],float))*gr.extra
    del fr
    q,u=z.real.copy(),z.imag.copy(); del z
    w=np.zeros(gr.shape); w[gr.field['box']]=gr.field['mask']
    bins=nmt.NmtBinFlat(np.arange(0,6000,50,dtype=float),
                        np.arange(50,6050,50,dtype=float))
    def field(qq,uu):
        m=gr.field; box=m['box']
        return nmt.NmtFieldFlat(m['lx'],m['ly'],m['mask'],
            [np.ascontiguousarray(qq[box]),np.ascontiguousarray(uu[box])],purify_b=True)
    def cell(a,b):
        return nmt.compute_coupled_cell_flat(a,b,bins)[3]/np.mean(gr.field['mask']**2)
    result={'ell':bins.get_effective_ells(), 'seed':seed,
            'tan2':gr.weighted(gr.tan2,'field'), 'level':5.,
            'trunc':gr.trunc}
    # Noiseless controls isolate the information cost of the edge.
    e,_=gr.eb(q,u)
    wf=gr.c_ee/(gr.c_ee+(5*T.ARCMIN_RAD)**2)
    def subtract(qq,uu,ee):
        tq,tu=gr.qu_of_b(gr.template(ee))
        return field(qq-tq,uu-tu)
    raw=field(q,u)
    result['bb_signal']=cell(raw,raw)
    for name,ee in [('oracle',e),('masked',pure_e(q,gr.usign*u,w,gr.res))]:
        f=subtract(q,u,ee)
        result['bb_signal_'+name]=cell(f,f)
        print(f'seed{seed}: noiseless {name}',flush=True)
    del e,raw,f,ee
    # Each split has Delta_P=5 uK arcmin, isotropic independent Q/U noise.
    split_fields={name:[] for name in ('raw','oracle','masked')}
    isolated={name:[] for name in ('oracle','masked')}
    for split in range(2):
        rng=np.random.default_rng(np.random.SeedSequence([827346,seed,split]))
        sigma=5*T.ARCMIN_RAD/gr.res
        nq=rng.standard_normal(gr.shape)*sigma
        nu=rng.standard_normal(gr.shape)*sigma
        qq,uu=q+nq,u+nu
        split_fields['raw'].append(field(qq,uu))
        ee,_=gr.eb(qq,uu)
        split_fields['oracle'].append(subtract(qq,uu,wf*ee))
        isolated['oracle'].append(subtract(q,u,wf*ee))
        ee=pure_e(qq,gr.usign*uu,w,gr.res)
        split_fields['masked'].append(subtract(qq,uu,wf*ee))
        isolated['masked'].append(subtract(q,u,wf*ee))
        del ee,nq,nu,qq,uu
        print(f'seed{seed}: split{split}',flush=True)
    for name,fields in split_fields.items():
        result['bb_cross_'+name]=cell(*fields)
        result['bb_auto_'+name]=.5*(cell(fields[0],fields[0])+cell(fields[1],fields[1]))
    for name,fields in isolated.items():
        result['bb_isolated_'+name]=.5*(cell(fields[0],fields[0])+cell(fields[1],fields[1]))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez(out/f'masked_template_seed{seed:04d}.npz',**result)
