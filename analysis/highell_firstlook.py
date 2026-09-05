import sys, glob, numpy as np
sys.path.insert(0,'analysis')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import spectra as S, theory as T, shear_leakage as SL, crawford as CR
import healpy as hp, pymaster as nmt
plt.rcParams['text.usetex']=False
PROJS=S.PROJS; NB=120; ell=None
ee=np.full((2,5,7,20,NB),np.nan); bb=ee.copy()
for j,p in enumerate(PROJS):
    for s in range(20):
        z=np.load(f'results/highell/namaster_spectra/bandpowers_{p}_seed{s:04d}.npz')
        ell=z['ell_eff']; ee[:,j,:,s,:]=z['ee'][:,0,:,0,:]; bb[:,j,:,s,:]=z['bb'][:,0,:,0,:]
# undo the spurious healpix pixwin division (run used healpix_and_flat)
b=nmt.NmtBinFlat(np.arange(0,6000,50).astype(float), np.arange(50,6050,50).astype(float))
pw=hp.pixwin(8192)[:6001]**2; hpc=b.bin_cell(np.arange(6001,dtype=float),pw)
ee*=hpc; bb*=hpc
print('nan count', np.isnan(ee).sum(), np.isnan(bb).sum())
th=T.load(); lth=th['ell'].astype(float); clee=th['unlensed_EE']
full=np.zeros(6001); full[:len(clee)][:6001]=clee[:6001]
clee_b=b.bin_cell(np.arange(6001,dtype=float), full)
m=5  # 5% disk
EE=ee[0,:,m].mean(1); BB=bb[0,:,m].mean(1)   # qu leg
EEeb=ee[1,:,m].mean(1); BBeb=bb[1,:,m].mean(1)
mom=SL.load_moments(verbose=False); print('mom keys', list(mom)[:12])
g2=mom['g2'] if 'g2' in mom else mom['g2_det']
print('g2 shape', g2.shape)
dl=ell*(ell+1)/2/np.pi
fig,ax=plt.subplots(2,2,figsize=(13,9))
col=dict(zip(PROJS,['C3','C0','C2','C1','C4']))
for j,p in enumerate(PROJS):
    ax[0,0].plot(ell, EE[j]/clee_b, color=col[p], label=p)
    ax[0,1].semilogy(ell, dl*BB[j], color=col[p], label=p)
    ax[0,1].semilogy(ell, dl*BBeb[j], color=col[p], ls=':', lw=0.8)
    r=BB[j]/EE[j]; ax[1,0].semilogy(ell, r, color=col[p], label=p)
    ax[1,0].axhline(2*g2[m,j], color=col[p], ls='--', lw=0.8)
    sel=(ell>500)&(ell<3000); sel2=(ell>4000)&(ell<6000)
    print(f'{p:9s} BB/EE 500-3000: {np.mean(r[sel]):.3e}  4000-6000: {np.mean(r[sel2]):.3e}  2<g2>={2*g2[m,j]:.3e}   EE/th @1000,3000,4500,5500: '+' '.join(f'{np.interp(x,ell,EE[j]/clee_b):.3f}' for x in (1000,3000,4500,5500)))
ax[0,0].set(title='EE / input EE, 5% disk, qu leg', xlabel='ell', ylim=(0,3)); ax[0,0].axhline(1,color='k',lw=0.5); ax[0,0].legend()
ax[0,1].set(title='D_l BB [uK^2] (solid qu, dotted eb-leg floor)', xlabel='ell'); 
ax[1,0].set(title='BB/EE with 2<g^2> dashed', xlabel='ell')
# Crawford comparison: SFL/ZEA EE ratio
jS=PROJS.index('ProjSFL'); jZ=PROJS.index('ProjZEA')
ax[1,1].plot(ell, EE[jS]/EE[jZ], 'C3', label='ours: SFL/ZEA EE, 5% disk')
for spec,ls in (('ee','-'),('tt','--')):
    try:
        c0,c5=CR.load(spec); lc=np.arange(60)*100+50
        ax[1,1].plot(lc, c0/c5, 'k', ls=ls, label=f'Crawford proj0/proj5 {spec.upper()}')
    except Exception as e: print('crawford', spec, e)
ax[1,1].set(title='proj0/proj5', xlabel='ell', ylim=(0,3.5)); ax[1,1].legend()
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130); print('saved')
