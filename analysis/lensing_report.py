"""Summarize projected-sky TT-QE mean fields and signed-injection responses."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import fft
from paper_style import PROJS,PROJ_LABEL,PROJ_COLORS,PROJ_MARKERS


def ratio_error(x,y):
    """Ratio of ensemble means, with a paired leave-one-seed-out SE."""
    value=x.mean()/y.mean()
    jack=(x.sum()-x)/(y.sum()-y)
    error=np.sqrt((len(x)-1)*np.mean((jack-jack.mean())**2))
    return float(value),float(error)


def mean_power(x):
    """Unbiased power of the ensemble mean, summed over selected modes."""
    n=len(x)
    return ((abs(x.sum(axis=0))**2-(abs(x)**2).sum(axis=0))/(n*(n-1))).sum().real


def power_error(x):
    value=mean_power(x)
    jack=np.array([mean_power(np.delete(x,i,axis=0)) for i in range(len(x))])
    return float(value),float(np.sqrt((len(x)-1)*np.mean((jack-jack.mean())**2)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,default=Path('results/highell/lensing_qe_sims'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args()
    result={}; arrays={}; files={}
    for proj in PROJS:
        matches=sorted(a.inputs.rglob(f'qe_{proj}_*.npz'))
        if a.allow_partial:
            matches=[f for f in matches if (f.parent/'.lightcone-manifest.json').exists()]
        if not a.allow_partial:
            assert len(matches)==20,(proj,len(matches))
        if len(matches)<3:
            continue
        d=[np.load(f) for f in matches]
        files[proj]=matches
        xx,yy=np.meshgrid(d[0]['lx'],d[0]['ly'])
        ls=np.hypot(xx,yy)
        kinj=d[0]['input_kappa']
        w4=np.sum(abs(d[0]['mask_w2'])**2)
        # Parseval: stored normalized FFTs give <W^4> on the full canvas.
        # Mask Fourier content above L=300 is negligible for a 2.5-degree taper.
        mf=np.array([v['spherical']-v['flat'] for v in d])
        rows=[]
        for lo,hi in zip((2,10,30,60,120),(10,30,60,120,240)):
            keep=(ls>=lo)&(ls<hi)
            powers=[power_error(mf[:,i,keep]/np.sqrt(w4)) for i in range(2)]
            # Mean-field power / single-realization reconstruction scatter.
            noise=np.sum(mf[:,:,keep].var(axis=0,ddof=1),axis=-1)/w4
            rows.append({'L':[lo,hi], 'mean_field_power':powers,
                         'single_realization_noise_power':noise.tolist()})
        responses=[]
        for lo,hi in ((30,60),(60,120),(120,240)):
            keep=(ls>=lo)&(ls<hi)
            sph=[]; flat=[]
            for label in ('gradient','curl'):
                sph.append(np.array([[np.vdot(kinj[keep],v[label][i][keep]).real for i in range(2)] for v in d]))
                flat.append(np.array([[np.vdot(kinj[keep],v['flat_'+label][i][keep]).real for i in range(2)] for v in d]))
            matrix=[]; error=[]
            for i in range(2):
                row=[]; err=[]
                for j in range(2):
                    val,se=ratio_error(sph[i][:,j],flat[i][:,i])
                    row.append(val); err.append(se)
                matrix.append(row); error.append(err)
            responses.append({'L':[lo,hi],'response_over_flat':matrix,'SE':error})
        metric=d[0]['metric_W']
        pe,pb=(np.sum(abs(v)**2) for v in metric[1:])
        keep=(ls>=2)&(ls<30)
        corr=[]
        for i,g in enumerate((d[0]['metric_W2'][0],d[0]['metric_W2'][2])):
            m=mf.mean(axis=0)[i]
            corr.append(float(np.vdot(g[keep],m[keep]).real/np.sqrt(
                np.vdot(g[keep],g[keep]).real*np.vdot(m[keep],m[keep]).real)))
        result[proj]={'seeds':len(d),'metric_rms':d[0]['metric_rms'].tolist(),
                      'metric_pseudo_B_fraction':float(pb/(pe+pb)),
                      'mean_field':rows,'response':responses,
                      'low_L_metric_correlation_kappa_B':corr}
        arrays[proj+'_mean_field']=mf.mean(axis=0)
        arrays[proj+'_metric']=d[0]['metric_W2']
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.out/'summary.npz',**arrays,lx=d[0]['lx'],ly=d[0]['ly'])
    print(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,
                         'ytick.labelsize':9,'legend.fontsize':9,'text.usetex':False,
                         'figure.autolayout':False,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharex=True)
    for proj,r in result.items():
        x=[np.sqrt(np.prod(row['L'])) for row in r['response']]
        for i,ax in enumerate(axs):
            y=[row['response_over_flat'][i][i] for row in r['response']]
            err=[row['SE'][i][i] for row in r['response']]
            ax.errorbar(x,y,yerr=err,color=PROJ_COLORS[proj],marker=PROJ_MARKERS[proj],label=PROJ_LABEL[proj],capsize=2)
    for ax,label in zip(axs,('Gradient response / flat control','Curl response / flat control')):
        ax.axhline(1,color='.6',lw=1,ls='--')
        ax.set(xlabel='$L$',ylabel=label,xscale='log')
    axs[0].legend(ncol=2,frameon=False)
    fig.tight_layout(); fig.savefig(a.out/'response.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,len(result),figsize=(12,5),sharex=True,sharey=True)
    for col,(proj,r) in enumerate(result.items()):
        m=arrays[proj+'_mean_field'].copy()
        m[:,ls>=30]=0
        maps=fft.ifft2(m,axes=(-2,-1)).real*m.shape[-1]*m.shape[-2]
        for row in range(2):
            ax=axs[row,col]
            im=ax.imshow(maps[row],origin='lower',extent=(-45,45,-30,30),
                         cmap='RdBu_r',vmin=-.04,vmax=.04,aspect='equal')
            ax.text(.04,.94,PROJ_LABEL[proj],transform=ax.transAxes,va='top',fontsize=10,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
            if row==1: ax.set_xlabel('$x$ [deg]')
            if col==0: ax.set_ylabel(('Gradient' if row==0 else 'Curl')+'\n$y$ [deg]')
    fig.subplots_adjust(left=.06,right=.88,bottom=.11,top=.98,wspace=.08,hspace=.12)
    cax=fig.add_axes([.90,.18,.015,.66])
    fig.colorbar(im,cax=cax,label='Reconstruction (convergence units)')
    fig.savefig(a.out/'mean_fields.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
