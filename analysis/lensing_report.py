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
    p.add_argument('--prediction',type=Path,help='Squeezed affine metric prediction NPZ')
    p.add_argument('--expected-seeds',type=int,default=20)
    a=p.parse_args()
    result={}; arrays={}; files={}
    for proj in PROJS:
        matches=sorted(a.inputs.rglob(f'qe_{proj}_*.npz'))
        if a.allow_partial:
            matches=[f for f in matches if (f.parent/'.lightcone-manifest.json').exists()]
        if not a.allow_partial:
            assert len(matches)==a.expected_seeds,(proj,len(matches))
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
        spherical=np.array([v['spherical'] for v in d])
        rows=[]
        for lo,hi in zip((2,10,30,60,120),(10,30,60,120,240)):
            keep=(ls>=lo)&(ls<hi)
            powers=[power_error(mf[:,i,keep]/np.sqrt(w4)) for i in range(2)]
            # Keep the spherical estimator noise separate from the extra
            # uncertainty introduced by the finite flat-control ensemble.
            noise=np.sum(spherical[:,:,keep].var(axis=0,ddof=1),axis=-1)/w4
            difference_noise=np.sum(mf[:,:,keep].var(axis=0,ddof=1),axis=-1)/w4
            rows.append({'L':[lo,hi], 'mean_field_power':powers,
                         'spherical_realization_noise_power':noise.tolist(),
                         'difference_realization_noise_power':difference_noise.tolist()})
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
            flat_normalized=[]
            flat_error=[]
            for i in range(2):
                values=[ratio_error(flat[i][:,j],flat[i][:,i]) for j in range(2)]
                flat_normalized.append([v[0] for v in values])
                flat_error.append([v[1] for v in values])
            responses.append({'L':[lo,hi],'response_over_flat':matrix,'SE':error,
                              'flat_response_over_its_diagonal':flat_normalized,
                              'flat_response_SE':flat_error})
        metric=d[0]['metric_W']
        pe,pb=(np.sum(abs(v)**2) for v in metric[1:])
        keep=(ls>=2)&(ls<30)
        result[proj]={'seeds':len(d),
                      'response_scope':'TT legs 300-1200, fixed grid-potential seed 904; SE covers TT skies only, conditional on this injection. Not a chart-only response or cosmological lensing bias.',
                      'response_normalization':'Rows are injected gradient/curl; columns are reconstructed gradient/curl. Each spherical element is divided by the flat diagonal for that injection, not by a full response-matrix inverse. Flat off-diagonals are reported separately.',
                      'metric_rms':d[0]['metric_rms'].tolist(),
                      'metric_pseudo_B_fraction':None if proj=='ProjSTG' else float(pb/(pe+pb)),
                      'mean_field':rows,'response':responses,
                      'low_L_mean_field_power':[power_error(mf[:,i,keep]/np.sqrt(w4)) for i in range(2)]}
        if a.prediction:
            predfile=np.load(a.prediction)
            prediction=predfile[proj]*np.exp(.5j*(xx+yy)*(float(d[0]['pixel_rad'])-float(predfile['pixel_rad'])))
            fits=[]
            for lo,hi in ((2,10),(10,30),(2,30),(30,60)):
                band=(ls>=lo)&(ls<hi)
                estimates=[]
                for i in range(2):
                    if proj=='ProjSTG' and i==1:
                        estimates.append(None)  # model is numerical zero
                        continue
                    model=prediction[i,band]
                    observed=mf[:,i,band]
                    slopes=np.einsum('k,sk->s',model.conj(),observed).real/np.sum(abs(model)**2)
                    avg=observed.mean(axis=0)
                    rho=np.vdot(model,avg).real/np.sqrt(np.vdot(model,model).real*np.vdot(avg,avg).real)
                    estimates.append({'amplitude':float(slopes.mean()),
                                      'SE':float(slopes.std(ddof=1)/np.sqrt(len(slopes))),
                                      'correlation':float(rho)})
                fits.append({'L':[lo,hi],'gradient_curl':estimates})
            result[proj]['squeezed_metric_prediction_fit']=fits
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
