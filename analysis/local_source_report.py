"""Summarize compact-source injections without treating them as a survey calibration."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJS=('ProjSFL','ProjZEA','ProjBICEP')


def key(r):
    return r['projection'],r['ra'],r['dec'],r['profile']


def report(geometry, sims, out, fine=None, sampling=None):
    data=json.loads(geometry.read_text())
    rows=data['rows']
    assert len(rows)==180,'Incomplete geometry campaign'
    assert sampling is not None,'The paired sampling convergence output is required'
    affine_errors, noise_model=affine_prediction_errors(data)
    from local_source_window_covariance import models
    finite=models(rows)
    for row in rows:
        row.update(noise_model[key(row)+(row['noise'],)])
        if row['noise']=='cmb_white':
            row.update(finite[key(row)])
    from scipy.constants import h, k, c
    nu, tcmb = 150e9, 2.7255
    z=h*nu/(k*tcmb)
    jy_sr_per_K=2*k*nu**2/c**2*z*z*np.exp(z)/np.expm1(z)**2*1e26
    solid_angle=2*np.pi*(1.2/np.sqrt(8*np.log(2))*np.pi/10800)**2
    result={'point_100uK_flux_mJy': float(100e-6*jy_sr_per_K*solid_angle*1e3),
            'scope': 'Single-frequency isotropic filter with 1.2 arcmin beam and 18 uK arcmin white noise; no TOD filtering, foregrounds, template size maximization, selection, or catalogue calibration.',
            'profiles': 'Point Gaussian and beta=1, core=1 arcmin, tapered at 8 arcmin, beam-convolved SZ proxy.',
            'sampling': '15 sky locations, not an area-weighted footprint average.',
            'gaussian_prediction_valid_only_for': {'profile':'point','noise':'white'},
            'geometry': {}, 'spherical_sims': {}}
    for proj in PROJS:
        result['geometry'][proj]={}
        for kind in ('point','cluster'):
            for noise in ('white','cmb_white'):
                rs=[r for r in rows if r['projection']==proj and r['profile']==kind and r['noise']==noise]
                result['geometry'][proj][kind+'_'+noise]={
                    'amplitude_range': [min(r['amplitude'] for r in rs),max(r['amplitude'] for r in rs)],
                    'sigma_uK_range': [min(r['sigma_uK'] for r in rs),max(r['sigma_uK'] for r in rs)],
                    'predicted_snr_relative_center_range': [min(r['snr_relative_center'] for r in rs),max(r['snr_relative_center'] for r in rs)],
                    'finite_patch_snr_relative_center_range': [min(r['snr_relative_center_finite_patch'] for r in rs),max(r['snr_relative_center_finite_patch'] for r in rs)] if noise=='cmb_white' else None,
                    'optimal_snr_relative_center_range': [min(r['optimal_snr_relative_center'] for r in rs),max(r['optimal_snr_relative_center'] for r in rs)],
                    'metric_amplitude_max_error': max(abs(r['metric_template_amplitude']-1) for r in rs),
                    'magnification_only_amplitude_range': [min(r['magnification_template_amplitude'] for r in rs),max(r['magnification_template_amplitude'] for r in rs)] if 'magnification_template_amplitude' in rs[0] else None,
                    'position_bias_max_arcsec': 60*max(np.linalg.norm(r['position_arcmin']) for r in rs),
                    'affine_monte_carlo': next((r['monte_carlo'] for r in rs if r['monte_carlo']),None)}
    gauss=[r for r in rows if r['profile']=='point' and r['noise']=='white']
    result['gaussian_max_absolute_prediction_error']=max(abs(r['amplitude']-r['gaussian_prediction']) for r in gauss)
    result['affine_prediction_errors']=affine_errors
    if fine:
        refined={key(r)+(r['noise'],):r for r in json.loads(fine.read_text())['rows']}
        result['refinement_max_amplitude_difference']=max(abs(r['amplitude']-refined[key(r)+(r['noise'],)]['amplitude']) for r in rows)
    paths=sorted(sims.glob('sources_*.json'))
    result['spherical_seed_count']=len(paths)
    all_sims=[json.loads(p.read_text()) for p in paths]
    assert sorted(d['seed'] for d in all_sims)==list(range(64)),'Expected all 64 spherical CMB seeds'
    for proj in PROJS:
        result['spherical_sims'][proj]={}
        for kind in ('point','cluster'):
            means=[]
            for dec in (-68.,-56.,-44.):
                for ra in (-45.,-22.5,0.,22.5,45.):
                    rs=[r for d in all_sims for r in d['rows'] if key(r)==(proj,ra,dec,kind)]
                    if not rs:
                        continue
                    fixed=np.array([r['fixed_uK']/r['input_uK'] for r in rs])
                    peak=np.array([r['peak_uK']/r['input_uK'] for r in rs])
                    geom=next(r for r in rows if key(r)==(proj,ra,dec,kind) and r['noise']=='cmb_white')
                    paired=np.array([(r['fixed_uK']-r['blank_uK'])/r['input_uK'] for r in rs])
                    assert np.max(abs(paired-geom['amplitude']))<1e-10
                    means.append(dict(ra=ra,dec=dec,amplitude_mean=float(fixed.mean()),
                        amplitude_sem=float(fixed.std(ddof=1)/np.sqrt(len(rs))),
                        paired_response=float(paired.mean()),peak_mean=float(peak.mean()),
                        peak_minus_fixed_mean=float((peak-fixed).mean()),
                        cmb_blank_std_uK=float(np.std([r['cmb_blank_uK'] for r in rs],ddof=1)) if 'cmb_blank_uK' in rs[0] else None,
                        white_blank_std_uK=float(np.std([r['white_blank_uK'] for r in rs],ddof=1)) if 'white_blank_uK' in rs[0] else None,
                        predicted_cmb_sigma_uK=geom['finite_patch_sigma_cmb_uK'],
                        predicted_white_sigma_uK=geom['sigma_white_uK'],
                        position_mean_arcmin=np.mean([r['position_arcmin'] for r in rs],axis=0).tolist(),
                        position_std_arcmin=np.std([r['position_arcmin'] for r in rs],axis=0,ddof=1).tolist()))
            result['spherical_sims'][proj][kind]=means
    result['noise_validation']=noise_validation(all_sims,rows)
    result['sampling_resolution']=sampling_report(all_sims,rows,sampling) if sampling else {}
    example=sampling/'examples_8192.npz' if sampling and (sampling/'examples_8192.npz').exists() else sims/'source_examples.npz'
    result['display_background_nside']=8192 if example.name=='examples_8192.npz' else 2048
    out.mkdir(parents=True,exist_ok=True)
    (out/'local_source_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(rows,out)
    plot(rows,out,'snr_relative_center_finite_patch')
    if example.exists():
        plot_examples(example,out)
    print(json.dumps(result,indent=2))


def sampling_report(base,geometry,directory):
    files=sorted(directory.glob('resolution_*.json'))
    data=[json.loads(p.read_text()) for p in files]
    assert set(d['nside'] for d in data)=={4096,8192},'Missing paired sampling resolution output'
    coarse={d['seed']:d for d in base}
    result={}
    for nside in sorted(set(d['nside'] for d in data)):
        fine=[d for d in data if d['nside']==nside]
        assert sorted(d['seed'] for d in fine)==list(range(16)),f'Incomplete Nside {nside} sampling control'
        pairs=[(coarse[d['seed']],d) for d in fine]
        item={'seeds':len(fine),'noise_validation':noise_validation(fine,geometry),
              'transferred_noise_validation':transferred_noise_validation(base,fine,geometry),
              'paired_variance_gain':{}}
        white_difference=[]
        source_difference=[]
        for old,new in pairs:
            old_rows={key(r):r for r in old['rows']}
            for r in new['rows']:
                before=old_rows[key(r)]
                white_difference.append(abs(r['white_blank_uK']-before['white_blank_uK']))
                source_difference.append(abs((r['fixed_uK']-r['blank_uK'])-(before['fixed_uK']-before['blank_uK'])))
        item['paired_white_max_difference_uK']=max(white_difference)
        item['paired_source_max_difference_uK']=max(source_difference)
        assert max(white_difference)<1e-10
        assert max(source_difference)<1e-10
        for proj in PROJS:
            item['paired_variance_gain'][proj]={}
            for kind in ('point','cluster'):
                gr=[r for r in geometry if r['projection']==proj and r['profile']==kind and r['noise']=='cmb_white']
                pred=np.array([r['finite_patch_sigma_cmb_uK'] for r in gr])
                def array(ds):
                    dictionaries=[{key(r):r for r in d['rows']} for d in ds]
                    return np.array([[d[key(g)]['cmb_blank_uK'] for g in gr] for d in dictionaries])/pred
                old=array([a for a,b in pairs]); new=array([b for a,b in pairs])
                def gain(a,b):
                    return np.var(b,axis=0,ddof=1).mean()/np.var(a,axis=0,ddof=1).mean()
                value=gain(old,new)
                leave=np.array([gain(np.delete(old,i,0),np.delete(new,i,0)) for i in range(len(old))])
                error=np.sqrt((len(old)-1)/len(old)*np.sum((leave-leave.mean())**2))
                item['paired_variance_gain'][proj][kind]=dict(ratio=float(value),jackknife_error=float(error))
        result[str(nside)]=item
    return result


def transferred_noise_validation(base,fine,geometry):
    """64-sky coarse variances calibrated by the 16 paired fine/coarse ratios.

    The jackknife preserves the overlap: deleting a paired sky removes it
    from all three variance estimates. Paired and coarse-only skies form
    separate jackknife strata; no independence of gain and coarse variance
    is assumed. This is a transfer estimate, not 64 fine-sky realizations.
    """
    base=sorted(base,key=lambda d:d['seed'])
    fine=sorted(fine,key=lambda d:d['seed'])
    paired=np.array([d['seed'] for d in fine])
    coarse_only=np.array([i for i in range(len(base)) if i not in paired])
    bd=[{key(r):r for r in d['rows']} for d in base]
    fd=[{key(r):r for r in d['rows']} for d in fine]
    result={}
    for proj in PROJS:
        result[proj]={}
        for kind in ('point','cluster'):
            gr=sorted([r for r in geometry if r['projection']==proj and r['profile']==kind and r['noise']=='cmb_white'],key=lambda r:(r['ra'],r['dec']))
            pred=np.array([g['finite_patch_sigma_cmb_uK']**2 for g in gr])
            white=np.array([g['finite_patch_sigma_white_uK']**2 for g in gr])
            amps=np.array([g['amplitude'] for g in gr])
            b=np.array([[d[key(g)]['cmb_blank_uK'] for g in gr] for d in bd])
            f=np.array([[d[key(g)]['cmb_blank_uK'] for g in gr] for d in fd])
            corner=next(i for i,g in enumerate(gr) if g['ra']==45 and g['dec']==-68)
            center=next(i for i,g in enumerate(gr) if g['ra']==0 and g['dec']==-56)
            def statistic(all_coarse,paired_coarse,paired_fine):
                gain=np.var(paired_fine,axis=0,ddof=1)/np.var(paired_coarse,axis=0,ddof=1)
                variance=np.var(all_coarse,axis=0,ddof=1)*gain
                snr=amps[corner]/amps[center]*np.sqrt((variance[center]+white[center])/(variance[corner]+white[corner]))
                return np.array([np.mean(variance/pred),snr])
            value=statistic(b,b[paired],f)
            leave=[]
            for i in range(len(base)):
                if i in paired:
                    mask=paired!=i
                    leave.append(statistic(np.delete(b,i,0),b[paired[mask]],f[mask]))
                else:
                    leave.append(statistic(np.delete(b,i,0),b[paired],f))
            leave=np.array(leave)
            variance=np.zeros(2)
            for stratum in (paired,coarse_only):
                subset=leave[stratum]
                variance+=(len(stratum)-1)/len(stratum)*np.sum((subset-subset.mean(0))**2,axis=0)
            error=np.sqrt(variance)
            result[proj][kind]=dict(cmb_variance_ratio=float(value[0]),
                cmb_variance_ratio_jackknife_error=float(error[0]),
                corner_snr=float(value[1]),corner_snr_jackknife_error=float(error[1]),
                covariance_note='Joint whole-sky jackknife in paired/coarse-only strata; includes overlap of coarse variances and transfer gains.')
    return result


def noise_validation(sims,geometry):
    """Whole-sky jackknife: locations within a CMB seed are not independent."""
    assert sims and all('cmb_blank_uK' in d['rows'][0] for d in sims),'Missing CMB-only blank estimators'
    by_seed=[{key(r):r for r in d['rows']} for d in sims]
    result={}
    for proj in PROJS:
        result[proj]={}
        for kind in ('point','cluster'):
            gr=sorted([r for r in geometry if r['projection']==proj and r['profile']==kind and r['noise']=='cmb_white'],key=lambda r:(r['ra'],r['dec']))
            cmb=np.array([[s[key(g)]['cmb_blank_uK'] for g in gr] for s in by_seed])
            predicted=np.array([g['finite_patch_sigma_cmb_uK']**2 for g in gr])
            white=np.array([g['finite_patch_sigma_white_uK']**2 for g in gr])
            amps=np.array([g['amplitude'] for g in gr])
            corner=next(i for i,g in enumerate(gr) if g['ra']==45 and g['dec']==-68)
            center=next(i for i,g in enumerate(gr) if g['ra']==0 and g['dec']==-56)
            def statistic(c):
                variance=np.var(c,axis=0,ddof=1)
                ratio=np.mean(variance/predicted)
                snr=amps[corner]/amps[center]*np.sqrt((variance[center]+white[center])/(variance[corner]+white[corner]))
                return np.array([ratio,snr])
            value=statistic(cmb)
            leave=np.array([statistic(np.delete(cmb,i,axis=0)) for i in range(len(cmb))])
            error=np.sqrt((len(cmb)-1)/len(cmb)*np.sum((leave-leave.mean(0))**2,axis=0))
            result[proj][kind]=dict(cmb_variance_ratio=float(value[0]),
                cmb_variance_ratio_jackknife_error=float(error[0]),
                corner_snr_empirical_cmb_plus_analytic_white=float(value[1]),
                corner_snr_jackknife_error=float(error[1]),
                corner_snr_affine_prediction=gr[corner]['snr_relative_center_finite_patch'],
                covariance_note='Jackknife over entire spherical CMB seeds; no independent-location assumption.')
    return result


def affine_prediction_errors(data):
    """Predict all profiles/weights from the local metric, not spherical samples."""
    from scipy import fft
    import geometry as G
    from local_measurements import profiles, ARCMIN
    theory=np.load(ROOT/'analysis/theory_spectra.npz')
    n,pixel=data['n'],data['res_arcmin']*ARCMIN
    axis=(np.arange(n)-n//2)*pixel
    y,x=np.meshgrid(axis,axis,indexing='ij')
    k=2*np.pi*fft.fftfreq(n,pixel)
    ell=np.hypot(k[:,None],k[None,:])
    sigma=1.2/np.sqrt(8*np.log(2))*ARCMIN
    cmb=np.interp(ell,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(ell*sigma)**2)
    powers={'white':np.full_like(ell,(18*ARCMIN)**2),'cmb_white':cmb+(18*ARCMIN)**2}
    powers['cmb_white'][ell<300]=np.inf
    templates={kind:fft.fft2(profiles(np.hypot(x,y)/ARCMIN,kind)) for kind in ('point','cluster')}
    errors={}
    noise_model={}
    central_sigma={}
    for kind,tf in templates.items():
        for name,power in powers.items():
            central_sigma[kind,name]=n/(pixel*np.sqrt(np.sum(abs(tf)**2/power).real))
    for row in data['rows']:
        jac=G.jacobian(row['projection'],[row['ra']],[row['dec']])[0]
        tangent=np.linalg.solve(jac,np.stack([x.ravel(),y.ravel()]))
        affine=profiles(np.linalg.norm(tangent,axis=0).reshape(x.shape)/ARCMIN,row['profile'])
        tf=templates[row['profile']]
        power=powers[row['noise']]
        af=fft.fft2(affine)
        norm=np.sum(abs(tf)**2/power).real
        predicted=np.sum(np.conj(tf)*af/power).real/norm
        label=row['profile']+'_'+row['noise']
        errors[label]=max(errors.get(label,0.),abs(predicted-row['amplitude']))
        qx=jac[0,0]*k[None,:]+jac[1,0]*k[:,None]
        qy=jac[0,1]*k[None,:]+jac[1,1]*k[:,None]
        q=np.hypot(qx,qy)
        mu=abs(np.linalg.det(jac))
        actual_cmb=mu*np.interp(q,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(q*sigma)**2)
        actual_white=mu*(18*ARCMIN)**2
        var_white=n*n*np.sum(abs(tf)**2*actual_white/power**2).real/(pixel**2*norm**2)
        var_cmb=n*n*np.sum(abs(tf)**2*actual_cmb/power**2).real/(pixel**2*norm**2) if row['noise']=='cmb_white' else 0.
        actual=actual_cmb+actual_white if row['noise']=='cmb_white' else np.full_like(ell,actual_white)
        actual=actual.copy()
        if row['noise']=='cmb_white':
            actual[ell<300]=np.inf
        optimal_sigma=n/(pixel*np.sqrt(np.sum(abs(af)**2/actual).real))
        reference=central_sigma[row['profile'],row['noise']]
        noise_model[key(row)+(row['noise'],)]=dict(sigma_cmb_uK=float(np.sqrt(var_cmb)),
            sigma_white_uK=float(np.sqrt(var_white)),
            snr_relative_center=float(row['amplitude']*reference/np.sqrt(var_cmb+var_white)),
            optimal_snr_relative_center=float(reference/optimal_sigma))
    return errors,noise_model


def plot(rows,out,quantity='amplitude'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as PS
    with plt.rc_context(PS.rc()):
        fig,axs=plt.subplots(2,3,figsize=(7.2,4.1),sharex=True)
        for j,proj in enumerate(PROJS):
            for i,kind in enumerate(('point','cluster')):
                ax=axs[i,j]
                ax.axhline(0,color='.7',lw=.6)
                for dec,color in zip((-68.,-56.,-44.),PS.SECONDARY[:3]):
                    rs=sorted([r for r in rows if r['projection']==proj and r['profile']==kind and r['noise']=='cmb_white' and r['dec']==dec],key=lambda r:r['ra'])
                    ax.plot([r['ra'] for r in rs],[100*(r[quantity]-1) for r in rs],'-o',ms=3,color=color,label=rf'$\delta={dec:g}^\circ$')
                if proj=='ProjBICEP' and kind=='point':
                    ax.legend(title='BICEP point',fontsize=6,title_fontsize=7,loc='center right',bbox_to_anchor=(1,.72))
                else:
                    ax.legend(title=proj[4:]+' '+kind,fontsize=6,title_fontsize=7,loc='best')
                ax.set_ylabel(r'$\hat A/A-1$ [\%]' if quantity=='amplitude' else r'$(S/N)/(S/N)_0-1$ [\%]')
                if i==1:
                    ax.set_xlabel('Right ascension [deg]')
        fig.tight_layout()
        name='local_source_bias' if quantity=='amplitude' else 'local_source_snr'
        fig.savefig(out/(name+'.png'),dpi=170)
        fig.savefig(out/(name+'.pdf'))
        plt.close(fig)


def plot_examples(path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    d=np.load(path)
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.6),sharex=True,sharey=True)
    for j,p in enumerate(PROJS):
        for i,kind in enumerate(('point','cluster')):
            image=d[p+'_'+kind+'_image']
            source=d[p+'_'+kind+'_source']
            # Display-only 0.4 arcmin Gaussian smoothing, not used by estimator.
            shown=gaussian_filter(image,2.)
            ax=axs[i,j]
            im=ax.imshow(shown,origin='lower',extent=(-25.7,25.5,-25.7,25.5),cmap='RdBu_r',vmin=-150,vmax=150)
            ax.contour(source,levels=[.2,.5,.8],origin='lower',extent=(-25.7,25.5,-25.7,25.5),colors='k',linewidths=.5)
            ax.plot([],[],color='k',label=p[4:]+' '+kind)
            ax.legend(fontsize=7,loc='upper right')
            ax.set(xlim=(-4,4),ylim=(-4,4),xticks=[-3,0,3],yticks=[-3,0,3])
            ax.set_xlabel(r'$\Delta x$ [arcmin]',fontsize=9)
            ax.tick_params(labelsize=8)
            if j==0:
                ax.set_ylabel(r'$\Delta y$ [arcmin]',fontsize=9)
    fig.tight_layout(rect=(0,0,.91,1))
    cax=fig.add_axes([.92,.16,.018,.68])
    bar=fig.colorbar(im,cax=cax)
    bar.set_label('Temperature [uK]',fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.savefig(out/'local_source_examples.png',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,default=ROOT/'results/highell/local_source_geometry/geometry.json')
    p.add_argument('--sims',type=Path,default=ROOT/'results/highell/local_source_sims')
    p.add_argument('--out',type=Path,default=ROOT/'analysis/results_disk_trend/local_sources')
    p.add_argument('--fine',type=Path)
    p.add_argument('--sampling',type=Path,default=ROOT/'results/highell/local_source_resolution')
    a=p.parse_args()
    report(a.geometry,a.sims,a.out,a.fine,a.sampling)
