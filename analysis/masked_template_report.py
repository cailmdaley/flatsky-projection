"""Report the edge-limited template and split-cross sample-variance test.

Power subtraction means subtraction of a FIXED ensemble R_BB<-EE C_EE
prediction, whose variance is exactly that of raw BB. A supplementary
EE-amplitude-conditioned diagonal reference is not the full R estimator.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--expected-seeds',type=int,default=20)
    a=p.parse_args()
    files=sorted(Path(a.input).glob('masked_template_seed*.npz'))
    data=[dict(np.load(f)) for f in files]
    n=len(data)
    assert sorted(int(d['seed']) for d in data)==list(range(a.expected_seeds)), 'incomplete or duplicated seed set'
    ell=data[0]['ell']
    rows={k:np.array([d[k] for d in data]) for k in data[0] if k.startswith(('bb_','ee_'))}
    variants=[k for k in ('oracle','wide','masked','pseudo') if 'bb_cross_'+k in rows]
    # Local white-E-noise prediction with the SAME isotropic Wiener prior.
    # The signal term uses the actual raw BB (no fitted noise amplitude).
    # Angular pixel response is averaged; boundary/pure-B coupling is omitted.
    import theory as T
    th=T.load();phi=(np.arange(256)+.5)*2*np.pi/256
    freq=ell[:,None]*np.deg2rad(2/60)/(2*np.pi)
    pix=(np.sinc(freq*np.cos(phi))*np.sinc(freq*np.sin(phi)))**2
    prior=np.interp(ell,th['ell'],th['unlensed_EE'])[:,None]*pix
    noise=(5*np.deg2rad(1/60))**2
    wf=prior/(prior+noise)
    pred=(np.mean((1-wf)**2,axis=1)*rows['bb_signal'].mean(0)
          +float(data[0]['tan2'])*noise*np.mean(wf**2,axis=1))
    rng=np.random.default_rng(98372)
    boots=rng.integers(n,size=(4000,n))
    import hashlib
    result={'nseed':n,'delta_P_per_split_uK_arcmin':5,
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'scope':'SFL, polar reflattening, 1500d 2.5-degree C2 analysis mask; coupled pure-B spectra divided by mean W^2. Masked/pseudo use a sharper 0.5-degree E taper; wide control uses the analysis taper. Scalar isotropic Wiener weight, not optimal cut-sky posterior. Signal sky B=0. Fixed ensemble power subtraction; a diagonal EE-amplitude-conditioned reference is supplementary, not the full R operator. Variance gains presume a known mean residual is subtracted in BOTH arms.',
        'bands':{}}
    for lo,hi in ((900,1100),(1000,2000),(2000,3000),(3000,4000)):
        sel=(ell>=lo)&(ell<hi)
        # Average C_l within the band before moments across seeds.
        vals={k:v[:,sel].mean(1) for k,v in rows.items()}
        band={}
        for key in variants:
            raw=vals['bb_signal']
            residual=vals['bb_isolated_'+key]
            cross=vals['bb_cross_'+key]
            orig=vals['bb_cross_raw']
            gain=np.std(orig,ddof=1)/np.std(cross,ddof=1) if n>1 else None
            interval=None
            if n>2:
                ratios=np.std(orig[boots],axis=1,ddof=1)/np.std(cross[boots],axis=1,ddof=1)
                interval=np.quantile(ratios,[.025,.975]).tolist()
            band[key]={
                'noiseless_residual_over_raw':float(vals['bb_signal_'+key].mean()/raw.mean()),
                'isolated_noisy_residual_over_raw':float(residual.mean()/raw.mean()),
                'isolated_noisy_residual_SE_over_raw':float(residual.std(ddof=1)/np.sqrt(n)/raw.mean()) if n>1 else None,
                'cross_mean_over_raw_signal':float(cross.mean()/raw.mean()),
                'cross_mean_SE_over_raw_signal':float(cross.std(ddof=1)/np.sqrt(n)/raw.mean()) if n>1 else None,
                'cross_standard_deviation_gain_vs_fixed_power_subtraction':float(gain) if gain is not None else None,
                'paired_seed_bootstrap_gain_95pct':interval,
                'sample_RMS_gain_without_residual_debiasing':float(np.std(orig,ddof=1)/np.sqrt(np.mean(cross**2))) if n>1 else None}
            if n>1 and 'ee_cross_raw' in vals:
                ratio=vals['bb_signal'].mean()/vals['ee_signal'].mean()
                conditioned=orig-ratio*vals['ee_cross_raw']
                band[key]['cross_standard_deviation_gain_vs_EE_amplitude_conditioned']=float(np.std(conditioned,ddof=1)/np.std(cross,ddof=1))
        band['local_Wiener_noise_prediction_over_raw']=float(pred[sel].mean()/vals['bb_signal'].mean())
        band['oracle_over_local_noise_prediction']=float(vals['bb_isolated_oracle'].mean()/pred[sel].mean())
        band['estimator_cost_isolated_masked_over_oracle']=float(
            vals['bb_isolated_masked'].mean()/vals['bb_isolated_oracle'].mean())
        result['bands'][f'{lo}-{hi}']=band
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as PS
    plt.rcParams.update(PS.rc());plt.rcParams['text.usetex']=False
    fig,axes=plt.subplots(1,2,figsize=(PS.TEXT_W,2.85),layout='constrained')
    keep=(ell>=500)&(ell<=4000);fac=ell*(ell+1)/(2*np.pi)
    styles={'signal':('Raw projection','-',PS.SECONDARY[0]),
            'isolated_oracle':('Noisy E, full-grid oracle','--',PS.SECONDARY[1]),
            'isolated_masked':('Noisy E, masked pure E','-',PS.SECONDARY[2]),
            'isolated_wide':('Noisy E, wide-taper control',':',PS.SECONDARY[1])}
    for key,(label,ls,col) in styles.items():
        if 'bb_'+key not in rows:
            continue
        axes[0].plot(ell[keep],(rows['bb_'+key].mean(0)*fac)[keep],ls=ls,color=col,label=label)
    axes[0].plot(ell[keep],(pred*fac)[keep],':',color=PS.SECONDARY[0],label='Local Wiener noise limit')
    axes[0].set(yscale='log',xlabel=r'$\ell$',ylabel=r'$D_\ell^{BB}$ [$\mu$K$^2$]')
    axes[0].legend(fontsize=6.5)
    if n>1:
        for key,ls,col in [('oracle','--',PS.SECONDARY[1]),('masked','-',PS.SECONDARY[0])]:
            centers=[1000,1500,2500,3500]
            stats=[band[key] for band in result['bands'].values()]
            gain=np.array([v['cross_standard_deviation_gain_vs_fixed_power_subtraction'] for v in stats])
            bounds=np.array([v['paired_seed_bootstrap_gain_95pct'] for v in stats])
            errors=np.maximum(np.array([gain-bounds[:,0],bounds[:,1]-gain]),0) if n>2 else None
            axes[1].errorbar(centers,gain,yerr=errors,ls=ls,color=col,marker='o' if key=='masked' else 's',mfc='white',capsize=2,label=key.capitalize()+' E')
        axes[1].axhline(1,color=PS.RULE,lw=.7)
        axes[1].set(xlabel=r'$\ell$',ylabel=r'$\sigma(BB_{\rm raw})/\sigma(BB_{\rm residual})$')
        axes[1].legend()
    else:
        for key,ls,col in [('oracle','--',PS.SECONDARY[1]),('masked','-',PS.SECONDARY[0])]:
            ratio=rows['bb_isolated_'+key].mean(0)/rows['bb_signal'].mean(0)
            axes[1].plot(ell[keep],ratio[keep],ls=ls,color=col,label=key.capitalize()+' E')
        axes[1].set(xlabel=r'$\ell$',ylabel='Isolated residual / raw BB',yscale='log')
        axes[1].legend()
    fig.savefig(out/'masked_template.png',dpi=180)
    fig.savefig(out/'masked_template.pdf')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
