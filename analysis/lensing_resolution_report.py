"""Compare matched 4'/2' TT-QE outputs, including an injection identity gate.

Both grids share the continuous potential and flat sky; spherical realizations
share their harmonic seed. This changes Nside and output pixels together,
so it is a numerical control, not an isolation of either error source.
`report` reads the lensing_qe_sims and lensing_qe_resolution outputs and the
squeezed metric prediction; `lensing_summary.py` runs it.
"""
import numpy as np
from lensing_report import ratio_error
from paper_style import PROJS


def paired_statistic(arrays, statistic):
    """Evaluate a statistic of paired seed means with its jackknife SE."""
    n=len(arrays[0])
    value=statistic(*[v.mean() for v in arrays])
    jack=np.array([statistic(*[(v.sum()-v[i])/(n-1) for v in arrays])
                   for i in range(n)])
    error=np.sqrt((n-1)*np.mean((jack-jack.mean())**2))
    return {'value':float(value),'SE':float(error)}


def report(coarse_dir,fine_dir,prediction):
    """Per projection, the paired coarse/fine response and template fits;
    each fine seed is paired with the coarse seed of the same file name."""
    result={}
    for proj in PROJS:
        fine_files=sorted(fine_dir.glob(f'qe_{proj}_*.npz'))
        coarse=[np.load(coarse_dir/f.name) for f in fine_files]
        fine=[np.load(f) for f in fine_files]
        x,y=np.meshgrid(coarse[0]['lx'],coarse[0]['ly'])
        ls=np.hypot(x,y)
        phase=np.exp(.5j*(x+y)*(float(fine[0]['pixel_rad'])-float(coarse[0]['pixel_rad'])))
        max_error=0.
        for c,f in zip(coarse,fine):
            np.testing.assert_allclose(c['lx'],f['lx'],rtol=1e-14,atol=1e-12)
            np.testing.assert_allclose(c['ly'],f['ly'],rtol=1e-14,atol=1e-12)
            np.testing.assert_allclose(c['input_kappa']*phase,f['input_kappa'],rtol=2e-14,atol=1e-18)
            max_error=max(max_error,float(np.max(abs(c['input_kappa']*phase-f['input_kappa']))))
        rows=[]
        for lo,hi in ((30,60),(60,120),(120,240)):
            keep=(ls>=lo)&(ls<hi)
            channels=[]
            for i,label in enumerate(('gradient','curl')):
                values=[]
                for data in (coarse,fine):
                    k=data[0]['input_kappa'][keep]
                    for key in (label,'flat_'+label):
                        values.append(np.array([np.vdot(k,v[key][i,keep]).real for v in data]))
                cs,cf,fs,ff=values
                cr,ce=ratio_error(cs,cf)
                fr,fe=ratio_error(fs,ff)
                channels.append({'channel':label,'coarse':{'value':cr,'SE':ce},
                                 'fine':{'value':fr,'SE':fe},
                                 'fractional_change':paired_statistic(values,lambda cs,cf,fs,ff:fs*cf/(ff*cs)-1)})
            rows.append({'L':[lo,hi],'response':channels})
        fits=[]
        keep=(ls>=2)&(ls<30)
        for i,label in enumerate(('gradient','curl')):
            if proj=='ProjSTG' and i==1:
                continue
            slopes=[]
            for data in (coarse,fine):
                model=prediction[proj][i]*np.exp(.5j*(x+y)*(float(data[0]['pixel_rad'])-float(prediction['pixel_rad'])))
                model=model[keep]
                slopes.append(np.array([np.vdot(model,(v['spherical']-v['flat'])[i,keep]).real/np.sum(abs(model)**2) for v in data]))
            fits.append({'channel':label,
                         'coarse':paired_statistic(slopes[:1],lambda x:x),
                         'fine':paired_statistic(slopes[1:],lambda x:x),
                         'fractional_change':paired_statistic(slopes,lambda c,f:f/c-1)})
        result[proj]={'seeds':[int(f.stem.rsplit('_',1)[1]) for f in fine_files],
                      'input_kappa_max_absolute_difference_after_phase':max_error,
                      'response':rows,'mean_field_template_fit_L2_30':fits}
    return result
