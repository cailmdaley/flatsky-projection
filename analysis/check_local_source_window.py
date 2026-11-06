"""Fixed-pixel, enlarged-box gate for source amplitude (not background covariance)."""
import json
from pathlib import Path
import numpy as np
from scipy import fft
from local_measurements import profiles,cutout,ARCMIN


def run():
    theory=np.load(Path(__file__).with_name('theory_spectra.npz'))
    rows=[]
    for n in (256,512,1024):
        pix=.2*ARCMIN
        k=2*np.pi*fft.fftfreq(n,pix)
        ell=np.hypot(k[:,None],k[None,:])
        sigma=1.2/np.sqrt(8*np.log(2))*ARCMIN
        power=np.interp(ell,theory['ell'],theory['lensed_TT'],right=0.)*np.exp(-(ell*sigma)**2)+(18*ARCMIN)**2
        power[ell<300]=np.inf
        for proj in ('ProjSFL','ProjZEA','ProjBICEP'):
            x,y,r=cutout(proj,45.,-68.,n,.2)
            for kind in ('point','cluster'):
                tf=fft.fft2(profiles(np.hypot(x,y)/ARCMIN,kind))
                sf=fft.fft2(profiles(r,kind))
                amp=np.sum(np.conj(tf)*sf/power).real/np.sum(abs(tf)**2/power).real
                rows.append(dict(n=n,projection=proj,profile=kind,amplitude=float(amp)))
    delta=[]
    for proj in ('ProjSFL','ProjZEA','ProjBICEP'):
        for kind in ('point','cluster'):
            amps=[r['amplitude'] for r in rows if r['projection']==proj and r['profile']==kind]
            delta.append(abs(amps[-1]-amps[0]))
            assert abs(amps[-1]-amps[-2])<1e-5,(proj,kind,amps)
    assert max(delta)<2e-4,delta
    print(json.dumps(dict(rows=rows,max_fixed_pixel_box_change=max(delta)),indent=2))


if __name__=='__main__':
    run()
