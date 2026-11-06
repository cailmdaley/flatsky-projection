"""Paired spherical band-limit intervention and local-affine alias prediction.

The existing cutoff-6000 seed must reproduce production before comparisons
are interpreted. The 5000 realization differs only in the harmonic cutoff.
"""
from pathlib import Path
import numpy as np
import spectra as S

DIRECTORY=S.ROOT/'results'/'highell'/'window_bandlimit_test'


def records():
    data={}
    for proj in S.PROJS:
        path=DIRECTORY/f'bandlimit_{proj}_seed0000.npz'
        if not path.exists():
            path=DIRECTORY.with_name(DIRECTORY.name+'__units')/f'0-{proj}'/path.name
        if not path.exists(): continue
        with np.load(path) as z: data[proj]={k:z[k] for k in z.files}
        j=S.PROJS.index(proj)
        for spec in ('EE','BB'):
            production=S.load_field(spec,correct_windows=False)[j,0]
            np.testing.assert_allclose(data[proj][spec.lower()+'_6000'],production,rtol=2e-12,atol=1e-25)
        print(proj,'cutoff6000 reproduces production',flush=True)
    return data


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as PS
    plt.rcParams.update(PS.rc())
    data=records()
    alias=np.load('/sptlocal/user/cailmdaley/pol_project/pipeline_aliases.npz')
    fig,axes=plt.subplots(2,4,figsize=(11,5),sharex=True,layout='constrained')
    for j,proj in enumerate(S.PROJS):
        if proj not in data: continue
        d=data[proj]; ell=d['ell']
        for row,spec in enumerate(('EE','BB')):
            ax=axes[row,j]
            for cutoff,color in ((6000,'#b13c4e'),(5000,'#276f9c')):
                key=f'{cutoff}_{proj}_{spec}'
                pred=alias[key+'_windowed']+alias[key+'_alias']
                ax.plot(ell,d[f'{spec.lower()}_{cutoff}']/S.theory('unlensed_EE'),color=color,label=str(cutoff))
                ax.plot(alias['ell'],pred/np.interp(alias['ell'],S.ELL,S.theory('unlensed_EE')),color=color,ls='--',lw=1)
                for lo,hi in ((4000,4500),(4500,4800),(4800,5400),(5400,5800)):
                    sel=(ell>=lo)&(ell<hi)
                    print(proj,spec,cutoff,lo,hi,'mean/input_EE',float(np.mean(d[f'{spec.lower()}_{cutoff}'][sel]/S.theory('unlensed_EE')[sel])),flush=True)
            ax.set_xlim(3500,6000)
            ax.set_yscale('log')
            ax.set_ylabel(rf'$C^{{{spec}}}/C^{{EE,\rm sky}}$ ({S.PROJ_LABEL[proj]})')
            if row==1: ax.set_xlabel(r'$\ell$')
            if row==0: ax.set_ylim(.1,20)
            else: ax.set_ylim(1e-6,2)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],color='#b13c4e'),Line2D([],[],color='#276f9c'),Line2D([],[],color='.3',ls='--')],
               ['Sky cutoff 6000','Sky cutoff 5000','Local-affine alias model'],ncol=3,loc='outside upper center')
    for ext in ('png','pdf'):
        fig.savefig(f'/sptlocal/user/cailmdaley/pol_project/pipeline_bandlimit_test.{ext}',dpi=160)
    plt.close(fig)

if __name__=='__main__': main()
