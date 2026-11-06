"""Window-plus-alias closure and paired-seed projection contrasts.

Aliases are included in this diagnostic prediction, not deconvolved from data.
The paper's model_residuals figure instead shows the window-only correction.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spectra as S
import paper_style as PS

DATA=Path('/sptlocal/user/cailmdaley/pol_project/pipeline_windows_production.npz')


def main():
    plt.rcParams.update(PS.rc())
    d=np.load(DATA)
    ell=d['ell']
    fig,axs=plt.subplots(2,3,figsize=(10,5.2),sharex=True,layout='constrained')
    for col,spec in enumerate(('TT','EE','BB')):
        ref=d[f'field_{spec}_ProjZEA_ratio_total']
        for proj in S.PROJS:
            if spec=='BB' and proj=='ProjSTG': continue
            r=d[f'field_{spec}_{proj}_ratio_total']
            for row,values in enumerate((r-1,r-ref)):
                mean=100*values.mean(0)
                err=100*values.std(0,ddof=1)/np.sqrt(r.shape[0])
                ax=axs[row,col]
                ax.plot(ell,mean,color=PS.PROJ_COLORS[proj],label=S.PROJ_LABEL[proj])
                ax.fill_between(ell,mean-err,mean+err,color=PS.PROJ_COLORS[proj],alpha=.15,lw=0)
                ax.axhline(0,color='.5',lw=.5)
                ax.set_xlim(1000,4800)
                ax.set_ylim((-1.5,2) if spec!='BB' else (-2,12) if row==0 else (-12,2))
        axs[0,col].set_ylabel(rf'$C^{{{spec}}}/C^{{{spec},\rm win+alias}}-1$ [\%]')
        axs[1,col].set_ylabel(rf'$r_{{{spec}}}-r_{{{spec},\rm ZEA}}$ [\%]')
        axs[1,col].set_xlabel(r'$\ell$')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='outside upper center')
    for ext in ('png','pdf'):
        path=DATA.parent/f'pipeline_window_residuals.{ext}'
        fig.savefig(path,dpi=160)
        print(path)
    plt.close(fig)

if __name__=='__main__': main()
