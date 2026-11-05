"""Plot the anisotropic quadrupole response computed by paper_numbers.py."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import paper_style as PS
import section5_geometry as G5

ROOT=Path(__file__).resolve().parents[1]


def run():
    data=json.loads((ROOT/'paper'/'section5_summary.json').read_text())
    with plt.rc_context(PS.rc()):
        fig,axs=plt.subplots(1,2,figsize=(PS.TEXT_W,2.6))
        g=np.linspace(0,.3,100)
        ax=axs[0]
        for attachment,coefficient,color,label in (
            ('sky',-.5,PS.SECONDARY[0],'Local sky scan'),
            ('grid',-1.5,PS.SECONDARY[2],'Output-grid filter')):
            response=[]
            for x in g:
                s=np.sqrt((1+x)/(1-x))
                response.append(G5.anisotropic_response(np.diag((s,1/s))[None],[1.],
                                                         attachment=attachment))
            ax.plot(g,100*np.array(response),color=color,label=label)
            ax.plot(g,100*.3*g*coefficient,'--',color=color,lw=.7)
        ax.set(xlabel=r'Singular-value shear $g$',ylabel=r'$\delta C/C$ [\%]',
               title=r'Local equal-area stretch: $n=-3$, $a_2=0.3$')
        ax.legend(loc='lower left')
        ax=axs[1]
        angle=np.linspace(0,180,200)
        for proj in ('ProjSFL','ProjZEA'):
            for footprint,style in (('field1500d','-'),('fieldEast','--')):
                row=data['anisotropic'][f'sky_{footprint}_{proj}']
                y=row['chi_zero']*np.cos(2*np.deg2rad(angle))
                y+=row['chi_diagonal']*np.sin(2*np.deg2rad(angle))
                label=f"{proj[4:]}: "+('1500d' if footprint=='field1500d' else 'eastern half')
                ax.plot(angle,100*y,style,color=PS.PROJ_COLORS[proj],label=label,
                        marker=PS.PROJ_MARKERS[proj],markevery=40,ms=2)
        ax.axhline(0,color=PS.RULE,lw=.5)
        ax.set(xlabel=r'Quadrupole direction $\chi_0$ [deg]',
               ylabel=r'$\delta C/C$ [\%]',title=r'Observed-power quadrupole: $n=-3$, $a_2=0.3$')
        ax.legend(loc='upper left',fontsize=6.6)
        fig.tight_layout(pad=.7,w_pad=1)
        path=ROOT/'figures'/'anisotropic_response.pdf'
        fig.savefig(path)
        fig.savefig(path.with_suffix('.png'),dpi=180)
        plt.close(fig)
        print(path)


if __name__=='__main__':
    run()
