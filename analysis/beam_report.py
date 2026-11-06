"""Model and plot the lc beam experiment, retaining its numerical controls."""

from pathlib import Path
import json
import sys

import numpy as np
# Load NaMaster's CFITSIO before spt3g (via geometry), which loads an older ABI.
import pymaster as nmt

import section5_geometry as G5

ROOT=Path(__file__).resolve().parents[1]
MAPS=ROOT/'analysis'/'beams'/'results'/'baseline'/'beam_response'
MODEL=ROOT/'analysis'/'results_disk_trend'/'beam_target_prediction.npz'
BCACHE=ROOT/'analysis'/'results_disk_trend'/'beam_b_diagnostics.npz'


def target_model(ell,chi):
    """Integrate numerator and denominator over each measured Fourier bin."""
    gx,gw=np.polynomial.legendre.leggauss(4)
    le=(ell[:,None]+50*gx).ravel()
    ca=(chi[:,None]+np.pi/24*gx).ravel()
    raw=G5.beam_target_prediction(le,ca)
    out={}
    for footprint,terms in raw.items():
        integrated=[]
        for term in terms:
            z=term.reshape(len(ell),4,len(chi),4)
            weight=(gw[None,:,None,None]*gw[None,None,None,:]
                    *le.reshape(len(ell),4)[:,:,None,None])
            integrated.append(np.sum(z*weight,axis=(1,3)))
        ns,ds,nr,dr=integrated
        out[footprint+'_source2d']=ns/ds
        out[footprint+'_sphere2d']=nr/dr
        out[footprint+'_source']=ns.sum(1)/ds.sum(1)
        out[footprint+'_sphere']=nr.sum(1)/dr.sum(1)
    np.savez(MODEL,ell=ell,chi=chi,**out)
    return out


def b_diagnostics():
    """Use the production pure-B workspaces and retain all cross terms."""
    sys.path.insert(0, str(ROOT/'src'/'stages'))
    from beam_reprojection import mask_slices
    lo=np.arange(0.,6000.,50.)
    bins=nmt.NmtBinFlat(lo,lo+50)
    out={'ell':lo+25}
    for proj in ('ProjZEA','ProjSTG'):
        pairs={tag:np.load(MAPS/f'{tag}_{proj}.npy') for tag in
               ('sfl_base','sfl_beam','sphere_base','sphere_beam')}
        pairs['mismatch']=pairs['sfl_base']-pairs['sphere_base']
        pairs['differential']=((pairs['sfl_beam']-pairs['sfl_base'])
                               -(pairs['sphere_beam']-pairs['sphere_base']))
        for footprint,mi,maskdir,workdir in (
            ('disk5',5,'flatsky_masks','coupling_matrices'),
            ('field1500d',0,'field_flatsky_masks','field_coupling')):
            metadata=np.load(ROOT/'results'/'highell'/maskdir/f'mask_{proj}_{mi}.npz')
            mask=metadata['mask']
            ys,xs=mask_slices(metadata)
            lx,ly=float(metadata['lx_rad']),float(metadata['ly_rad'])
            workspace=nmt.NmtWorkspaceFlat()
            workspace.read_from(str(ROOT/'results'/'highell'/workdir/f'w22_{proj}_{mi}.fits'))
            fields={tag:nmt.NmtFieldFlat(lx,ly,mask,list(pair[:,ys,xs]),purify_b=True)
                    for tag,pair in pairs.items()}
            prefix=f'{footprint}_{proj}_'
            for tag,fld in fields.items():
                coupled=nmt.compute_coupled_cell_flat(fld,fld,bins)
                out[prefix+tag]=workspace.decouple_cell(coupled)
            for other in ('mismatch','sfl_base','sphere_base'):
                cl=nmt.compute_coupled_cell_flat(fields['differential'],fields[other],bins)
                out[prefix+'cross_'+other]=workspace.decouple_cell(cl)
            print('Decoupled B controls',footprint,proj,flush=True)
            del fields
        del pairs
    np.savez(BCACHE,**out)
    return out


def plot(data,model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as PS
    with plt.rc_context(PS.rc()):
        fig,axs=plt.subplots(2,3,figsize=(PS.TEXT_W,4.5))
        ell=data['ell']
        index=np.argmin(abs(ell-3000))
        for row,footprint in enumerate(('disk5','field1500d')):
            prefix=footprint+'_ProjZEA_'
            s=data[prefix+'sfl_response']
            r=data[prefix+'sphere_response']
            ax=axs[row,0]
            ax.plot(ell,100*(s/r-1),'o',color=PS.PROJ_COLORS['ProjSFL'],ms=2.4,label='Maps')
            ax.plot(ell,100*(model[footprint+'_source']/model[footprint+'_sphere']-1),
                    color=PS.PROJ_COLORS['ProjSFL'],label='Local model')
            ax.set(xlim=(500,4000),ylabel=r'$R_{\rm SFL}/R_{\rm sphere}-1$ [\%]')
            ax.set_title(r'5\% disk' if row==0 else '1500d field')
            if row==0:
                ax.legend(fontsize=6.5)
            ax=axs[row,1]
            chi=np.rad2deg(data['chi'])
            for tag,label,color,marker in (
                ('sfl','SFL-filtered',PS.PROJ_COLORS['ProjSFL'],'o'),
                ('sphere','Sphere-filtered',PS.SECONDARY[1],'+')):
                ax.plot(chi,data[prefix+tag+'_response2d'][index],marker,
                        color=color,ms=3,label=label)
                key=footprint+('_source2d' if tag=='sfl' else '_sphere2d')
                ax.plot(chi,model[key][index],color=color)
            ax.set(xlim=(0,180),ylabel='Power response',xlabel=r'Target mode direction $\chi$ [deg]')
            ax.set_title(rf'ZEA target, $\ell\simeq{ell[index]:.0f}$')
            if row==0:
                ax.legend(fontsize=6.5)
            ax=axs[row,2]
            p=footprint+'_ProjSTG_'
            base=data[p+'baseline_ee']
            for tag,label,style in (('mismatch','Unfiltered path mismatch',':'),
                                    ('differential','Filter differential','-')):
                ax.semilogy(ell,data[p+tag+'_bb']/base,style,color=PS.SECONDARY[0],label=label)
            ax.set(xlim=(500,4000),ylabel=r'Pure pseudo-$BB$/baseline pseudo-$EE$')
            ax.set_title('Conformal STG target')
            if row==0:
                ax.legend(fontsize=5.9)
        for ax in axs[:,0]:
            ax.set_xlabel(r'Target multipole $\ell$')
        for ax in axs[:,2]:
            ax.set_xlabel(r'Target multipole $\ell$')
        fig.tight_layout(pad=.8,w_pad=1.2,h_pad=1.2)
        path=ROOT/'figures'/'beam_reprojection.pdf'
        fig.savefig(path)
        fig.savefig(path.with_suffix('.png'),dpi=180)
        plt.close(fig)
        print(path)


def report(data,model,bdiag):
    idx=np.argmin(abs(data['ell']-3000))
    result={'ell_response':float(data['ell'][idx]),'response':{},'b_diagnostic':{}}
    for footprint in ('disk5','field1500d'):
        p=footprint+'_ProjZEA_'
        measured=data[p+'sfl_response'][idx]/data[p+'sphere_response'][idx]-1
        predicted=model[footprint+'_source'][idx]/model[footprint+'_sphere'][idx]-1
        result['response'][footprint]={
            'fractional_difference':float(measured),'prediction':float(predicted),
            'sfl_mean':float(data[p+'sfl_response'][idx]),
            'sphere_mean':float(data[p+'sphere_response'][idx]),
            'sfl_sector_min':float(data[p+'sfl_response2d'][idx].min()),
            'sfl_sector_max':float(data[p+'sfl_response2d'][idx].max())}
        response=result['response'][footprint]
        response['bands']={}
        measured_ratio=data[p+'sfl_response']/data[p+'sphere_response']-1
        predicted_ratio=model[footprint+'_source']/model[footprint+'_sphere']-1
        for lo,hi in ((1000,2000),(2000,3000),(3000,4000)):
            band=(data['ell']>=lo)&(data['ell']<hi)
            response['bands'][f'{lo}_{hi}']={
                'measured':float(measured_ratio[band].mean()),
                'predicted':float(predicted_ratio[band].mean())}
        clean=(data['ell']>=1000)&(data['ell']<=3000)
        residual=data[p+'sfl_response2d']-model[footprint+'_source2d']
        response['sector_model_max_abs_1000_3000']=float(abs(residual[clean]).max())
        sphere_residual=data[p+'sphere_response2d']-model[footprint+'_sphere2d']
        response['sphere_sector_model_max_abs_1000_3000']=float(abs(sphere_residual[clean]).max())
        cosine=np.cos(2*data['chi'])
        measured2d=data[p+'sfl_response2d']/data[p+'sphere_response2d']-1
        predicted2d=model[footprint+'_source2d']/model[footprint+'_sphere2d']-1
        observed_amplitude=2*np.mean(measured2d*cosine,axis=1)
        predicted_amplitude=2*np.mean(predicted2d*cosine,axis=1)
        response['quadrupole_amplitude_slope_1000_3000']=float(
            np.dot(observed_amplitude[clean],predicted_amplitude[clean])
            /np.dot(predicted_amplitude[clean],predicted_amplitude[clean]))
        keep=(bdiag['ell']>=1000)&(bdiag['ell']<3000)
        for proj in ('ProjZEA','ProjSTG'):
            p=f'{footprint}_{proj}_'
            ee=bdiag[p+'sphere_base'][0,keep].mean()
            vals={tag:float(bdiag[p+tag][3,keep].mean()/ee) for tag in
                  ('sfl_base','sfl_beam','sphere_base','sphere_beam','mismatch','differential')}
            for other in ('mismatch','sfl_base','sphere_base'):
                vals['cross_'+other]=float(bdiag[p+'cross_'+other][3,keep].mean()/ee)
            vals['mismatch_power_change'] = vals['differential'] + 2*vals['cross_mismatch']
            result['b_diagnostic'][footprint+'_'+proj]=vals
    path=ROOT/'paper'/'beam_summary.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def run():
    if '--model-only' in sys.argv:
        target_model(np.arange(550.,4000.,100.),(np.arange(12)+.5)*np.pi/12)
        return
    with np.load(MAPS/'summary.npz') as f:
        data={k:f[k] for k in f.files}
    if '--plot-only' in sys.argv or '--reuse-model' in sys.argv:
        with np.load(MODEL) as f:
            model={k:f[k] for k in f.files}
        np.testing.assert_allclose(model['ell'],data['ell'])
        np.testing.assert_allclose(model['chi'],data['chi'])
    else:
        model=target_model(data['ell'],data['chi'])
    if '--plot-only' in sys.argv:
        with np.load(BCACHE) as f:
            bdiag={k:f[k] for k in f.files}
    else:
        bdiag=b_diagnostics()
    plot(data,model)
    report(data,model,bdiag)


if __name__=='__main__':
    run()
