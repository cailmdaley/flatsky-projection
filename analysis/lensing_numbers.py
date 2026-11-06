"""Generated TT-QE numbers for import by analysis/paper_numbers.py.

Run analysis/lensing_mean_prediction.py and analysis/lensing_report.py first;
see notes/cartographic-lensing-reconstruction.md for the scope of the test.
Response macros without 'Sky' are conditional on grid-potential seed 904;
their SE covers TT skies only. qeSky*Response macros use shared physical
spherical potentials varying by seed, with a matched-template overlap and
errors resampling TT+lens pairs. Both use TT legs 300-1200 and neither is
an intrinsic chart response or a cosmological lensing-power bias.
This module does not write numbers.tex: paper_numbers.py remains its writer.
"""
import json
from pathlib import Path


def numbers(summary=None,sky_summary=None):
    if summary is None:
        summary=Path(__file__).resolve().parents[1]/'results/lensing_qe_report/summary.json'
    data=json.loads(Path(summary).read_text())
    out={}
    for proj,row in data.items():
        label=proj.removeprefix('Proj')
        response=row['response'][0]  # 30 <= L < 60
        for i,kind in enumerate(('Gradient','Curl')):
            val=response['response_over_flat'][i][i]
            err=response['SE'][i][i]
            out[f'qe{label}{kind}Response']=rf'{val:.3f}\pm{err:.3f}'
        out[f'qe{label}GammaRMS']=f"{row['metric_rms'][1]:.4f}" if label!='STG' else '0'
        out[f'qe{label}ShearBFraction']=f"{row['metric_pseudo_B_fraction']:.3f}" if label!='STG' else r'\text{--}'
        fit=next(b for b in row['squeezed_metric_prediction_fit'] if b['L']==[2,30])['gradient_curl']
        if fit[1]:
            out[f'qe{label}CurlTemplate']=rf"{fit[1]['amplitude']:.3f}\pm{fit[1]['SE']:.3f}"
        power,error=row['low_L_mean_field_power'][1]
        if power>0 and power>3*error:
            rms=power**.5
            out[f'qe{label}CurlMeanRMS']=rf'{rms:.4f}\pm{error/(2*rms):.4f}'
    out['qeSeeds']=str(data['ProjSFL']['seeds'])
    tail=next(b for b in data['ProjSFL']['mean_field'] if b['L']==[60,120])['mean_field_power'][1]
    out['qeSFLCurlTail']=rf"({tail[0]*1e4:.2f}\pm{tail[1]*1e4:.2f})\times10^{{-4}}"
    paired=json.loads((Path(__file__).resolve().parents[1]/'results/lensing_qe_resolution_report/paired.json').read_text())
    shifts=[abs(c['fractional_change']['value']) for row in paired.values()
            for band in row['response'] for c in band['response']]
    out['qeRefinementShift']=f'{100*max(shifts):.1f}\\%'
    if sky_summary is None:
        sky_summary=Path(__file__).resolve().parents[1]/'results/lensing_qe_sky_report/summary.json'
    sky=json.loads(Path(sky_summary).read_text())
    for proj,row in sky.items():
        label=proj.removeprefix('Proj')
        response=row['response'][0]
        for i,kind in enumerate(('Gradient','Curl')):
            val=response['spherical_over_flat_diagonal'][i][i]
            err=response['SE'][i][i]
            out[f'qeSky{label}{kind}Response']=rf'{val:.3f}\pm{err:.3f}'
            lo,hi=response['indicative_t95_interval'][i][i]
            out[f'qeSky{label}{kind}ResponseInterval']=rf'{val:.3f}\;[{lo:.3f},\,{hi:.3f}]'
    out['qeSkySeeds']=str(sky['ProjSFL']['seeds'])
    return out
