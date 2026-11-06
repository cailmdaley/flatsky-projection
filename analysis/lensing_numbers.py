"""Generated TT-QE numbers for import by analysis/paper_numbers.py.

Run analysis/lensing_mean_prediction.py and analysis/lensing_report.py first;
see notes/cartographic-lensing-reconstruction.md for the scope of the test.
This module does not write numbers.tex: paper_numbers.py remains its writer.
"""
import json
from pathlib import Path


def numbers(summary=None):
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
        power,error=row['low_L_mean_field_power'][1]
        if power>0 and power>3*error:
            rms=power**.5
            out[f'qe{label}CurlMeanRMS']=rf'{rms:.4f}\pm{error/(2*rms):.4f}'
    out['qeSeeds']=str(data['ProjSFL']['seeds'])
    return out
