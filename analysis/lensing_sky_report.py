"""TT response overlaps for physical scalar lenses shared between charts.

The template is W^2 times spherical kappa, projected as a scalar. Each
response is calibrated by the same template overlap of the same-mask flat
control driven by the chart-projected physical displacement. It is not a
cosmological power-spectrum calibration, nor an intrinsic chart response.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from lensing_report import ratio_error
from paper_style import PROJS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=Path('results/highell/lensing_qe_sky_response'))
    p.add_argument('--out',type=Path,default=Path('results/lensing_qe_sky_report/summary.json'))
    a=p.parse_args()
    result={}; reference={}
    for proj in PROJS:
        files=sorted(a.inputs.glob(f'sky_qe_{proj}_*.npz'))
        assert len(files)==4,(proj,len(files))
        data=[np.load(f) for f in files]
        for seed,v in enumerate(data):
            mode=np.concatenate([v['lens_k'],v['lens_direction'].ravel(),v['lens_phase'],v['lens_amplitude']])
            if seed in reference:
                np.testing.assert_array_equal(mode,reference[seed])
            else:
                reference[seed]=mode
        x,y=np.meshgrid(data[0]['lx'],data[0]['ly'])
        ls=np.hypot(x,y)
        rows=[]
        for lo,hi in ((30,60),(60,120),(120,240)):
            keep=(ls>=lo)&(ls<hi)
            spherical=[]; flat=[]
            for label in ('gradient','curl'):
                spherical.append(np.array([[np.vdot(v['input_template'][keep],v[label][i,keep]).real for i in range(2)] for v in data]))
                flat.append(np.array([[np.vdot(v['input_template'][keep],v['flat_'+label][i,keep]).real for i in range(2)] for v in data]))
            matrix=[]; errors=[]; flat_matrix=[]
            for i in range(2):
                estimates=[ratio_error(spherical[i][:,j],flat[i][:,i]) for j in range(2)]
                matrix.append([v[0] for v in estimates]); errors.append([v[1] for v in estimates])
                flat_matrix.append([ratio_error(flat[i][:,j],flat[i][:,i])[0] for j in range(2)])
            rows.append({'L':[lo,hi],'spherical_over_flat_diagonal':matrix,'SE':errors,
                         'flat_over_its_diagonal':flat_matrix})
        result[proj]={'seeds':4,'response':rows,
                      'scope':'Noise-free TT legs 300-1200; shared physical spherical potentials vary with seed; errors resample TT+lens pairs. Rows=injected spherical gradient/curl, columns=reconstructed grid gradient/curl. Scalar W^2*kappa template; each row divided by the matching flat diagonal. Small signed off-diagonals do not imply negligible map-level mixing.'}
    assert not np.array_equal(reference[0],reference[1])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
