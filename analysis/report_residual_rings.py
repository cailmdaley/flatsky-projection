"""Report the paired synthesis-ring and finite-subsample interventions."""
from pathlib import Path
import json
import numpy as np
import spectra as S

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')


def main():
    rows = []
    ref = np.load(ROOT/'pipeline_windows_production.npz')
    for proj in ('ProjSTG', 'ProjZEA'):
        ds = [np.load(S.ROOT/'results/highell/residual_ring_test'/f'rings_{proj}_seed{s:04d}.npz') for s in range(2)]
        native = np.array([d['n8192_r8'] for d in ds])
        for spec, index in [('EE', 0), ('BB', 3)]:
            np.testing.assert_allclose(native[:, index], ref[f'field_{spec}_{proj}_raw'][:2]*S.flat_pixwin(),
                                       rtol=2e-9, atol=1e-25)
        print(proj, 'native roundtrips passed', flush=True)
        for lo, hi in ((100, 500), (500, 1000), (1000, 4000), (4000, 4500), (4500, 4800)):
            sel = (S.ELL >= lo) & (S.ELL < hi)
            normalization = S.theory('unlensed_EE')*S.flat_pixwin()
            for name in ('n8192_r8', 'n4096_r8', 'n4096_r16'):
                value = np.array([d[name] for d in ds])
                bb = (value[:, 3, sel]/normalization[sel]).mean(-1)
                ee = (value[:, 0, sel]/native[:, 0, sel]).mean(-1)
                leakage = (value[:, 3, sel]/value[:, 0, sel]).mean(-1)
                baseline = (native[:, 3, sel]/native[:, 0, sel]).mean(-1)
                row = dict(projection=proj, intervention=name, lo=lo, hi=hi, nseed=len(ds),
                           bb_input_ee=float(bb.mean()), bb_input_ee_se=float(bb.std(ddof=1)/np.sqrt(len(bb))),
                           ee_native_ratio=float(ee.mean()), ee_native_ratio_se=float(ee.std(ddof=1)/np.sqrt(len(ee))),
                           bb_own_ee=float(leakage.mean()), bb_own_ee_se=float(leakage.std(ddof=1)/np.sqrt(len(leakage))),
                           paired_bb_ee_change=float((leakage-baseline).mean()),
                           paired_bb_ee_change_se=float((leakage-baseline).std(ddof=1)/np.sqrt(len(leakage))))
                rows.append(row)
                print(f'{lo}-{hi} {name}: BB/inputEE {bb.mean():.7g} +/- {row["bb_input_ee_se"]:.2g}; EE/native {ee.mean():.7g}; BB/ownEE {leakage.mean():.7g}; paired BB/EE change {row["paired_bb_ee_change"]:+.7g} +/- {row["paired_bb_ee_change_se"]:.2g}', flush=True)
        for d in ds:
            d.close()
    (ROOT/'residual_ring_summary.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
