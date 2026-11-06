"""Gate new projected/flat MC means against the existing 20-sky field suite.

Input means are checked in wide ell ranges; the approximate comparison
uncertainty includes both independent ensembles' measured mean variances.
This is a pilot smoke gate, not a test of individual-bin covariance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import spectra as sp
from projection_covariance import PROJS
from validate_fine_spectra import read_matrix, decouple_many


def load_native(directories):
    result = {}
    for proj in PROJS:
        paths = sorted(path for directory in directories
                       for path in directory.glob(f'pseudo_{proj}_seed*_mask0.npz'))
        raw = [np.load(path) for path in paths]
        counts = raw[0]['counts']
        edges = np.arange(0, 6001, 50)
        for spin, key in ((0, 'tt'), (2, 'pol')):
            matrix, old_edges = read_matrix(sp.ROOT/'results/highell/field_coupling', proj, 0, spin)
            np.testing.assert_array_equal(old_edges, edges)
            cl, _ = decouple_many(matrix, sp.rebin_pseudo(np.array([r[key] for r in raw]), counts, edges))
            for spec, index in ([('tt', 0)] if spin == 0 else [('ee', 0), ('bb', 3)]):
                result[proj, spec] = cl[:, index]
        for r in raw:
            r.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    ref = load_native([sp.ROOT/'results/highell/fine_field_pseudo_spectra'])
    new = load_native(args.directories)
    rows = []
    for (proj, spec), arr in new.items():
        for lo, hi in ((500, 2000), (2000, 3500), (3500, 4800)):
            sel = (sp.ELL >= lo) & (sp.ELL < hi)
            # Divide each bin by the reference mean before averaging, to
            # avoid letting the low-ell side dominate a falling spectrum.
            mean = ref[proj, spec][:, sel].mean(0)
            a = (arr[:, sel]/mean).mean(1)
            b = (ref[proj, spec][:, sel]/mean).mean(1)
            se = np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
            row = dict(projection=proj, spectrum=spec, ell_lo=lo, ell_hi=hi,
                       nseed=len(a), ratio=float(a.mean()),
                       se=float(se), delta_sigma=float((a.mean()-1)/se))
            rows.append(row)
            print(row)
    args.out.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
