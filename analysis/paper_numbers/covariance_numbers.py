"""Matched-ensemble bandpower covariance macros, covariance_numbers.tex.

Formats `bin_covariance_comparison` (projected skies against homogeneous
controls), decoupled bandpowers:

  covNumProjected                          the number of projected skies
  covSFL<spec>HighVariance[Error]          SFL's fractional-variance ratio to
                                           the controls over 3500 <= l < 4800,
                                           at the bandpower width
  covZEA<TT|EE>SPTAdjacent[Error]          ZEA's adjacent-bin correlation over
                                           2000 <= l < 3000 at width 100, the
                                           width of Balkenhol et al.'s Table III

Every comparison uses whole ensembles: an incomplete seed set fails.

Run:  bash env.sh python3 analysis/paper_numbers/covariance_numbers.py
          --bin-covariance-comparison <dir> --delta-ell d50 --out <dir>
Out:  <out>/covariance_numbers.tex and <out>/covariance_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from _decisions import parse_delta_ell  # noqa: E402
from macros import Macros  # noqa: E402

PROJS = ('ProjZEA', 'ProjSFL', 'ProjSTG')
# Balkenhol et al. (2018), Table III, bin SPT's spectra in width 100 above 2000.
SPT_WIDTH = 100


def complete(data, ensemble):
    """Size of one ensemble, refusing any chart whose seeds are not 0..n-1."""
    sizes = set()
    for proj in PROJS:
        seeds = data['inputs'][ensemble+'_seeds'][proj]
        if sorted(seeds) != list(range(len(seeds))):
            raise ValueError(f'{proj}: the {ensemble} ensemble is incomplete')
        sizes.add(len(seeds))
    (n,) = sizes
    return n


def macros(data, delta_ell):
    complete(data, 'control')
    M = Macros()
    M.add('covNumProjected', str(complete(data, 'projected')))

    def select(kind, proj, spec, lo, hi, width, **extra):
        conditions = dict(projection=proj, spectrum=spec, ell_lo=lo, ell_hi=hi,
                          width=width, estimator='decoupled', **extra)
        rows = [r for r in data[kind] if all(r[k] == v for k, v in conditions.items())]
        assert len(rows) == 1, conditions
        return rows[0]

    for spec in ('tt', 'ee', 'bb'):
        row = select('variances', 'ProjSFL', spec, 3500, 4800, delta_ell)
        name = 'covSFL'+spec.upper()+'HighVariance'
        M.add(name, f"{row['fractional_variance_ratio']:.2f}")
        M.add(name+'Error', f"{row['jackknife_se']:.2f}")
    for spec in ('tt', 'ee'):
        row = select('correlations', 'ProjZEA', spec, 2000, 3000, SPT_WIDTH, separation=SPT_WIDTH)
        name = 'covZEA'+spec.upper()+'SPTAdjacent'
        M.add(name, f"{row['projected_rho']:.3f}")
        M.add(name+'Error', f"{row['projected_se']:.3f}")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'bin_covariance_comparison')
    ap.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.bin_covariance_comparison/'bin_covariance_comparison.json').read_text())
    macros(data, args.delta_ell).write(args.out, 'covariance_numbers')


if __name__ == '__main__':
    main()
