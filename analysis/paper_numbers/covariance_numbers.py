"""Matched-ensemble bandpower covariance macros, covariance_numbers.tex.

Formats `bin_covariance_comparison` (projected skies against homogeneous
controls) and checks `bin_covariance_local_prediction` against it:

  covNum{Projected,Control}, covBinWidth   the ensembles and the bin width
  cov<proj><spec>HighVariance[Error]       fractional-variance ratio over
                                           3500 <= l < 4800
  cov<proj><spec><SPT|High>Adjacent...     adjacent-bin correlation and its
                                           excess over the controls, at the
                                           width of Balkenhol et al.'s Table III
                                           (100) below 3000 and at the bandpower
                                           width above 3500
  covLocal{Variance,Adjacent}Worst         the worst disagreement of the leading
                                           local prediction with the coupled
                                           Monte Carlo over the two broad bands

Every comparison uses whole ensembles: an incomplete seed set fails.

Run:  bash env.sh python3 analysis/paper_numbers/covariance_numbers.py
          --bin-covariance-comparison <dir> --bin-covariance-local-prediction <dir>
          --delta-ell d50 --out <dir>
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


def local_summary(comparison, prediction, delta_ell, n_projected, n_control):
    """Scoped local predictions against the final coupled Monte Carlo statistics.

    The reported maxima cover only these broad-band summaries, not every matrix
    entry or an exact covariance model. No simulation-tuned correction is applied.
    """
    rows = []
    for lo, hi, width in ((2000, 3000, SPT_WIDTH), (3500, 4800, delta_ell)):
        for proj in PROJS:
            for spec in ('tt', 'ee', 'bb'):
                if proj == 'ProjSTG' and spec == 'bb':
                    continue
                select = dict(projection=proj, spectrum=spec, width=width,
                              estimator='coupled', ell_lo=lo, ell_hi=hi)
                keep = lambda r: all(r[k] == v for k, v in select.items())
                v = next(r for r in comparison['variances'] if keep(r))
                c = next(r for r in comparison['correlations'] if keep(r) and r['separation'] == width)
                assert v['n_projected'] == n_projected and v['n_control'] == n_control
                p = [r for r in prediction['rows'] if r['projection'] == proj
                     and r['spectrum'] == spec and r['width'] == width and lo <= r['ell'] < hi]
                assert len(p) == (hi-lo)//delta_ell
                average = lambda key: sum(r[key] for r in p)/len(p)
                rows.append(dict(**select,
                    predicted_variance_ratio=average('fractional_variance_ratio'),
                    measured_variance_ratio=v['fractional_variance_ratio'],
                    predicted_adjacent_rho=sum(r['projected_rho'][0] for r in p)/len(p),
                    measured_adjacent_rho=c['projected_rho']))
    return dict(
        worst_fractional_variance_difference=max(abs(
            r['predicted_variance_ratio']/r['measured_variance_ratio']-1) for r in rows),
        worst_adjacent_rho_difference=max(abs(
            r['predicted_adjacent_rho']-r['measured_adjacent_rho']) for r in rows))


def macros(data, prediction, delta_ell):
    n_projected, n_control = complete(data, 'projected'), complete(data, 'control')
    M = Macros()
    M.add('covNumProjected', str(n_projected))
    M.add('covNumControl', str(n_control))
    M.add('covBinWidth', str(delta_ell))

    def select(kind, proj, spec, lo, hi, width, **extra):
        conditions = dict(projection=proj, spectrum=spec, ell_lo=lo, ell_hi=hi,
                          width=width, estimator='decoupled', **extra)
        rows = [r for r in data[kind] if all(r[k] == v for k, v in conditions.items())]
        assert len(rows) == 1, conditions
        return rows[0]

    for proj, spec in (('ProjSFL', 'tt'), ('ProjSFL', 'ee'), ('ProjSFL', 'bb'),
                       ('ProjZEA', 'bb')):
        row = select('variances', proj, spec, 3500, 4800, delta_ell)
        name = 'cov'+proj[4:]+spec.upper()+'HighVariance'
        M.add(name, f"{row['fractional_variance_ratio']:.2f}")
        M.add(name+'Error', f"{row['jackknife_se']:.2f}")
    for proj, spec, lo, hi, band, width in (
            ('ProjZEA', 'tt', 2000, 3000, 'SPT', SPT_WIDTH),
            ('ProjZEA', 'ee', 2000, 3000, 'SPT', SPT_WIDTH),
            ('ProjZEA', 'bb', 3500, 4800, 'High', delta_ell),
            ('ProjSFL', 'bb', 3500, 4800, 'High', delta_ell)):
        row = select('correlations', proj, spec, lo, hi, width, separation=width)
        prefix = 'cov'+proj[4:]+spec.upper()+band+'Adjacent'
        for field, label in (('projected', ''), ('excess', 'Excess')):
            M.add(prefix+label, f"{row[field+'_rho']:.3f}")
            M.add(prefix+label+'Error', f"{row[field+'_se']:.3f}")
    summary = local_summary(data, prediction, delta_ell, n_projected, n_control)
    M.add('covLocalVarianceWorst', f"{100*summary['worst_fractional_variance_difference']:.1f}\\%")
    M.add('covLocalAdjacentWorst', f"{summary['worst_adjacent_rho_difference']:.3f}")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'bin_covariance_comparison', 'bin_covariance_local_prediction')
    ap.add_argument('--delta-ell', type=parse_delta_ell, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.bin_covariance_comparison/'bin_covariance_comparison.json').read_text())
    prediction = json.loads((args.bin_covariance_local_prediction
                             / 'bin_covariance_local_prediction.json').read_text())
    macros(data, prediction, args.delta_ell).write(args.out, 'covariance_numbers')


if __name__ == '__main__':
    main()
