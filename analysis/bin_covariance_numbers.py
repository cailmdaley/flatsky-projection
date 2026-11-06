"""Macro dictionary for the final, matched 260/256-sky covariance comparison.

Only analysis/paper_numbers.py writes the LaTeX file. This module refuses
partial ensembles and the old production20, which have different input spectra.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/highell/bin_covariance_comparison/comparison.json'


def numbers(source=SOURCE, prediction=ROOT/'results/highell/bin_covariance_local_prediction/prediction.json'):
    data = json.loads(Path(source).read_text())
    for proj in ('ProjZEA', 'ProjSFL', 'ProjSTG'):
        for ensemble, n in (('projected', 260), ('control', 256)):
            seeds = data['inputs'][ensemble+'_seeds'][proj]
            if sorted(seeds) != list(range(n)):
                raise ValueError(f'{proj}: paper numbers require the complete {n}-sky {ensemble} ensemble')
    out = dict(covNumProjected='260', covNumControl='256', covBinWidth='50')

    def select(kind, proj, spec, lo, hi, width=50, **extra):
        conditions = dict(projection=proj, spectrum=spec, ell_lo=lo, ell_hi=hi,
                          width=width, estimator='decoupled', **extra)
        rows = [r for r in data[kind] if all(r[k] == v for k, v in conditions.items())]
        assert len(rows) == 1, conditions
        return rows[0]

    for proj, spec in (('ProjSFL', 'tt'), ('ProjSFL', 'ee'), ('ProjSFL', 'bb'),
                       ('ProjZEA', 'bb')):
        row = select('variances', proj, spec, 3500, 4800)
        name = 'cov'+proj[4:]+spec.upper()+'HighVariance'
        out[name] = f"{row['fractional_variance_ratio']:.2f}"
        out[name+'Error'] = f"{row['jackknife_se']:.2f}"
    # Keep only the specific comparisons needed by the proposed paper claims.
    # Balkenhol Table III uses width100 above2000, not width50.
    for proj, spec, lo, hi, band, width in (
            ('ProjZEA', 'tt', 2000, 3000, 'SPT', 100),
            ('ProjZEA', 'ee', 2000, 3000, 'SPT', 100),
            ('ProjZEA', 'bb', 3500, 4800, 'High', 50),
            ('ProjSFL', 'bb', 3500, 4800, 'High', 50)):
        row = select('correlations', proj, spec, lo, hi, width=width, separation=width)
        prefix = 'cov'+proj[4:]+spec.upper()+band+'Adjacent'
        for field, label in (('projected', ''), ('excess', 'Excess')):
            out[prefix+label] = f"{row[field+'_rho']:.3f}"
            out[prefix+label+'Error'] = f"{row[field+'_se']:.3f}"
    from summarize_bin_covariance import local_summary
    summary = local_summary(data, json.loads(Path(prediction).read_text()))
    out['covLocalVarianceWorst'] = f"{100*summary['worst_fractional_variance_difference']:.1f}\\%"
    out['covLocalAdjacentWorst'] = f"{summary['worst_adjacent_rho_difference']:.3f}"
    assert all(name.isalpha() for name in out)
    return out
