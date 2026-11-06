"""Macro dictionary for the final, matched 260/256-sky covariance comparison.

Only analysis/paper_numbers.py writes the LaTeX file. This module refuses
partial ensembles and the old production20, which have different input spectra.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/highell/bin_covariance_comparison/comparison.json'


def numbers(source=SOURCE):
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
    for proj in ('ProjZEA', 'ProjSFL'):
        for spec in ('tt', 'ee', 'bb'):
            for lo, hi, band, width in ((2000, 3000, 'SPT', 100), (3500, 4800, 'High', 50)):
                # Balkenhol Table III uses width100 above2000, not width50.
                for separation, lag in ((width, 'Adjacent'), (2*width, 'Second')):
                    row = select('correlations', proj, spec, lo, hi, width=width,
                                 separation=separation)
                    prefix = 'cov'+proj[4:]+spec.upper()+band+lag
                    for field, label in (('projected', ''), ('control', 'Control'), ('excess', 'Excess')):
                        out[prefix+label] = f"{row[field+'_rho']:.3f}"
                        out[prefix+label+'Error'] = f"{row[field+'_se']:.3f}"
    assert all(name.isalpha() for name in out)
    return out
