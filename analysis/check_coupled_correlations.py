"""Compare coupled and mask-decoupled correlations for the same field skies."""
from pathlib import Path
import json
import numpy as np
import spectra as sp
from projection_covariance import PROJS, jackknife_lag
from validate_fine_spectra import read_pseudos

OUT = sp.ROOT/'results/highell/bin_correlation_diagnostic'
rows = []
for proj in PROJS:
    pol, tt, counts = read_pseudos(sp.ROOT/'results/highell/fine_field_pseudo_spectra', proj, 0)
    for width in (30, 50, 100):
        edges = np.arange(0, 6001, width)
        ell = (edges[:-1]+edges[1:])/2
        pp = sp.rebin_pseudo(pol, counts, edges)
        pt = sp.rebin_pseudo(tt, counts, edges)
        for spec, data in [('tt', pt[:, 0]), ('ee', pp[:, 0]), ('bb', pp[:, 3])]:
            for lo, hi in ((500, 2000), (2000, 3500), (3500, 4800)):
                for lag in (1, 2, 4):
                    row = dict(projection=proj, spectrum=spec, width=width,
                               ell_lo=lo, ell_hi=hi, separation=lag*width,
                               **jackknife_lag(data, ell, lo, hi, lag))
                    rows.append(row)
                    if width == 50 and lo == 3500 and lag == 1:
                        print('COUPLED', row)
(OUT/'coupled_measured.json').write_text(json.dumps(rows, indent=2)+'\n')
