"""What correcting STG's magnification costs against ZEA, on 260 skies.

Three estimators of one amplitude per 500-wide band over 500 < l < 4000, each
from the matched `bin_covariance` skies (noise-free, unlensed, 1500d field):

  equal     the conditional inverse of `choosing_projection.fit_amplitudes`,
            which weights every bin equally in fractional error;
  weighted  the same design with bins weighted by their measured scatter;
  direct    Fourier-cell-weighted bandpowers, with STG's edges moved by its
            mean linear magnification and no inverse.

The ratio is the band-averaged STG fractional variance over ZEA's, with a
whole-sky bootstrap error. No noise, lensing or off-diagonal weighting enters,
so this bounds the two-point cost of the correction in those conditions only.

Run:  bash env.sh python3 analysis/stg_correction_variance.py
Out:  analysis/figures/stg_correction_variance.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import choosing_projection as CP
import compare_bin_covariance as CB
import shear_leakage as SL
import spectra as S
import theory as T
from field_1500d import pipeline_moments

OUT = Path(__file__).resolve().parent / 'figures' / 'stg_correction_variance.json'
DIRS = [S.ROOT / 'results/highell/bin_covariance_pilot', S.ROOT / 'results/highell/bin_covariance_sims']
EDGES = np.arange(500, 4001, 500)
STG_SCALE = 1 / 1.0222
PROJS = ('ProjZEA', 'ProjSTG')


def direct(proj):
    """Per-sky fractional bandpowers of the unit-width pseudo-spectra."""
    paths = sorted(p for d in DIRS for p in d.glob(f'pseudo_{proj}_seed*_mask0.npz'))
    tt, ee = [], []
    for path in paths:
        with np.load(path) as z:
            tt.append(z['tt'][0])
            ee.append(z['pol'][0])
            counts = z['counts']
    scale = STG_SCALE if proj == 'ProjSTG' else 1.
    index = np.arange(len(counts))

    def bands(a):
        a = np.array(a)
        out = []
        for lo in EDGES[:-1]:
            s = (index >= lo * scale) & (index < (lo + 500) * scale)
            out.append((a[:, s] * counts[s]).sum(1) / counts[s].sum())
        out = np.stack(out, 1)
        return out / out.mean(0)

    return {'TT': bands(tt), 'EE': bands(ee)}


def inverses():
    """Fitted band amplitudes per sky, for both bin weights."""
    CB.PROJS = PROJS
    arrays, seeds = CB.load_ensemble(DIRS, (50,))
    assert seeds['ProjZEA'] == seeds['ProjSTG']
    th = T.load()
    lp = th['ell'].astype(float)
    mom = pipeline_moments(nside=256)
    select = (S.ELL >= 500) & (S.ELL < 4000)
    result = {}
    for proj in PROJS:
        j = S.PROJS.index(proj)
        r = SL.coupling_matrix(mom, 0, j, S.ELL, lp, band_limit=S.LMAX)
        data = {s: arrays[proj, 50, s.lower(), 'decoupled'][:, :120] / S.simulation_window(s, field=True)[j]
                for s in ('TT', 'EE', 'BB')}
        joint = np.c_[data['EE'], data['BB']]
        for spec, matrix, fit, shape, spin in (
                ('TT', r['TT<-TT'], data['TT'], th['unlensed_TT'], False),
                ('EE', r, joint, th['unlensed_EE'], True)):
            amplitude, _, _ = CP.fit_amplitudes(matrix, fit, S.bin_theory(shape), select,
                                                EDGES, lp, shape, spin)
            result[proj, spec, 'equal'] = amplitude[:, :len(EDGES) - 1]
        for spec, block in (('TT', 'TT<-TT'), ('EE', 'EE<-EE')):
            shape = th[f'unlensed_{spec}']
            basis = np.array([shape * ((lp >= a) & (lp < b))
                              for a, b in zip(EDGES[:-1], EDGES[1:])]).T
            design = (r[block] @ basis)[select]
            scatter = data[spec][:, select].std(0, ddof=1)
            inverse = np.linalg.pinv(design / scatter[:, None])
            result[proj, spec, 'weighted'] = ((data[spec] - data[spec].mean(0))[:, select] / scatter) @ inverse.T
    return result


def ratio(stg, zea, draws):
    value = stg.var(0, ddof=1).mean() / zea.var(0, ddof=1).mean()
    boot = [stg[i].var(0, ddof=1).mean() / zea[i].var(0, ddof=1).mean() for i in draws]
    return {'ratio': float(value), 'error': float(np.std(boot))}


def main():
    estimates = inverses()
    for proj in PROJS:
        for spec, value in direct(proj).items():
            estimates[proj, spec, 'direct'] = value
    n = estimates['ProjZEA', 'TT', 'equal'].shape[0]
    assert n == 260
    draws = np.random.default_rng(2).integers(0, n, (1500, n))
    summary = {'nsky': n, 'bands': EDGES.tolist()}
    for spec in ('TT', 'EE'):
        for estimator in ('equal', 'weighted', 'direct'):
            summary[f'{spec}_{estimator}'] = ratio(estimates['ProjSTG', spec, estimator],
                                                   estimates['ProjZEA', spec, estimator], draws)
    OUT.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
