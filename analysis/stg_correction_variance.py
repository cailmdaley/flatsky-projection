"""What correcting STG's magnification costs against ZEA, on the covariance skies.

Three estimators of one amplitude per 500-wide band over 500 < l < 4000, each
from the matched `bin_covariance_sims` skies (noise-free, unlensed, 1500d field):

  equal     the conditional inverse of `choosing_projection.fit_amplitudes`,
            which weights every bin equally in fractional error;
  weighted  the same design with bins weighted by their measured scatter;
  direct    Fourier-cell-weighted bandpowers, with STG's edges moved by its
            mean linear magnification and no inverse.

The mean linear magnification is <sqrt(det J)> of STG over the field under the
squared mask, from the field's shear moments; STG is conformal, so sqrt(det J)
is its local scale factor. The ratio is the band-averaged STG fractional
variance over ZEA's, with a whole-sky bootstrap error. No noise, lensing or
off-diagonal weighting enters, so this bounds the two-point cost of the
correction in those conditions only.

Run:  bash env.sh python3 analysis/stg_correction_variance.py
          --bin-covariance-sims results/fiducial/bin_covariance_sims
          --fine-coupling-matrices results/fiducial/fine_coupling_matrices
          --bandpowers results/fiducial/bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra data/theory_spectra.npz
          --shear-moments results/fiducial/shear_moments --out <dir>
Out:  <out>/stg_correction_variance.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import choosing_projection as CP
import compare_bin_covariance as CB
import shear_leakage as SL
import spectra as S
import theory as T

EDGES = np.arange(500, 4001, 500)
PROJS = ('ProjZEA', 'ProjSTG')


def linear_magnification(mom, proj):
    """Mean sqrt(det J) of one chart over the field, weighted by the squared mask."""
    s1, s2, w = SL._cells(mom, 0, S.PROJS.index(proj))
    return float(np.sum(w * np.sqrt(s1 * s2)) / np.sum(w))


def direct(sims, proj, magnification):
    """Per-sky fractional bandpowers of the unit-width pseudo-spectra."""
    paths = sorted(Path(sims).glob(f'pseudo_{proj}_seed*_mask0.npz'))
    tt, ee = [], []
    for path in paths:
        with np.load(path) as z:
            tt.append(z['tt'][0])
            ee.append(z['pol'][0])
            counts = z['counts']
    scale = 1 / magnification
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


def inverses(run, sims, workspaces, mom):
    """Fitted band amplitudes per sky, for both bin weights, on the bandpowers' bins."""
    arrays, seeds = CB.load_ensemble(sims, workspaces, (run.delta_l,), PROJS)
    assert seeds['ProjZEA'] == seeds['ProjSTG']
    th = T.load(run.theory_spectra)
    lp = th['ell'].astype(float)
    select = (run.ell >= 500) & (run.ell < 4000)
    result = {}
    for proj in PROJS:
        j = S.PROJS.index(proj)
        r = SL.coupling_matrix(mom, 0, j, run.ell, lp, band_limit=run.lmax)
        data = {s: arrays[proj, run.delta_l, s.lower(), 'decoupled'][:, :run.nbins]
                / run.simulation_window(s, field=True)[j]
                for s in ('TT', 'EE', 'BB')}
        joint = np.c_[data['EE'], data['BB']]
        for spec, matrix, fit, shape, spin in (
                ('TT', r['TT<-TT'], data['TT'], th['unlensed_TT'], False),
                ('EE', r, joint, th['unlensed_EE'], True)):
            amplitude, _, _ = CP.fit_amplitudes(matrix, fit, run.bin_theory(shape), select,
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'bin_covariance_sims', 'fine_coupling_matrices', 'bandpowers',
                 'simulation_windows', 'theory_spectra', 'shear_moments')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    run = S.Run.from_args(args)
    mom = run.field_moments()
    magnification = linear_magnification(mom, 'ProjSTG')
    estimates = inverses(run, args.bin_covariance_sims, args.fine_coupling_matrices, mom)
    for proj in PROJS:
        scale = magnification if proj == 'ProjSTG' else 1.
        for spec, value in direct(args.bin_covariance_sims, proj, scale).items():
            estimates[proj, spec, 'direct'] = value
    n = estimates['ProjZEA', 'TT', 'equal'].shape[0]
    draws = np.random.default_rng(2).integers(0, n, (1500, n))
    summary = {'nsky': n, 'bands': EDGES.tolist(), 'stg_linear_magnification': magnification}
    for spec in ('TT', 'EE'):
        for estimator in ('equal', 'weighted', 'direct'):
            summary[f'{spec}_{estimator}'] = ratio(estimates['ProjSTG', spec, estimator],
                                                   estimates['ProjZEA', spec, estimator], draws)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'stg_correction_variance.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
