"""The scalar leg against the coupling matrix: measured closure and diagonal-transfer errors.

Twenty CMB TT skies and twenty independent unit-C_l white skies pass through
the scalar leg (`scalar_bandpowers`) on the sweep's reference footprint,
read from the sweep, and on the 1500d field, read alone; the reference
footprint is the 1500d field. Per footprint, sky, projection and band this compares the measured
bandpowers, simulation windows removed, with the scalar block of the forward
model (`shear_leakage.coupling_matrix`, TT<-TT) applied to the known input:

  ratio                measured over forward model, for both skies.
  diagonal_white       the white bandpower recovered by a diagonal transfer
                       function calibrated on the TT skies, F_b = measured TT /
                       input TT, over the white truth; `predicted_diagonal_white`
                       the same from the forward model alone.
  full_white           measured white over R applied to the white input: a
                       known-input closure of the full coupling matrix. It
                       does not invert a noisy, finite-band matrix.

`full_mean_sem` and `diagonal_mean_sem` are the standard errors of the mean of
those curves over 1000 <= l < 4000, with seed influences keeping the
covariance between bands of one realization. This output feeds
`scalar_numbers` and `fig_transfer_error`.

Run:  bash env.sh python3 analysis/scalar_summary.py
          --scalar-bandpowers results/fiducial/scalar_bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra results/fiducial/theory_spectra
          --shear-moments results/fiducial/shear_moments --out <dir>
Out:  <out>/scalar_summary.npz and <out>/scalar_summary.csv. Arrays are
      (footprint, sky, projection, seed, band), footprints reference then field,
      skies TT then white.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import shear_leakage as SL
import spectra as S
import theory as T

REFERENCE = S.FIELD
CLEAN = (1000, 4000)


def scalar_forward(run, mom, mask, j, ell, lp, inputs):
    """Apply the scalar block of the coupling matrix without unused spin blocks."""
    s1, s2, weight = SL._cells(mom, mask, j)
    chi = (np.arange(128) + .5) * np.pi / 128
    stretch = np.sqrt((s1[:, None] * np.cos(chi))**2
                      + (s2[:, None] * np.sin(chi))**2)
    scale = weight * (s1 * s2)**2 / np.sum(weight * s1 * s2)
    values = np.zeros((len(inputs), len(ell)))
    for start in range(0, len(ell), 8):
        source = stretch[:, :, None] * ell[start:start + 8]
        for k, cl in enumerate(inputs):
            mapped = np.interp(source, lp, np.where(lp <= run.lmax, cl, 0.))
            values[k, start:start + 8] = np.sum(
                mapped.mean(1) * scale[:, None], axis=0)
    return values


def check_scalar_forward(run, mom, lp, inputs):
    """Gate on agreement with the full matrix, including its edge."""
    ell = np.array([1025., 3025., 5725.])
    for j in range(len(S.PROJS)):
        direct = scalar_forward(run, mom, 0, j, ell, lp, inputs)
        matrix = SL.coupling_matrix(mom, 0, j, ell, lp, band_limit=run.lmax)['TT<-TT']
        expected = np.stack([matrix @ cl for cl in inputs])
        np.testing.assert_allclose(direct, expected, rtol=2e-12, atol=1e-14)
    print('Scalar integration agrees with the full coupling matrix.')


def compute(run):
    th = T.load(run.theory_spectra)
    lp = th['ell'].astype(float)
    inputs = np.stack([th['unlensed_TT'], np.where(lp >= 2, 1., 0.)])
    reference = S.SWEEP.index(REFERENCE)
    check_scalar_forward(run, run.moments([REFERENCE]), lp, inputs)
    # Use the production integer-multipole average on [lo, hi).
    fine = np.arange(run.lmax, dtype=float)
    truth = np.stack([np.interp(fine, lp, cl).reshape(run.nbins, run.delta_l).mean(1)
                      for cl in inputs])
    np.testing.assert_allclose(truth[0], run.theory('unlensed_TT'), rtol=1e-14)
    np.testing.assert_allclose(truth[1, 0], (run.delta_l - 2) / run.delta_l)
    measured, predictions = [], []
    for field in (False, True):
        mom = run.field_moments() if field else run.moments([REFERENCE])
        measured.append(np.stack([run.load_scalar(spec, field=field) if field
                                  else run.load_scalar(spec)[reference]
                                  for spec in ('TT', 'white')]))
        pred = []
        for j in range(len(S.PROJS)):
            values = scalar_forward(run, mom, 0, j, fine, lp, inputs)
            pred.append(values.reshape(2, run.nbins, run.delta_l).mean(2))
        predictions.append(np.stack(pred, axis=1))
    # Axes: footprint, sky(TT/white), projection, seed, band.
    measured = np.stack(measured)
    predictions = np.stack(predictions)
    mean = measured.mean(axis=3)
    sem = measured.std(axis=3, ddof=1) / np.sqrt(run.nsims)
    closure = mean / predictions
    # Independent realization ensembles: their sample-mean errors add.
    diagonal_white = mean[:, 1] / (mean[:, 0] / truth[0]) / truth[1]
    diagonal_sem = diagonal_white * np.sqrt(
        (sem[:, 1] / mean[:, 1])**2 + (sem[:, 0] / mean[:, 0])**2)
    predicted_diagonal = predictions[:, 1] / (predictions[:, 0] / truth[0]) / truth[1]
    # Measured white over R applied to the white input: a known-input closure.
    full_white = closure[:, 1]
    full_sem = sem[:, 1] / predictions[:, 1]
    clean = (run.ell >= CLEAN[0]) & (run.ell < CLEAN[1])
    # Seed influences retain covariance between bands of the same realization.
    summary_sems = {}
    for name, curve in (('diagonal', diagonal_white), ('full', full_white)):
        iw = ((measured[:, 1] / mean[:, 1, :, None, :] - 1)
              * curve[:, :, None, :])[..., clean].mean(-1)
        it = -((measured[:, 0] / mean[:, 0, :, None, :] - 1)
               * curve[:, :, None, :])[..., clean].mean(-1)
        calibration = it.var(-1, ddof=1) if name == 'diagonal' else 0.
        summary_sems[name] = np.sqrt((iw.var(-1, ddof=1) + calibration) / run.nsims)
    return dict(ell=run.ell, projections=S.PROJS, footprints=['reference', 'field1500d'],
                spectra=['TT', 'white'], measured=measured, mean=mean, sem=sem,
                predicted=predictions, ratio=closure, ratio_sem=sem / predictions,
                diagonal_white=diagonal_white, diagonal_white_sem=diagonal_sem,
                predicted_diagonal_white=predicted_diagonal, truth=truth,
                full_white=full_white, full_white_sem=full_sem,
                full_mean_sem=summary_sems['full'],
                diagonal_mean_sem=summary_sems['diagonal'], summary_range=list(CLEAN))


def write_csv(d, path):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['footprint', 'projection', 'spectrum', 'ell', 'measured',
                         'sem', 'forward_model', 'measured_over_forward', 'ratio_sem',
                         'white_after_tt_diagonal', 'diagonal_sem', 'predicted_diagonal',
                         'white_over_forward', 'full_sem'])
        for fi, footprint in enumerate(d['footprints']):
            for j, proj in enumerate(d['projections']):
                for k, sky in enumerate(d['spectra']):
                    for b, ell in enumerate(d['ell']):
                        writer.writerow([
                            footprint, proj, sky, ell, d['mean'][fi, k, j, b],
                            d['sem'][fi, k, j, b], d['predicted'][fi, k, j, b],
                            d['ratio'][fi, k, j, b], d['ratio_sem'][fi, k, j, b],
                            d['diagonal_white'][fi, j, b] if k else '',
                            d['diagonal_white_sem'][fi, j, b] if k else '',
                            d['predicted_diagonal_white'][fi, j, b] if k else '',
                            d['full_white'][fi, j, b] if k else '',
                            d['full_white_sem'][fi, j, b] if k else ''])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'scalar_bandpowers', 'simulation_windows', 'theory_spectra', 'shear_moments')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    d = compute(S.Run.from_args(args))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'scalar_summary.npz', **d)
    write_csv(d, args.out / 'scalar_summary.csv')
    print(f"wrote {args.out / 'scalar_summary.npz'} and scalar_summary.csv")


if __name__ == '__main__':
    main()
