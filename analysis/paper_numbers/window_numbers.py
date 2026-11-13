"""Window-control and corrected-residual macros.

From the projector and interpolator controls (`pipeline_window_controls`),
the window-corrected production residuals of the 1500d field
(`pipeline_windows`) and the sky band-limit intervention
(`pipeline_bandlimit_summary`):

  winRing{East,North}          interpolation power at l = L_WALL along each
                               axis at the patch centre's declination
  win{Projector,RingControl}Error, winSpinControlErrorDeg
                               the controls' worst deviations
  win<spec><proj><band>...     per-seed and ZEA-paired residuals, with and
                               without aliases, alias fractions, and for BB
                               the STG-floor-subtracted residual
  zeaFloorField<band>          ZEA's BB excess in units of the STG floor
  winWhite..., winFieldWall... the white sky's alias and Nyquist residuals
                               and the window correction at the wall
  winCutoff<spec><proj><band>  power retained and removed by the cutoff
                               change, and the predicted removal

The interpolation window takes the synthesis Nside from the masks' index files
and the declination from `_footprints`.

Run:  bash env.sh python3 analysis/paper_numbers/window_numbers.py
          --pipeline-windows <dir> --pipeline-window-controls <dir>
          --pipeline-bandlimit-summary <dir>
          --healpix-masks results/fiducial/healpix_masks
          --flatsky-masks results/fiducial/flatsky_masks --out <dir>
Out:  <out>/window_numbers.tex and <out>/window_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from _footprints import DEC_DEG  # noqa: E402
from macros import Macros, f, pct  # noqa: E402
from pipeline_windows import Sampling, interpolation_power  # noqa: E402

L_WALL = 4800.


def window_numbers(d, controls, cutoff, sampling):
    """Window-control and corrected-residual macros from checked outputs."""
    control, ring, spin = controls['projector'], controls['ring'], controls['spin']
    M = Macros()
    dec = np.deg2rad(DEC_DEG)
    M.add('winRingEast', f(float(interpolation_power(L_WALL, 0., dec, sampling.nside)), 4))
    M.add('winRingNorth', f(float(interpolation_power(0., L_WALL, dec, sampling.nside)), 4))
    error = max(abs(r['ratio'] - 1) for r in control if r['interp'] and r['rebin'] == 8)
    M.add('winProjectorError', pct(error, 4))
    M.add('winRingControlError', pct(max(abs(r['residual']) for r in ring), 4))
    M.add('winSpinControlErrorDeg', f(max(abs(r['spin_error_deg']) for r in spin), 5))
    ell = d['ell']
    band_names = [('Clean', 1000, 4000), ('Upper', 4000, 4500), ('Wall', 4500, 4800)]
    for spec in ('TT', 'EE', 'BB'):
        residuals = []
        for proj in S.PROJS:
            if spec == 'BB' and proj == 'ProjSTG':
                continue
            r = d[f'field_{spec}_{proj}_ratio']
            ref = d[f'field_{spec}_ProjZEA_ratio']
            for name, lo, hi in band_names:
                sel = (ell >= lo) & (ell < hi)
                per_seed = (r[:, sel] - 1).mean(-1)
                paired = (r[:, sel] - ref[:, sel]).mean(-1)
                prefix = 'win' + spec + proj[4:] + name
                M.add(prefix + 'Residual', pct(per_seed.mean(), 3))
                M.add(prefix + 'Error', pct(per_seed.std(ddof=1) / np.sqrt(len(per_seed)), 3))
                M.add(prefix + 'Paired', pct(paired.mean(), 3))
                M.add(prefix + 'PairedError', pct(paired.std(ddof=1) / np.sqrt(len(paired)), 3))
                total = d[f'field_{spec}_{proj}_ratio_total'][:, sel].mean(-1) - 1
                M.add(prefix + 'TotalResidual', pct(total.mean(), 3))
                M.add(prefix + 'TotalError', pct(total.std(ddof=1) / np.sqrt(len(total)), 3))
                alias = d[f'field_{spec}_{proj}_alias'][sel]
                windowed = d[f'field_{spec}_{proj}_windowed'][sel]
                M.add(prefix + 'AliasFraction', pct(np.mean(alias / windowed), 3))
                if spec == 'BB':
                    sub = d[f'field_{spec}_{proj}_ratio_floor_subtracted'][:, sel].mean(-1) - 1
                    M.add(prefix + 'FloorSubResidual', pct(sub.mean(), 3))
                    M.add(prefix + 'FloorSubError', pct(sub.std(ddof=1) / np.sqrt(len(sub)), 3))
                if name == 'Clean':
                    residuals.append(abs(per_seed.mean()))
        M.add('win' + spec + 'CleanMaxResidual', pct(max(residuals), 3))
    for name, lo, hi in band_names[1:]:
        sel = (ell >= lo) & (ell < hi)
        total = d['field_BB_ProjZEA_ratio_total'][:, sel].mean()
        floor = total - d['field_BB_ProjZEA_ratio_floor_subtracted'][:, sel].mean()
        M.add('zeaFloorField' + name, f((total - 1) / floor, 1))
    M.add('winWhiteAliasWorst', pct(max(abs(d[f'field_white_{proj}_ratio_total'][:, (ell >= lo) & (ell < hi)].mean() - 1)
                                        for proj in S.PROJS for _, lo, hi in band_names[1:]), 2))
    k = np.argmin(abs(ell - 4775))
    corrections = np.concatenate([d[f'field_{s}_correction'][:, k] for s in ('TT', 'EE')])
    M.add('winFieldWallCorrectionMin', f(corrections.min(), 4))
    M.add('winFieldWallCorrectionMax', f(corrections.max(), 4))
    for proj in S.PROJS:
        sel = (ell >= 1000) & (ell < 4000)
        a = d[f'field_white_{proj}_ratio']
        b = d[f'field_TT_{proj}_ratio']
        per_seed = (a[:, sel] - b[:, sel]).mean(-1)
        M.add('winWhiteMinusTT' + proj[4:], pct(per_seed.mean(), 3))
        M.add('winWhiteMinusTT' + proj[4:] + 'Error', pct(per_seed.std(ddof=1) / np.sqrt(len(per_seed)), 3))
    for proj in S.PROJS:
        sel = (ell >= 4800) & (ell < 5400)
        r = d[f'field_white_{proj}_ratio_total'][:, sel].mean(-1) - 1
        M.add('winWhite' + proj[4:] + 'NyquistResidual', pct(r.mean(), 3))
        M.add('winWhite' + proj[4:] + 'NyquistError', pct(r.std(ddof=1) / np.sqrt(len(r)), 3))
        for spec in ('EE', 'BB'):
            for name, lo, hi in (('Upper', 4000, 4500), ('Wall', 4500, 4800), ('Nyquist', 4800, 5400)):
                value = cutoff[f'{proj}_{spec}_{lo}_{hi}']['retained']
                row = cutoff[f'{proj}_{spec}_{lo}_{hi}']
                prefix = 'winCutoff' + spec + proj[4:] + name
                M.add(prefix + 'Retained', pct(value, 3))
                M.add(prefix + 'Removed', pct(row['delta_over_upper'], 1))
                M.add(prefix + 'Predicted', pct(row['predicted_delta_over_upper'], 1))
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, 'pipeline_windows', 'pipeline_window_controls', 'pipeline_bandlimit_summary',
                 'healpix_masks', 'flatsky_masks')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    M = window_numbers(
        np.load(args.pipeline_windows / 'pipeline_windows.npz'),
        json.loads((args.pipeline_window_controls / 'pipeline_window_controls.json').read_text()),
        json.loads((args.pipeline_bandlimit_summary / 'pipeline_bandlimit_summary.json').read_text()),
        Sampling.read(args.healpix_masks, args.flatsky_masks))
    M.write(args.out, 'window_numbers')


if __name__ == "__main__":
    main()
