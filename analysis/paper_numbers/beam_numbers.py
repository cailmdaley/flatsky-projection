"""Measured beam macros: the beam reprojection experiment against its model.

Formats the `beam_summary` output, per footprint (Disk: the 5% disk, Field:
the 1500d field):

  beamMeasuredEll                             the multipole of the single-bin values
  beamMeasured<fp>{Difference,Prediction}     SFL/sphere response - 1 on ZEA,
                                              measured and predicted, at that bin
  beamMeasured<fp>{SFL,Sphere}                the two responses there
  beamMeasured<fp>Sector{Min,Max}             range of the SFL response over direction
  beamMeasured<fp>ModelResidual               largest sector residual against the
                                              model, percentage points, 1000-3000
  beamMeasured<fp>QuadrupoleSlope             measured on predicted cos 2chi amplitude
  beamMeasured<fp><Low|Mid|High>{Difference,Prediction}  band means
  beamMeasured<fp><ZEA|STG><term>             BB/EE of the unfiltered SFL - sphere
                                              Mismatch, the Differential filtered
                                              map, their cross (CrossMismatch)
                                              and the change in mismatch power
                                              (MismatchChange), 1000-3000
  beamBandAgreement                           largest measured-predicted gap, 1000-3000

Run:  bash env.sh python3 analysis/paper_numbers/beam_numbers.py
          --beam-summary <beam_summary dir> --out <dir>
Out:  <out>/beam_numbers.tex and <out>/beam_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros, f, pct, sci  # noqa: E402


def macros(data):
    M = Macros()
    M.add('beamMeasuredEll', f(data['ell_response'], 0))
    for footprint, name in (('disk5', 'Disk'), ('field1500d', 'Field')):
        response = data['response'][footprint]
        prefix = 'beamMeasured' + name
        M.add(prefix + 'Difference', pct(response['fractional_difference'], 3))
        M.add(prefix + 'Prediction', pct(response['prediction'], 3))
        M.add(prefix + 'SFL', f(response['sfl_mean'], 3))
        M.add(prefix + 'Sphere', f(response['sphere_mean'], 3))
        M.add(prefix + 'SectorMin', f(response['sfl_sector_min'], 3))
        M.add(prefix + 'SectorMax', f(response['sfl_sector_max'], 3))
        # Absolute response residual, in percentage points (not percent of response).
        M.add(prefix + 'ModelResidual', f(100*response['sector_model_max_abs_1000_3000'], 3))
        M.add(prefix + 'QuadrupoleSlope', f(response['quadrupole_amplitude_slope_1000_3000'], 3))
        for band, label in (('1000_2000', 'Low'), ('2000_3000', 'Mid'), ('3000_4000', 'High')):
            M.add(prefix + label + 'Difference', pct(response['bands'][band]['measured'], 3))
            M.add(prefix + label + 'Prediction', pct(response['bands'][band]['predicted'], 3))
        for projection in ('ProjZEA', 'ProjSTG'):
            diagnostic = data['b_diagnostic'][footprint + '_' + projection]
            for key, label in (('mismatch', 'Mismatch'), ('differential', 'Differential'),
                               ('cross_mismatch', 'CrossMismatch'),
                               ('mismatch_power_change', 'MismatchChange')):
                M.add(prefix + projection[4:] + label, sci(diagnostic[key], 3))
    gaps = [abs(r['bands'][b]['measured'] - r['bands'][b]['predicted'])
            for r in data['response'].values() for b in ('1000_2000', '2000_3000')]
    M.add('beamBandAgreement', f(100 * max(gaps), 3))
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "beam_summary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.beam_summary / 'beam_summary.json').read_text())
    macros(data).write(args.out, "beam_numbers")
