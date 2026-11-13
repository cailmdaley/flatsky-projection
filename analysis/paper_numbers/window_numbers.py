"""Window-control and corrected-residual macros, paper/window_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/window_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def window_numbers():
    """Write window-control and corrected-residual macros from checked outputs.

    Run report_pipeline_windows.py and both controlled projector checks first.
    """
    from pipeline_windows import interpolation_power
    directory = Path('/sptlocal/user/cailmdaley/pol_project')
    d = np.load(directory / 'pipeline_windows_production.npz')
    control = json.loads((directory / 'pipeline_projector_control.json').read_text())
    ring = json.loads((directory / 'pipeline_window_control.json').read_text())
    spin = json.loads((directory / 'pipeline_spin_control.json').read_text())
    cutoff = json.loads((directory / 'pipeline_bandlimit_summary.json').read_text())
    M = Macros()
    dec = np.deg2rad(-56.)
    M.add('winRingEast', f(float(interpolation_power(4800., 0., dec)), 4))
    M.add('winRingNorth', f(float(interpolation_power(0., 4800., dec)), 4))
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
    assert all(name.isalpha() for name in M)
    path = ROOT / 'paper' / 'window_numbers.tex'
    path.write_text(M.tex('window_numbers'))
    print(f'Wrote {path}')


if __name__ == "__main__":
    window_numbers()
