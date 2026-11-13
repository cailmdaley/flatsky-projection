"""Masked and noisy SFL template macros: isolated autos and split-cross scatter.

Formats the `masked_template_summary` output: the noisy masked-E template's
isolated residual over the raw BB per band, the oracle and the local Wiener
prediction at l ~ 1000, the noise-free residual, the estimator's cost over
the oracle, and the split-cross standard-deviation gains with their paired
bootstrap intervals.

Run:  bash env.sh python3 analysis/paper_numbers/masked_template_numbers.py
          --masked-template-summary <masked_template_summary dir> --out <dir>
Out:  <out>/masked_template_numbers.tex and <out>/masked_template_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros  # noqa: E402


def macros(d):
    low=d['bands']['900-1100'];m=low['masked'];o=low['oracle']
    high=d['bands']['2000-3000']['masked']
    interval=m['paired_seed_bootstrap_gain_95pct']
    high_interval=high['paired_seed_bootstrap_gain_95pct']
    wide=low['wide']
    M = Macros()
    for name, value in {
        'maskedTemplateResidualLowBand':f"{100*d['bands']['1000-2000']['masked']['isolated_noisy_residual_over_raw']:.0f}\\%",
        'maskedTemplateResidualMidBand':f"{100*high['isolated_noisy_residual_over_raw']:.0f}\\%",
        'maskedTemplateResidualTopBand':f"{100*d['bands']['3000-4000']['masked']['isolated_noisy_residual_over_raw']:.0f}\\%",
        'maskedTemplateWienerBias':f"{100*high['cross_mean_over_raw_signal']:.1f}\\%",
        'maskedTemplateWideResidual':f"{100*wide['isolated_noisy_residual_over_raw']:.1f}\\%",
        'maskedTemplateSeeds':str(d['nseed']),
        'maskedTemplateResidualPercent':f"{100*m['isolated_noisy_residual_over_raw']:.3f}\\%",
        'maskedTemplateResidualSEPercent':f"{100*m['isolated_noisy_residual_SE_over_raw']:.3f}\\%",
        'maskedTemplateOracleResidualPercent':f"{100*o['isolated_noisy_residual_over_raw']:.3f}\\%",
        'maskedTemplateLocalPredictionPercent':f"{100*low['local_Wiener_noise_prediction_over_raw']:.3f}\\%", 
        'maskedTemplateNoiseless':f"\\num{{{m['noiseless_residual_over_raw']:.2e}}}",
        'maskedTemplateEstimatorCost':f"{low['estimator_cost_isolated_masked_over_oracle']:.4f}",
        'maskedTemplateStdGain':f"{m['cross_standard_deviation_gain_vs_fixed_power_subtraction']:.1f}",
        'maskedTemplateStdGainRounded':f"{m['cross_standard_deviation_gain_vs_fixed_power_subtraction']:.0f}",
        'maskedTemplateStdGainInterval':f"$[{interval[0]:.0f}, {interval[1]:.0f}]$", 
        'maskedTemplateRMSGain':f"{m['sample_RMS_gain_without_residual_debiasing']:.1f}",
        'maskedTemplateConditionedGain':f"{m['cross_standard_deviation_gain_vs_EE_amplitude_conditioned']:.1f}",
        'maskedTemplateHighStdGain':f"{high['cross_standard_deviation_gain_vs_fixed_power_subtraction']:.2f}",
        'maskedTemplateHighStdGainInterval':f"$[{high_interval[0]:.2f}, {high_interval[1]:.2f}]$",
        'maskedTemplateHighRMSGain':f"{high['sample_RMS_gain_without_residual_debiasing']:.3f}",
    }.items():
        M.add(name, value)
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "masked_template_summary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = json.loads((args.masked_template_summary / "masked_template_summary.json").read_text())
    macros(d).write(args.out, "masked_template_numbers")


if __name__ == "__main__":
    main()
