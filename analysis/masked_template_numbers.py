"""Macro dictionary for the sole writer analysis/paper_numbers.py.

Generate analysis/masked_template_results/summary.json with
analysis/masked_template_report.py before calling numbers().
This module does not write paper/numbers.tex or edit the paper.
"""
from pathlib import Path
import json


SUMMARY=Path(__file__).resolve().parent/'masked_template_results'/'summary.json'


def numbers(path=SUMMARY):
    d=json.loads(Path(path).read_text())
    assert d['nseed']==20
    low=d['bands']['900-1100'];m=low['masked'];o=low['oracle']
    high=d['bands']['2000-3000']['masked']
    interval=m['paired_seed_bootstrap_gain_95pct']
    high_interval=high['paired_seed_bootstrap_gain_95pct']
    wide=low['wide']
    return {
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
    }


if __name__=='__main__':
    print(json.dumps(numbers(),indent=2))
