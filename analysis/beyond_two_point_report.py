"""Summarize the OSG paired scalar-lognormal projection controls.

Run: bash env.sh python3 analysis/beyond_two_point_report.py
Writes analysis/beyond_two_point_results/{summary.json,comparison.png,report.md}.
SEs use paired ratio-of-ensemble-means influence functions, NOT independent
errors for the two arms. A single-patch scatter is explicitly signal-only.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/highell/beyond_two_point'
CONTROL = ROOT/'results/highell/beyond_two_point_resolution'
OUT = ROOT/'analysis/beyond_two_point_results'


def load(path):
    return json.loads(path.read_text())


def summarize():
    data = [load(p) for p in sorted(BASE.glob('seed*.json'))]
    assert len(data) == 24
    assert [d['seed'] for d in data] == list(range(24))
    names = data[0]['statistics']
    records = []
    for j, row in enumerate(data[0]['rows']):
        t = np.array([d['rows'][j]['reference'] for d in data])
        m = np.array([d['rows'][j]['measured'] for d in data])
        assert np.array_equal(t[:, :2], m[:, :2])
        difference = m-t
        bias = difference.mean(0)/t.mean(0)
        se = (difference-bias*t).std(0, ddof=1)/len(data)**.5/t.mean(0)
        record = dict(chart=row['chart'], diameter=row['diameter'],
                      g2=row['g2'], mean_log_mu=row['mean_log_mu'], statistics={})
        for k, name in enumerate(names):
            controls = {}
            for resolution in ('sphere', 'plane', 'both'):
                c = [load(CONTROL/f'seed{s:03d}_{resolution}.json')['rows'][j]
                     for s in range(4)]
                cm = np.array([r['measured'] for r in c])
                ct = np.array([r['reference'] for r in c])
                cbias = (cm-ct).mean(0)/ct.mean(0)
                # Paired four-seed drift: do not confuse cosmic scatter
                # of the subset with sampling-resolution dependence.
                bbias = difference[:4].mean(0)/t[:4].mean(0)
                controls[resolution] = dict(bias=float(cbias[k]),
                                           paired_drift=float(cbias[k]-bbias[k]))
            record['statistics'][name] = dict(bias=float(bias[k]), se=float(se[k]),
                reference_mean=float(t[:, k].mean()),
                single_patch_signal_std=float(t[:, k].std(ddof=1)),
                bias_over_single_patch_signal_std=float(difference[:, k].mean()/t[:, k].std(ddof=1)),
                controls=controls)
        records.append(record)
    return dict(nseed=len(data), ncontrol=4, model='signal-only scalar lognormal',
                records=records)


def figure(summary):
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'text.usetex': False})
    colors = dict(TAN='#bd544a', STG='#687ba5', ZEA='#7b9b65')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, key, label in zip(axes.flat,
            ('smooth_variance', 'smooth_third', 'wavelet_l1', 'peaks_gt_2sigma'),
            ('Variance difference (%)', 'Third-moment difference (%)',
             'Wavelet L1 difference (%)', 'Peak-count difference (%)')):
        for chart, color in colors.items():
            rows = [r for r in summary['records'] if r['chart'] == chart]
            ax.errorbar([r['diameter'] for r in rows],
                        [100*r['statistics'][key]['bias'] for r in rows],
                        yerr=[100*r['statistics'][key]['se'] for r in rows],
                        color=color, marker='o', ms=4, lw=1, capsize=2, label=chart)
        ax.axhline(0, color='.7', lw=.7)
        ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([14.6, 30, 60], ['14.6', '30', '60'])
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('Sky-disk diameter (degrees)')
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png', dpi=180)
    plt.close(fig)


def report(summary):
    lines = ['# Scalar lognormal projection control', '',
        'A fixed planar smoothing kernel changes non-Gaussian statistics even on a conformal chart.',
        'These measurements are signal-only controls, not weak-lensing survey forecasts.', '',
        '## Paired differences', '',
        'Values are percent differences of ensemble means, with paired Monte Carlo standard errors from 24 skies.',
        'The reference applies the Gaussian filter on the sphere before projection; the measurement filters after projection.', '',
        '| Chart | Sky-disk diameter | Smoothed variance | Third central moment | Wavelet L1 | Peaks > 2σ |',
        '|---|---:|---:|---:|---:|---:|']
    keys = ('smooth_variance', 'smooth_third', 'wavelet_l1', 'peaks_gt_2sigma')
    for row in summary['records']:
        cells = [f'{100*row["statistics"][k]["bias"]:.2f} ± {100*row["statistics"][k]["se"]:.2f}' for k in keys]
        lines.append(f'| {row["chart"]} | {row["diameter"]:g}° | '+ ' | '.join(cells)+' |')
    lines += ['', '## Model and estimator', '',
        'The scalar field is exp(G − 0.7²/2) − 1, where G is an isotropic Gaussian sphere with rms 0.7.',
        'Its input spectrum is proportional to (ℓ + 20)^−1.5 exp[−ℓ(ℓ+1)(0.25°)²], with monopole and dipole zero.',
        'The nonlinear field is harmonically filtered with Gaussian sigma 0.5° and 1°; their difference defines the wavelet.',
        'The wavelet statistic is its mean absolute amplitude, not a Morlet scattering transform.',
        'Patch diameters describe spherical disks, not the 14.6° square patches of DES.',
        'All area moments use solid-angle weights and subtract the patch mean.',
        'Peaks exceed twice the full-sphere smoothed rms and are maxima within a fixed ±0.3° chart square.',
        'The neighbourhood suppresses interpolation-scale duplicate maxima; it is part of this estimator, not a universal peak definition.',
        'Five Gaussian-sigma overscan excludes artificial map-edge effects.',
        'Both arms use identical chart pixel locations and solid-angle weights.',
        'Unfiltered paired moments agree exactly because a scalar is unchanged by a coordinate relabelling.', '',
        '## Sampling controls', '',
        'Four identical input-alm seeds repeat with doubled HEALPix sampling, doubled chart sampling, and both.',
        'The baseline is Nside 512, 0.1° chart pixels and fixed ℓmax=1024; controls use Nside 1024 and/or 0.05° pixels.',
        'The table lists the largest absolute paired four-seed drift in percentage points across all nine configurations.',
        'The statistical error bars do not include these numerical shifts.',
        'The small wavelet and ZEA shifts are not continuum-accuracy measurements or upper bounds.',
        'Peak shifts on the smallest disks remain unresolved at the sampling-control level; the large TAN effect is substantially larger than that drift.', '',
        '| Statistic | Sphere only | Plane only | Both |', '|---|---:|---:|---:|']
    for k in keys:
        vals = [max(abs(r['statistics'][k]['controls'][c]['paired_drift']) for r in summary['records'])*100
                for c in ('sphere', 'plane', 'both')]
        lines.append('| '+k+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines += ['', '## Interpretation', '',
        'There is no universal ⟨g²⟩ law for these summaries: STG has g=0 everywhere, yet changes the smoothed moments.',
        'A fixed chart kernel has a position-dependent angular size on the sphere.',
        'Locally, isotropic magnification μ changes its sky smoothing scale by δ ln σ = −½ ln μ.',
        'The leading response of a scale-dependent statistic is therefore proportional to its smoothing-scale derivative and ⟨ln μ⟩, not just shear.',
        'For these azimuthal charts the magnification begins at second order in radius; g² begins at fourth order.',
        'At fixed determinant, isotropic ensemble statistics can instead have a leading quadratic shear response, but its coefficient depends on the field, filter and statistic.',
        'Raw scalar one-point statistics and continuous unfiltered extrema are invariant on a matched footprint with the correct measure.',
        'A nonzero chart dependence enters through filtering, sampling, area weighting, or a spin-field reconstruction, not through a scalar value changing under projection.', '',
        '## Inference and scope', '',
        'No largest survey-safe patch size is established here.',
        'A signal-only single-patch scatter is available in summary.json, but it is not a realistic error bar for DES, Euclid or LSST and must not be described as one.',
        'Shape noise, masks, source distributions, cosmological derivatives, multiscale covariance and spin-2 mass mapping are absent.',
        'A dedicated weak-lensing analysis is needed to set a tolerance in survey-error units.',
        'The 14.6° disk, correlation spectrum and filter here do not reproduce Gatti et al.’s 14.6° square Morlet-scattering analysis.', '',
        'Jeffrey et al. (2024, arXiv:2403.02314, §6.3) explicitly apply the same flattening to data and simulations and distinguish unbiased inference from potentially suboptimal compression.',
        'That shared deterministic transformation is a valid forward model when implemented consistently at every parameter value, including masks, noise, filtering and spin transport.',
        'It removes this particular model/data mismatch; these toy measurements do not demonstrate calibrated cosmological posteriors or recover information discarded by compression.', '',
        '## Reproduction', '',
        '`lc run --executor htcondor --profile htcondor9 beyond_two_point beyond_two_point_resolution -u highell`',
        '`bash env.sh python3 analysis/check_beyond_two_point.py`',
        '`bash env.sh python3 analysis/beyond_two_point_report.py`', '',
        'Raw paired measurements and their lightcone manifests live under results/highell/beyond_two_point and beyond_two_point_resolution.',
        'summary.json contains every table cell, paired standard error, signal-only scatter and sampling-control comparison.', '']
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    summary = summarize()
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    figure(summary)
    report(summary)
    print(OUT/'report.md')
