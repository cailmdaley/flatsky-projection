"""Paired OSG lognormal controls, including variance-normalized summaries.

All reported errors pair the same skies and account for shared denominators.
A third-moment change alone is not evidence of a change beyond its variance.

Run:  bash env.sh python3 analysis/beyond_two_point_report.py
          --beyond-two-point results/fiducial/beyond_two_point
          --beyond-two-point-resolution results/fiducial/beyond_two_point_resolution
          --beyond-two-point-calibrated results/fiducial/beyond_two_point_calibrated
          --beyond-two-point-calibrated-resolution results/fiducial/beyond_two_point_calibrated_resolution
          --out <dir>
Out:  <out>/beyond_two_point_summary.json, and <out>/report.md with its
      figure <out>/comparison.png, which the paper does not include.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from paper_style import PROJ_COLORS, PROJ_MARKERS
import spectra as S

KEYS = ('smooth_variance', 'smooth_third', 'reduced_skewness',
        'wavelet_l1', 'peaks_gt_2sigma')


def load(path):
    return json.loads(path.read_text())


def ensemble(samples):
    """Original six moments of the ensemble and S3=M3/M2^2, not mean(S3_i)."""
    mean = np.mean(samples, axis=-2)
    return np.concatenate((mean, (mean[..., 3]/mean[..., 2]**2)[..., None]), axis=-1)


def bootstrap_draws(n):
    """20000 whole-sky resamples of n paired skies, from a fixed seed."""
    return np.random.default_rng(4).integers(0, n, (20000, n))


def estimate(t, m):
    tv, mv = ensemble(t), ensemble(m)
    bias = mv/tv-1
    influence = (m-t-bias[:6]*t)/tv[:6]
    # Log derivative of (M3_m/M3_t)*(M2_t/M2_m)^2.
    s3_influence = (1+bias[-1])*(
        (m[:, 3]-mv[3])/mv[3]-(t[:, 3]-tv[3])/tv[3]
        +2*(t[:, 2]-tv[2])/tv[2]-2*(m[:, 2]-mv[2])/mv[2])
    influence = np.column_stack((influence, s3_influence))
    se = influence.std(0, ddof=1)/len(t)**.5
    # Use delete-one-whole-sky jackknife for the additional nonlinear ratio.
    leave_one = []
    for i in range(len(t)):
        keep = np.arange(len(t)) != i
        leave_one.append((ensemble(m[keep])/ensemble(t[keep])-1)[-1])
    se[-1] = np.std(leave_one)*np.sqrt(len(t)-1)
    boot = bootstrap_draws(len(t))
    bootstrap = ensemble(m[boot])/ensemble(t[boot])-1
    return bias, se, bootstrap


def summarize_arm(directory, controls, calibrated):
    data = [load(p) for p in sorted(directory.glob('seed*.json'))]
    assert [d['seed'] for d in data] == list(range(len(data)))
    assert all(d['mean_scale_calibration'] == calibrated for d in data)
    names = data[0]['statistics']+['reduced_skewness']
    records = []
    for j, row in enumerate(data[0]['rows']):
        np.testing.assert_allclose(row['plane_sigma_deg'],
            .5*np.exp(row['mean_log_mu']/2) if calibrated else .5, rtol=1e-12)
        t = np.array([d['rows'][j]['reference'] for d in data])
        m = np.array([d['rows'][j]['measured'] for d in data])
        assert np.array_equal(t[:, :2], m[:, :2])  # Construction identity only.
        bias, se, bootstrap = estimate(t, m)
        interval = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        record = dict(chart=row['chart'], diameter=row['diameter'],
            g2=row['g2'], mean_log_mu=row['mean_log_mu'],
            plane_sigma_deg=row['plane_sigma_deg'], statistics={})
        for k, name in enumerate(names):
            sampling = {}
            for resolution, c in controls.items():
                ct = np.array([d['rows'][j]['reference'] for d in c])
                cm = np.array([d['rows'][j]['measured'] for d in c])
                cbias = ensemble(cm)/ensemble(ct)-1
                bbias = ensemble(m[:len(c)])/ensemble(t[:len(c)])-1
                sampling[resolution] = dict(bias=float(cbias[k]),
                                           paired_drift=float(cbias[k]-bbias[k]))
            stat = dict(bias=float(bias[k]), se=float(se[k]),
                bootstrap_se=float(bootstrap[:, k].std(ddof=1)),
                bootstrap_95_interval=interval[:, k].tolist(),
                reference_mean=float(ensemble(t)[k]), controls=sampling)
            if k < 6:
                stat['single_patch_signal_std'] = float(t[:, k].std(ddof=1))
                stat['bias_over_single_patch_signal_std'] = float(
                    (m-t)[:, k].mean()/t[:, k].std(ddof=1))
            if name == 'peaks_gt_2sigma':
                stat['reference_total'] = int(t[:, k].sum())
                stat['measured_total'] = int(m[:, k].sum())
                stat['net_count_delta'] = int((m-t)[:, k].sum())
            record['statistics'][name] = stat
        records.append(record)
    return records


def control_skies(paths):
    """The control skies at `paths`, which must be seeds 0..k-1: they repeat
    the first k skies of the arm they control."""
    data = [load(p) for p in sorted(paths)]
    assert [d['seed'] for d in data] == list(range(len(data)))
    return data


def summarize(base, control, calibrated_dir, calibrated_control):
    controls = {r: control_skies(control.glob(f'seed*_{r}.json'))
                for r in ('sphere', 'plane', 'both')}
    calibrated_controls = {'both': control_skies(calibrated_control.glob('seed*.json'))}
    records = summarize_arm(base, controls, False)
    calibrated = summarize_arm(calibrated_dir, calibrated_controls, True)
    raw_spreads = []
    for diameter in sorted({r['diameter'] for r in records}):
        rows = [r for r in records if r['diameter'] == diameter]
        entry = dict(diameter=diameter)
        for key in ('raw_variance', 'raw_third'):
            means = np.array([r['statistics'][key]['reference_mean'] for r in rows])
            entry[key] = float(np.ptp(means)/means[2])  # ZEA normalization.
        raw_spreads.append(entry)
    baseline_data = [load(p) for p in sorted(base.glob('seed*.json'))]
    calibrated_data = [load(p) for p in sorted(calibrated_dir.glob('seed*.json'))]
    decomposition = []
    for j, row in enumerate(records):
        truth = np.array([d['rows'][j]['reference'][4] for d in baseline_data])
        fixed = np.array([d['rows'][j]['measured'][4] for d in baseline_data])
        matched = np.array([d['rows'][j]['measured'][4] for d in calibrated_data])
        difference = fixed-matched
        term = difference.mean()/truth.mean()
        error = (difference-term*truth).std(ddof=1)/len(truth)**.5/truth.mean()
        decomposition.append(dict(chart=row['chart'], diameter=row['diameter'],
                                  mean_scale_term=float(term), se=float(error)))
    return dict(nseed=len(baseline_data), ncontrol=len(controls['both']),
                model='signal-only scalar lognormal',
                records=records, calibrated_records=calibrated,
                raw_cross_chart_spreads=raw_spreads,
                wavelet_calibration_decomposition=decomposition)


def figure(summary, out):
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'text.usetex': False})
    colors = dict(TAN='#bd544a', STG=PROJ_COLORS['ProjSTG'], ZEA=PROJ_COLORS['ProjZEA'])
    markers = dict(TAN='o', STG=PROJ_MARKERS['ProjSTG'], ZEA=PROJ_MARKERS['ProjZEA'])
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, key, label in zip(axes.flat,
            ('smooth_third', 'reduced_skewness', 'wavelet_l1', 'peaks_gt_2sigma'),
            ('Third-moment difference (%)', 'Reduced-skewness difference (%)',
             'Wavelet L1 difference (%)', 'Peak-count difference (%)')):
        for chart, color in colors.items():
            rows = [r for r in summary['records'] if r['chart'] == chart]
            ax.errorbar([r['diameter'] for r in rows],
                        [100*r['statistics'][key]['bias'] for r in rows],
                        yerr=[100*r['statistics'][key]['se'] for r in rows],
                        color=color, marker=markers[chart], ms=4, lw=1, capsize=2, label=chart)
        ax.axhline(0, color='.7', lw=.7)
        ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([14.6, 30, 60], ['14.6', '30', '60'])
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('Sky-disk diameter (degrees)')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)


def table(lines, records):
    lines += ['| Chart | Sky-disk diameter | Variance | Third moment | Reduced skewness | Wavelet L1 | Peaks > 2σ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        cells = [f'{100*row["statistics"][k]["bias"]:.2f} ± {100*row["statistics"][k]["se"]:.2f}' for k in KEYS]
        lines.append(f'| {row["chart"]} | {row["diameter"]:g}° | '+' | '.join(cells)+' |')


def report(summary, out):
    lines = ['# Scalar lognormal projection control', '',
        'Fixed central filter scales change the variance and third moment, but the third-moment response largely follows the variance.',
        'At fixed central scale, reduced skewness S3 = M3/M2² shows no significant change in this toy field.',
        'Most of the large raw-moment and peak shifts disappear after matching the mean angular filter scale.',
        'These are signal-only filtering controls, not irreducible non-Gaussian biases or weak-lensing survey forecasts.', '',
        '## Fixed central filter scale', '',
        'Values are percent differences with paired Monte Carlo standard errors from 24 skies.',
        'The reference filters on the sphere before projection; the measurement filters after projection with the same central angular width.',
        'Reduced skewness is formed from ensemble-mean second and third moments, not by averaging the skewness of individual patches.', '',
        '![Paired fractional differences for third moments, reduced skewness, wavelet L1 and peaks.](comparison.png)', '',
        'Bars are paired Monte Carlo errors and exclude the sampling drifts below.', '']
    table(lines, summary['records'])
    lines += ['', 'Errors use paired influence functions for the original summaries and a delete-one-whole-sky jackknife for reduced skewness; 20,000 whole-sky bootstrap resamples provide a cross-check.',
        'The large third-moment shift must be quoted together with reduced skewness: it is not evidence for an independent non-Gaussian response beyond variance.',
        'The reduced-skewness null is specific to this lognormal field and these filters; it is not a theorem that projection affects only variance.', '',
        '## Mean angular-scale calibration', '',
        'For each disk and chart, multiply both planar Gaussian widths by exp(⟨ln μ⟩/2), with a solid-angle average over that disk.',
        'This matches the mean log angular smoothing scale; local magnification variation and shear remain.',
        'It is a global scale calibration, not a map-level removal of projection.', '',
        'The same 24 skies give the following residual differences on 60° disks.', '']
    table(lines, [r for r in summary['calibrated_records'] if r['diameter'] == 60.])
    wavelet = next(r for r in summary['calibrated_records']
                   if r['chart'] == 'TAN' and r['diameter'] == 60.)['statistics']['wavelet_l1']
    term = next(r for r in summary['wavelet_calibration_decomposition']
                if r['chart'] == 'TAN' and r['diameter'] == 60.)
    lines += ['', 'Wavelet L1 is the exception: mean-scale calibration does not remove most of its shift.',
        f'TAN 60° retains a {100*wavelet["bias"]:.2f} ± {100*wavelet["se"]:.2f}% wavelet response with its sign reversed.',
        f'The fixed-scale value partly cancels a paired mean-scale term of +{100*term["mean_scale_term"]:.2f} ± {100*term["se"]:.2f}% against that residual.',
        'These paired errors use the same reference and skies, not independent subtraction errors.',
        'Calibrated reduced skewness is a separate, marginal measurement; the fixed-scale null must not be used to assert that every calibrated statistic is zero.',
        'ZEA has μ=1, so its scale factor is exactly one and its calibrated row repeats the same measurement rather than adding evidence.']
    lines += ['', 'The unit-central-scale percentages are therefore not invariant properties of a chart or statistic.',
        'Calibrated residuals describe this chosen mean-scale matching rule; no universal residual law is established.', '',
        '## Model and estimator', '',
        'The field is exp(G − 0.7²/2) − 1, with an isotropic Gaussian log-field of rms 0.7.',
        'Its spectrum is proportional to (ℓ + 20)^−1.5 exp[−ℓ(ℓ+1)(0.25°)²], with monopole and dipole zero.',
        'The spherical Gaussian widths are sigma 0.5° and 1°; their difference defines the wavelet.',
        'Wavelet L1 is the mean absolute amplitude, not a Morlet scattering transform.',
        'The 14.6°, 30° and 60° footprints are spherical disks, not DES square patches.',
        'Area moments use solid-angle weights and subtract each patch mean.',
        'Peaks exceed twice the full-sphere reference smoothed rms and are maxima within a fixed ±0.3° chart square.',
        'That neighbourhood suppresses interpolation-scale duplicates and is part of this estimator.',
        'The peak threshold is held fixed between the spherical and planar arms, including after scale calibration.',
        'Six reference Gaussian-sigma overscan excludes map-edge effects for both filter normalizations.',
        'The paired raw-moment equality is a construction identity, since both arms use the same raw array; it is not an empirical test of invariance.', '',
        'Independently sampled raw scalar moments agree across charts on the matched disk to the following fractional spreads.', '',
        '| Diameter | Raw variance spread (%) | Raw third-moment spread (%) |', '|---|---:|---:|']
    for row in summary['raw_cross_chart_spreads']:
        lines.append(f'| {row["diameter"]:g}° | {100*row["raw_variance"]:.3f} | {100*row["raw_third"]:.3f} |')
    lines += ['', 'The 14.6° peak percentages involve only a few count changes across all 24 skies.', '',
              '| Chart | Reference count | Planar count | Net change |', '|---|---:|---:|---:|']
    for row in summary['records'][:3]:
        s = row['statistics']['peaks_gt_2sigma']
        lines.append(f'| {row["chart"]} | {s["reference_total"]} | {s["measured_total"]} | {s["net_count_delta"]:+d} |')
    lines += ['', '## Sampling controls', '',
        'Four identical-alm seeds repeat at doubled HEALPix and/or chart sampling.',
        'The baseline is Nside 512, 0.1° pixels and fixed ℓmax=1024; controls use Nside 1024 and/or 0.05° pixels.',
        'All listed drifts are maxima of absolute paired four-seed differences over the nine chart/footprint configurations, in percentage points.',
        'They are numerical regression diagnostics, not independent physical bounds.', '',
        '| Statistic | Sphere only | Plane only | Both | Calibrated, both |', '|---|---:|---:|---:|---:|']
    for k in KEYS:
        values = [max(abs(r['statistics'][k]['controls'][c]['paired_drift']) for r in summary['records'])*100
                  for c in ('sphere', 'plane', 'both')]
        values += [max(abs(r['statistics'][k]['controls']['both']['paired_drift']) for r in summary['calibrated_records'])*100]
        lines.append('| '+k+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    control = wavelet['controls']['both']
    coarse = control['bias']-control['paired_drift']
    lines += ['',
        f'The calibrated TAN 60° wavelet shift is stable under doubled sampling on the four control skies: {100*coarse:.3f}% → {100*control["bias"]:.3f}%.',
        'This is evidence that the residual is not just the observed interpolation drift; it is still not a continuum value or survey bound.',
        'The fixed-scale TAN wavelet shift has a much larger sampling drift and should not be quoted as a precision result.',
        'Small ZEA shifts are not continuum-accuracy measurements or upper limits.',
        'The small-disk peak changes remain unresolved at the sampling-control level.', '',
        '## Interpretation and inference', '',
        'At fixed central chart scale there is no universal shear-only law: STG has g=0 yet its smoothed moments change.',
        'Locally, magnification changes a chart kernel’s sky width by δ ln sigma = −½ ln μ.',
        'For the monotonic second and third moments here, a smoothing-scale derivative times ⟨ln μ⟩ explains the leading response.',
        'The second-order-in-radius magnification response versus fourth-order g² comparison assumes unit chart scale at the tangent point; a global scale matching removes most of it.',
        'It must not be presented as an irreducible distortion that survives scale calibration.',
        'The independent Opus scale scan found wavelet L1 near a stationary point of its response in this toy spectrum; its small fixed-scale shift partly cancels the surviving residual and is not evidence of general wavelet robustness.',
        'No quantitative universal wavelet response law is established.',
        'For homogeneous isotropic fields and local isotropic filters, pure fixed-determinant shear can have a leading quadratic response with a field- and statistic-dependent coefficient.',
        'Continuous unfiltered scalar values and extrema are invariant under coordinate relabelling on a matched footprint with the correct measure.', '',
        'No largest survey-safe patch size is established.',
        'Shape noise, masks, source distributions, cosmological derivatives, multiscale covariance and spin-two mass mapping are absent.',
        'The stored signal-only single-patch scatter is not a realistic survey error bar.',
        'A dedicated weak-lensing analysis is needed for survey-error tolerances, and this disk/lognormal/Gaussian example does not reproduce DES scattering patches.', '',
        'Jeffrey et al. (arXiv:2403.02314, §6.3; bibliography key jeffrey25) apply the same flattening to data and simulations and distinguish correct forward modelling from potentially suboptimal compression.',
        'A shared deterministic transformation removes this particular data/model mismatch only when masks, noise, filtering and spin transport are included consistently at every parameter value.',
        'These toys do not demonstrate calibrated cosmological posteriors or recover information discarded by compression.', '',
        '## Suggested manuscript scope', '',
        'The useful result is a filtering-scale calibration example with a reduced-skewness null, not a discovery of an extra higher-order bias.',
        'If retained, put it after the scalar-versus-spin-two footprint discussion and quote both raw and normalized moments.',
        'The conclusion should reserve survey-error bounds for a dedicated study.', '',
        r'On scalar lognormal skies, applying a fixed central-width Gaussian filter after projection rather than on the sphere changes the third moment by \ngTANThirdLarge\ on a $60^\circ$ gnomonic disk, but the reduced skewness $S_3=M_3/M_2^2$ is consistent with zero shift (\ngTANReducedSkewnessLarge\ with jackknife error \ngTANReducedSkewnessLargeError).',
        r'Matching the mean angular filter scale leaves a third-moment shift of \ngCalTANThirdLarge\ (error \ngCalTANThirdLargeError), whereas the mean absolute amplitude of a $0.5^\circ$ minus $1^\circ$ Gaussian wavelet retains a shift of \ngCalTANWaveletLarge\ (error \ngCalTANWaveletLargeError).',
        r'The large raw third-moment shift alone does not establish an independent higher-order bias, and mean-scale calibration does not remove every statistic’s response.',
        r'Projection therefore belongs in the simulations and filtering convention of a higher-order analysis \citep{jeffrey25}; realistic safe-patch bounds require a dedicated weak-lensing study.', '',
        'The macros are the beyond_two_point_numbers output, paper/beyond_two_point_numbers.tex.', '',
        '## Reproduction', '',
        '`lc run --executor htcondor --profile htcondor9 beyond_two_point beyond_two_point_resolution beyond_two_point_calibrated beyond_two_point_calibrated_resolution -u fiducial`',
        '`lc run beyond_two_point_numbers -u fiducial`', '',
        'Raw measurements and manifests live under results/fiducial/beyond_two_point*; the beyond_two_point_summary output holds all cells, bootstrap errors, counts and controls.', '']
    (out/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(p, 'beyond_two_point', 'beyond_two_point_resolution',
                 'beyond_two_point_calibrated', 'beyond_two_point_calibrated_resolution')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(a.beyond_two_point, a.beyond_two_point_resolution,
                        a.beyond_two_point_calibrated, a.beyond_two_point_calibrated_resolution)
    (a.out/'beyond_two_point_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    figure(summary, a.out)
    report(summary, a.out)
    print(a.out/'report.md')
