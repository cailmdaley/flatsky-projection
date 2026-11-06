# Scalar lognormal projection control

A fixed planar smoothing kernel changes non-Gaussian statistics even on a conformal chart.
These measurements are signal-only controls, not weak-lensing survey forecasts.

## Paired differences

Values are percent differences of ensemble means, with paired Monte Carlo standard errors from 24 skies.
The reference applies the Gaussian filter on the sphere before projection; the measurement filters after projection.

| Chart | Sky-disk diameter | Smoothed variance | Third central moment | Wavelet L1 | Peaks > 2σ |
|---|---:|---:|---:|---:|---:|
| TAN | 14.6° | 0.58 ± 0.03 | 1.10 ± 0.13 | 0.09 ± 0.02 | 3.26 ± 1.81 |
| STG | 14.6° | 0.19 ± 0.02 | 0.35 ± 0.06 | 0.03 ± 0.01 | -1.08 ± 1.06 |
| ZEA | 14.6° | -0.01 ± 0.01 | -0.02 ± 0.04 | -0.00 ± 0.00 | 0.00 ± 1.57 |
| TAN | 30° | 2.35 ± 0.08 | 4.77 ± 0.35 | 0.17 ± 0.04 | 4.79 ± 1.33 |
| STG | 30° | 0.77 ± 0.03 | 1.56 ± 0.12 | 0.07 ± 0.01 | 1.26 ± 0.93 |
| ZEA | 30° | -0.00 ± 0.01 | -0.01 ± 0.02 | 0.01 ± 0.01 | -0.50 ± 0.50 |
| TAN | 60° | 9.03 ± 0.14 | 19.56 ± 0.78 | 0.30 ± 0.07 | 22.50 ± 1.13 |
| STG | 60° | 2.92 ± 0.04 | 6.06 ± 0.22 | 0.21 ± 0.02 | 6.40 ± 0.51 |
| ZEA | 60° | -0.03 ± 0.01 | -0.09 ± 0.04 | 0.01 ± 0.01 | -0.18 ± 0.26 |

## Model and estimator

The scalar field is exp(G − 0.7²/2) − 1, where G is an isotropic Gaussian sphere with rms 0.7.
Its input spectrum is proportional to (ℓ + 20)^−1.5 exp[−ℓ(ℓ+1)(0.25°)²], with monopole and dipole zero.
The nonlinear field is harmonically filtered with Gaussian sigma 0.5° and 1°; their difference defines the wavelet.
The wavelet statistic is its mean absolute amplitude, not a Morlet scattering transform.
Patch diameters describe spherical disks, not the 14.6° square patches of DES.
All area moments use solid-angle weights and subtract the patch mean.
Peaks exceed twice the full-sphere smoothed rms and are maxima within a fixed ±0.3° chart square.
The neighbourhood suppresses interpolation-scale duplicate maxima; it is part of this estimator, not a universal peak definition.
Five Gaussian-sigma overscan excludes artificial map-edge effects.
Both arms use identical chart pixel locations and solid-angle weights.
Unfiltered paired moments agree exactly because a scalar is unchanged by a coordinate relabelling.

## Sampling controls

Four identical input-alm seeds repeat with doubled HEALPix sampling, doubled chart sampling, and both.
The baseline is Nside 512, 0.1° chart pixels and fixed ℓmax=1024; controls use Nside 1024 and/or 0.05° pixels.
The table lists the largest absolute paired four-seed drift in percentage points across all nine configurations.
The statistical error bars do not include these numerical shifts.
The small wavelet and ZEA shifts are not continuum-accuracy measurements or upper bounds.
Peak shifts on the smallest disks remain unresolved at the sampling-control level; the large TAN effect is substantially larger than that drift.

| Statistic | Sphere only | Plane only | Both |
|---|---:|---:|---:|
| smooth_variance | 0.050 | 0.017 | 0.052 |
| smooth_third | 0.152 | 0.124 | 0.153 |
| wavelet_l1 | 0.089 | 0.005 | 0.088 |
| peaks_gt_2sigma | 2.778 | 1.408 | 2.778 |

## Interpretation

There is no universal ⟨g²⟩ law for these summaries: STG has g=0 everywhere, yet changes the smoothed moments.
A fixed chart kernel has a position-dependent angular size on the sphere.
Locally, isotropic magnification μ changes its sky smoothing scale by δ ln σ = −½ ln μ.
The leading response of a scale-dependent statistic is therefore proportional to its smoothing-scale derivative and ⟨ln μ⟩, not just shear.
For these azimuthal charts the magnification begins at second order in radius; g² begins at fourth order.
At fixed determinant, isotropic ensemble statistics can instead have a leading quadratic shear response, but its coefficient depends on the field, filter and statistic.
Raw scalar one-point statistics and continuous unfiltered extrema are invariant on a matched footprint with the correct measure.
A nonzero chart dependence enters through filtering, sampling, area weighting, or a spin-field reconstruction, not through a scalar value changing under projection.

## Inference and scope

No largest survey-safe patch size is established here.
A signal-only single-patch scatter is available in summary.json, but it is not a realistic error bar for DES, Euclid or LSST and must not be described as one.
Shape noise, masks, source distributions, cosmological derivatives, multiscale covariance and spin-2 mass mapping are absent.
A dedicated weak-lensing analysis is needed to set a tolerance in survey-error units.
The 14.6° disk, correlation spectrum and filter here do not reproduce Gatti et al.’s 14.6° square Morlet-scattering analysis.

Jeffrey et al. (2024, arXiv:2403.02314, §6.3) explicitly apply the same flattening to data and simulations and distinguish unbiased inference from potentially suboptimal compression.
That shared deterministic transformation is a valid forward model when implemented consistently at every parameter value, including masks, noise, filtering and spin transport.
It removes this particular model/data mismatch; these toy measurements do not demonstrate calibrated cosmological posteriors or recover information discarded by compression.

## Reproduction

`lc run --executor htcondor --profile htcondor9 beyond_two_point beyond_two_point_resolution -u highell`
`bash env.sh python3 analysis/check_beyond_two_point.py`
`bash env.sh python3 analysis/beyond_two_point_report.py`

Raw paired measurements and their lightcone manifests live under results/highell/beyond_two_point and beyond_two_point_resolution.
summary.json contains every table cell, paired standard error, signal-only scatter and sampling-control comparison.
