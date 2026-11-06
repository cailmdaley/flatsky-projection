# Compact-source matched filters on the 1500d field

Exact spherical compact-source injections confirm percent-level amplitude errors when a circular map-plane filter ignores the local metric.
At the 15 sampled SFL locations, the largest losses are 6.98% for a beam-sized source and 4.85% for an extended cluster proxy with CMB+noise weighting.
ZEA stays below 0.08%; BICEP can either increase or decrease the point-source amplitude because its magnification changes the apparent size.
A template using the known local metric removes the noiseless mismatch to below 1.7e-7.
These are controlled, single-frequency template responses, **not measurements of an SPT catalogue's flux bias or completeness**.

## Definition and prediction

The fitted parameter is central surface brightness, with a spherical radial template normalized to unity at its centre.
Multiplying that parameter by the fixed spherical profile integral gives a nominal flux with the same fractional error.
It is not the integral of brightness over the projected plane, which acquires an additional magnification factor.

For a local sky-to-plane Jacobian J, magnification mu=|det J|, and an isotropic spherical profile tau, the map-plane Fourier profile is mu tau(J^T k).
A circular matched filter with assumed stationary power P(k) therefore has response

    R = integral tau*(k) mu tau(J^T k)/P(k) d²k
        / integral |tau(k)|²/P(k) d²k.

For a Gaussian in white noise, this is

    R = 2 mu / sqrt[(1+s1²)(1+s2²)],

where s1,s2 are the singular values of J.
For mu=1 it reduces to sqrt(1-g²), with g=(s1-s2)/(s1+s2).
`local_source_report.py` separately predicts every source/profile/weight combination using the local affine metric, then compares that prediction against exact inverse-chart spherical samples.
The maximum absolute response differences are 6.6e-8 (Gaussian white), 5.5e-8 (Gaussian CMB+white), 1.2e-7 (cluster white), and 7.4e-8 (cluster CMB+white).
The Gaussian closed-form prediction also agrees to 6.6e-8.

## Injection and recovery

- The charts are the production spt3g SFL, ZEA and BICEP transforms, centred at RA=0, declination=-56 degrees.
- The source positions are the Cartesian product of RA=(-45,-22.5,0,22.5,45) degrees and declination=(-68,-56,-44) degrees.
  The extrema below are over those samples, not over every point of the full footprint, and their mean is not a solid-angle-weighted field average.
- The beam is a 1.2 arcmin FWHM Gaussian.
  The point source is the beam profile.
  The cluster proxy is a beta=1, core=1 arcmin profile, multiplied by an 8 arcmin Gaussian taper and convolved with that beam using a radial Bessel integral.
  This is a specified SZ-like shape, not a pressure-profile or mass calibration.
  The radial beam convolution uses its small-angle form before the resulting radial function is defined on the sphere.
- Each 51.2 arcmin patch has 256² pixels of width 0.2 arcmin.
  Source values are evaluated directly at the spherical coordinates of those pixels.
  Compact sources deliberately bypass intermediate HEALPix pixelization, so the experiment isolates projection/template geometry rather than the pipeline's interpolation response to a barely resolved source.
- White noise is 18 uK arcmin per sky solid angle.
  Its map-pixel variance is mu(18/0.2)² uK² in the local affine approximation.
  This is not an SPT scanning-noise model.
- The circular filter assumes isotropic CMB power plus white noise on its map plane.
  Modes below ell=300 are discarded in the CMB+noise calculation; the patch's fundamental mode is ell=422, so this removes DC only.
  There is no anisotropic TOD transfer function, foreground power, multifrequency combination, or maximization over source/cluster size.
- The geometric output also contains white-only controls and 64 affine, periodic CMB+noise realizations at the southeastern sampled location, using 500 uK sources.
  A separate production output contains 64 independent full-sphere Gaussian CMB realizations drawn from the lensed TT spectrum, shared across projections within each seed, synthesized at Nside 2048 to ell=6000 with the spherical Gaussian beam.
  The CMB is interpolated from HEALPix; each compact source remains an exact analytic spherical sample.
  Those runs inject 100 uK point sources and 300 uK cluster proxies, add local sky-white noise, and recover fixed-location and locally position-maximized amplitudes.

## Responses

| Projection | Point, white | Cluster, white | Point, CMB+white | Cluster, CMB+white |
|---|---:|---:|---:|---:|
| SFL | 0.93965–1.00000 | 0.95518–1.00000 | 0.93019–1.00000 | 0.95148–1.00000 |
| ZEA | 0.99935–1.00000 | 0.99952–1.00000 | 0.99924–1.00000 | 0.99948–1.00000 |
| BICEP | 0.86796–1.17494 | 0.87342–1.19313 | 0.88447–1.12056 | 0.97266–1.00000 |

BICEP's large point-source range includes magnification, rather than being a pure shear measurement.
Using a circular template whose size is scaled by sqrt(mu) reduces its CMB+white point-source range to 0.97593–1.00000 and cluster range to 0.98641–1.00000.
Using the full metric, including ellipticity and orientation, gives unit response to within 1.7e-7 for every noiseless test.
The CMB-weighted cluster response need not follow the Gaussian white-noise formula; the independent affine-profile calculation reproduces it instead.

## Signal-to-noise and astrometry

Amplitude response alone understates detection-significance loss.
Relative to the same profile and filter at the projection centre, the local-affine prediction for SFL signal-to-noise falls to 0.894 for points and 0.817 for the cluster proxy at (45,-68) degrees.
The circular filter's CMB+noise error increases from 21.83 to 22.72 uK for points and from 17.02 to 19.82 uK for the cluster proxy there.
BICEP's predicted signal-to-noise ranges are 0.931–1.000 for points and 0.872–1.000 for the cluster proxy; magnification-induced amplitude enhancements are not enhancements in information.
ZEA stays above 0.99886 for points and 0.99753 for the cluster proxy.
These are forecasts from the transformed local CMB covariance, rather than a direct catalogue measurement.

An elliptical metric template together with the correctly transformed CMB+white covariance restores predicted signal-to-noise to within 0.14% of the central value for all sampled locations and profiles.
That is a separate control from the unit-amplitude result: a correct source shape alone need not give an optimal filter if its noise weighting is wrong.
`check_local_measurements.py --geometry results/highell/local_source_geometry/geometry.json` gates all-profile Fourier responses, the white-Gaussian signal-to-noise formula, and this corrected-filter information invariance.

The noiseless map-coordinate position displacement is below 0.022 arcsec in these pixel-centred tests.
A separate exact-spherical fractional-pixel injection gate recovers offsets (0.07,-0.11) arcmin to 0.001 arcmin in all three charts.
The first 16 noisy spherical-CMB realizations at the sampled SFL corner gave point-source position scatter of (13.6,9.3) arcsec for a 100 uK source and cluster scatter of (6.4,5.5) arcsec for a 300 uK proxy.
The final aggregate retains the 64-seed scatter and whole-seed uncertainty summaries; neither ensemble establishes a subarcsecond bound on mean astrometry.
The noiseless displacement and noise scatter must not be conflated.

For the spherical-CMB runs, subtracting the *same realization's* blank amplitude from its injected fixed-location amplitude reproduces the noiseless response to better than 1e-10.
That is a paired linear-response bookkeeping check, not an independent high-significance measurement of bias.
For example, the first 16 southeastern SFL point-source injections gave a noisy mean of 0.994±0.069 while their paired response was 0.93019; that noisy mean alone did not resolve the predicted 7% loss.
The final 64-seed summary reports the mean and its error separately from this algebraic identity.
Position-maximized noisy amplitudes also have positive maximization bias even on nearly unsheared ZEA, so they do not directly estimate the geometric fixed-location response.

## SPT-like scale and interpretation

A 100 uK, 1.2 arcmin FWHM Gaussian source corresponds to 5.50 mJy at 150 GHz using the thermodynamic dB/dT conversion at T_CMB=2.7255 K.
The chosen 18 uK arcmin noise is an SPT-SZ-like depth, rather than the substantially deeper SPTpol or SPT-3G depths.
The affine CMB+noise filter errors are roughly 22–23 uK on SFL, giving an individual-source signal-to-noise ratio near four to five.
A 7% amplitude error is smaller than one such source's random error, but it does not average away and can matter for a percent-level ensemble flux calibration.
The predicted 11–18% loss of detection significance is a different and potentially larger concern for a threshold-selected sample, although it still does not specify a catalogue-count shift.
That last statement is an interpretation of the measured response scale, not a catalogue-count forecast.
The experiment does not predict a change in the number of SPT detections, because no source counts, selection function, size bank, or survey calibration has been modeled.

Bleem et al. (2024), arXiv:2311.07512, sections 2–3, use 0.25 arcmin SFL maps, 95/150 GHz spatial-spectral filters and beta=1 templates with twelve core sizes from 0.25 to 3 arcmin, incorporating beam/transfer functions and astrophysical/instrumental noise.
Its actual catalogue response cannot be inferred from the present fixed-size, single-frequency filter.
The tested cluster proxy uses one core size from that range, but is not their full analysis.

Ansarinejad et al. (2024), arXiv:2404.02153, sections 2.1 and 3.1, use 0.5 arcmin SFL maps.
They estimate local gradient directions from the central 6 arcmin of 60 arcmin, Wiener-filtered tSZ-free cutouts, then rotate and stack central 10 arcmin, unfiltered CMB cutouts with noise/gradient weights and random-stack subtraction.
I did not substitute a radial matched-filter toy for that dipole estimator.
A faithful follow-up needs the gradient estimation, rotation, weighting and random subtraction together; the optional stacked-lensing test is therefore not part of this bounded measurement.
The existing spherical TT-QE thread addresses a different estimator.

## Numerical gates and reproducibility

`analysis/check_local_measurements.py` checks the Gaussian response, metric-template correction, Fourier normalization against 2,000 fresh pixel-noise draws, shift sign, and exact-spherical subpixel recovery.
At fixed 51.2 arcmin box size, refining pixels from 0.2 to 0.1 arcmin changes amplitudes by at most 3.6e-6.
`analysis/check_local_source_window.py` enlarges the box from 51.2 to 102.4 to 204.8 arcmin at fixed 0.2 arcmin pixels for the three charts and both profiles at (45,-68) degrees.
The largest change from the original box is 1.66e-4 in amplitude; the last doubling changes it by less than 7.5e-6.
These gates constrain the source-response calculation; periodic affine noise errors are not claimed to be exact uncertainties for nonperiodic spherical-CMB cutouts.

The production runs used:

```bash
lc run --executor htcondor --profile htcondor9 local_source_geometry local_source_sims -u highell
bash env.sh python3 analysis/check_local_measurements.py
bash env.sh python3 analysis/check_local_source_window.py
bash env.sh python3 analysis/local_source_report.py
bash env.sh python3 analysis/paper_numbers.py --local-source-only
```

The additive canonical outputs and all 16 unit-directory hashes passed focused lightcone verification.
No claim is made about unrelated outputs in the universe.
The aggregate is `analysis/results_disk_trend/local_sources/local_source_summary.json`; plots beside it show the noiseless response and actual seed-zero cutouts.
The display cutouts use 0.4 arcmin Gaussian smoothing only for display; black contours mark the injected spherical profiles.

## Paper changes supported

Keep the sqrt(1-g²) Gaussian white-noise result and its assumptions.
The other-flat-sky-work paragraph can add that a controlled 1.2 arcmin, single-frequency injection test on 15 field locations gives SFL amplitude losses of `sourceSFLPointLoss` and `sourceSFLClusterLoss` with CMB+white weighting.
The associated local-affine detection-significance losses are `sourceSFLPointSnrLoss` and `sourceSFLClusterSnrLoss`; do not describe amplitude loss as the complete detection effect.
A metric-aware template removes the amplitude mismatch, and a metric-plus-covariance-aware filter restores the forecast signal-to-noise.
Do not identify these with Bleem et al.'s catalogue bias or with the ACT bias quoted from Naess et al.
Do not turn the small noiseless astrometric displacement into an observed catalogue-position bound.
`analysis/paper_numbers.py` generates the new macros in its normal run; `--local-source-only` writes an inspection fragment without requiring unrelated paper products to be regenerated.
