# Projection choice after a conditional response correction

On the 1500d footprint, STG avoids geometrical E-to-B leakage, while ZEA gives slightly smaller TT and EE variance in the conditional inverse tested here.
SFL is a poor choice for this particular binned inverse: its broad response makes the fitted amplitudes much noisier.
These statements concern two-point spectra, not map-level information loss or an optimal likelihood.

## Measured comparison

`analysis/choosing_projection.py` reads the 20 matched, noise-free, unlensed field skies through the window-corrected production loaders.
It builds each chart's full geometric response with `shear_leakage.coupling_matrix`, using the actual field taper at Nside 256.
The fitted parameters multiply the known input TT or EE shape within each sky band between 500 and 4000.
The exterior ensemble spectrum is held at its true input model, while individual skies still contain exterior realization fluctuations.
This does not condition on each realization's exterior modes.
The BB parameters multiply the **input EE** shape, so their true amplitudes are zero.
EE and BB are solved together using both spin blocks.
The fit uses equal weight per fractional output bin, not an estimated covariance or an optimal likelihood.
Its weighting, bandwidth and shape prior are part of the estimator definition.
The simulation-window division also uses the known simulated input.

For sky bands of width 500, averaging the measured variance across the seven fitted bands gives:

| Chart | TT variance / ZEA | EE variance / ZEA | Mean corrected BB / input EE |
|---|---:|---:|---:|
| ZEA | 1 | 1 | (1.28 ± 0.27) × 10⁻⁶ |
| STG | 1.094 ± 0.038 | 1.256 ± 0.089 | (3.743 ± 0.006) × 10⁻⁷ |
| SFL | 121 ± 43 | 142 ± 64 | (−2.24 ± 2.46) × 10⁻⁴ |

Variance-ratio errors delete one matched whole sky at a time.
The BB errors are the standard errors of the seven-band mean across skies.
A negative corrected BB is allowed; the estimator is linear and no positivity prior is applied.
The small, significantly positive STG BB is the numerical pipeline floor, not geometrical leakage.
It is not subtracted, and the tiny scatter is not the uncertainty in that systematic floor.
STG's smaller aggregate BB residual does not mean a smaller bias in every bin: its highest width-250 bin has larger mean bias than ZEA's.

The corresponding mean TT residuals are +0.271 ± 0.056%, +0.245 ± 0.056%, and +0.327 ± 0.325% for ZEA, STG and SFL.
For EE they are +0.369 ± 0.062%, +0.138 ± 0.075%, and −0.123 ± 0.251%.
A single mean can conceal opposite-sign residuals in different bands: the EE rms of the per-band mean residual is 0.446%, 0.263%, and 1.790%, respectively.
These offsets remain after the known simulation-window correction; this test does not establish their cause.

At width 250, the TT variance ratios are 1.115 ± 0.061 for STG and 366 ± 92 for SFL; EE ratios are 1.307 ± 0.110 and 400 ± 87.
The design-matrix condition numbers for the joint EE/BB inverse are 1.06, 1.43 and 26.1 at width 500, rising to 1.16, 1.91 and 55.9 at width 250.
The large SFL variance is an observed property of this inverse, not proof that an optimal analysis loses those factors of information.
The broad high-ell SFL scatter is visible in `comparison.png`; the raw per-seed fitted amplitudes are retained to check other estimators.

There is no measured instrumental-noise or nonzero-input-B ensemble here.
In particular, the null-B scatter must not be advertised as an error bar for lensing BB or primordial BB.
This comparison supports a choice based on leakage control; a full noisy-survey ranking remains unmeasured.

## Footprint tuning

The mask-squared-weighted mean shear squared is 1.9502 × 10⁻⁴ for the default ZEA centre (RA 0°, dec −56°).
Optimizing its tangent point moves the centre to RA 0°, dec −57.850° and gives 1.9125 × 10⁻⁴, a 1.9% reduction.
A determinant-one diagonal affine stretch applied to ZEA, optimized jointly with its centre, gives 1.4132 × 10⁻⁴, a 27.5% reduction from the default.
Its centre is dec −57.789° and its coordinate scale factors are exp(±0.007067), approximately 1.00709 and 0.99296.
This is an optimization within the stated affine-ZEA family, **not** a proof of a globally optimal equal-area chart.
Its spectral performance has not been simulated.
The quoted affine optimization is axis-aligned, rather than a search over all affine orientations.
Repeating the geometry at Nside 64, 128 and 256 changes the quoted shear improvements by less than 0.01 percentage point.
SFL's mean shear squared is 0.034834; moving the central declination merely translates that chart, so it cannot improve its shear.
The symmetric footprint already selects RA 0° within SFL.

If the sole objective is to minimize mean shear squared over arbitrary charts, the problem is already solved: any conformal chart covering the footprint attains the global lower bound zero.
STG does so at every tangent point.
Optimizing finite-difference roundoff would produce a spurious preferred STG centre, which the code deliberately avoids.
A unique best chart therefore needs another objective, such as equal area, pixel count, magnification variation or a specified noise-weighted likelihood.

## Magnification and noise

On the default STG chart the mask-squared-weighted area magnification is 1.04556.
The local sky area of a fixed plane pixel spans 0.8312–1.0000 times its plane area over the nonzero taper.
These are coordinate effects.
For sky white noise N_sky, the local plane white spectrum is N_plane = |det J| N_sky.
A plane pixel of area A_plane covers A_sky = A_plane / |det J|, so its variance is N_sky / A_sky.
Transforming back recovers the same noise per sky area; magnification does not physically worsen the instrument depth.
The mask-weighted plane white-spectrum factor, including the plane-area weighting used for spectra, is 1.04684.
A simulation that assigns the same variance to all equal-size plane pixels in every chart would instead change the sky depth and would not be a matched-noise comparison.
Resampling correlations and pixel windows remain separate, implementation-dependent effects.

## Paper recommendation

Keep the present split recommendation: STG for leakage-limited B-mode spectra, ZEA as a pragmatic choice for TT/EE spectra.
Do not reject STG by quoting its raw EE ratio at a single acoustic trough.
Its scalar response is correctable, and the measured variance increase in this conditional broad-band test is about 9% in TT and 26% in EE, not an order-unity noise-depth penalty.
If discussing footprint tuning, state that shifting the ZEA centre changes little here; the affine alternative is a geometrical candidate, not a validated replacement.
Do not quote the large SFL variance ratios without specifying the estimator.

## Reproduction and outputs

Run `bash env.sh python3 analysis/check_choosing_projection.py` for exact scalar and joint-polarization injection tests and geometry/noise checks.
Run `bash env.sh python3 analysis/choosing_projection.py --out /sptlocal/user/cailmdaley/pol_project/choosing_projection` to regenerate the measured results and diagnostic figure.
The compact result is versioned at `analysis/figures/choosing_projection_summary.json`.
The per-seed amplitudes, full JSON, PNG and PDF are in that output directory.
Run `bash env.sh python3 analysis/paper_numbers.py --choosing-only` for `paper/choosing_numbers.tex`; the paper owner can import only the macros whose caveats fit the text.

## Independent review

A fresh Sol worker independently reproduced all 18 measurement rows from the underlying spectra, ran the exact-injection tests, and repeated the geometry at Nside 64, 128 and 256.
It found no blocking algebraic error.
Its interpretation qualifications are incorporated above: residual offsets remain, the exterior is fixed in ensemble expectation, STG is not uniformly lower-bias in every bin, the affine result optimizes a proxy within an axis-aligned family, and the variance ranking belongs to this estimator.
