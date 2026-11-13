# Residuals after the simulation windows

Most repeatable low-multipole TT/EE structure comes from comparing the estimator with the wrong radial reference, rather than from a missing leading projection distortion.
The original upper-band STG TT excess also contains an input-spectrum mismatch.
For BB, numerical sampling of the masked pure-B estimator explains about 30% of the STG/ZEA additive contamination at ell=4500–4800, but only about 12% at 4000–4500.
A separate synthesis-grid/finite-averaging-dependent additive B term is demonstrated at Nside4096; its size in the Nside8192 production maps remains unmeasured.
Neither result supports a universal STG floor subtraction or sub-percent agreement in every bin.

The production arrays, default loaders, geometric coupling matrix and manuscript are unchanged by this diagnostic.
All uncertainties below are standard errors of per-sky band means unless stated otherwise.
The small ring ensemble has only two skies, so its errors should not be read as a precision calibration.

## Estimator and spherical conventions

The previous window model averages the footprint-averaged geometric spectrum at integer multipoles in each width-50 bin.
The measured native spectra have already had the continuous square-pixel power window divided out at the nominal bin centre.
`residual_estimator_response.py` restores that window, applies NaMaster's actual `workspace.decouple_cell(workspace.couple_cell(k, C_native))`, and divides out the same nominal window afterwards.
The independently controlled interpolation and finite-subsample windows remain inside the coupled spectrum.
This includes discrete Fourier-cell weighting, within-bin spectral variation and the mask's radial response.

The leading large-ell spherical/planar correspondence is `P_ell(cos theta) ≈ J_0[(ell+1/2) theta]`.
With `(2ell+1) d ell = 2k dk`, the corresponding planar spectrum is `C_plane(k) ≈ C_sphere(k-1/2)`.
The diagnostic shifts the input spectrum by minus one half before applying the geometric response, without fitting an offset or changing the area normalization.
The intermediate radial and half-multipole shifts are order-dependent: the old integer average already sits half a multipole below the bin centre.
Only their net effect should be interpreted; they are not separately isolated physical pipeline mechanisms.

Applying a radial response to a footprint-averaged projected spectrum is an approximation to a spatially varying, anisotropic sky.
It is not an exact forward model of that sky and is not used to calibrate the additive BB floor.
The FFT gate of `residual_controls.py` independently verifies Fourier-cell counts, normalization and TT/EE mask coupling using a padded two-dimensional FFT covariance convolution on all four actual grids.
Its masked-map normalization agrees with NaMaster to the asserted 1e-10 tolerance, without fitting a scale.
Over ell=100–4800, the maximum coupled-bin discrepancy is 0.30% and rms discrepancies are 0.015–0.049%; the asserted envelope is 0.5%.
The independent reviewer found band-mean differences no larger than 0.006% over 1000–4000.
This gate checks the radial calculation, not BB or the homogeneity approximation.
The input limit of 8000 exceeds the native grid's Fourier-corner radius.

## Independent skies and remaining bin structure

The 260 independent skies comprise `bin_covariance_pilot` seeds 0–3 and `bin_covariance_sims` seeds 4–259.
They use independent T/E draws, zero true B, Nside8192, lmax6000 and the production interpolation, rebin8, FlattenPol, masks and pure-B estimator.
Their theory was exported from `analysis/theory_spectra.npz`.
Unit-width pseudo-spectra are Fourier-cell weighted to width50 and decoupled with the original production matrices.
These skies are never pooled with the 20 original production skies, whose pre-existing alms have a different realized input mean.
BICEP has no independent 260-sky test.

For independent ZEA TT at ell=100–500, the residual changes from −1.246±0.106% against the integer-bin model to +0.003±0.108% after the net estimator/spherical correction.
The intermediate radial-only result is +0.421±0.108%, but should not be interpreted as an independent mechanism.
Over ell=1000–4000, the net diagnostic gives these band means:

| Projection | TT [%] | EE [%] |
|---|---:|---:|
| SFL | +0.0228±0.0183 | +0.0069±0.0186 |
| STG | −0.0193±0.0132 | −0.0482±0.0128 |
| ZEA | −0.0103±0.0140 | −0.0316±0.0130 |

Small band means do not establish agreement in each bin.
The full-covariance TT test over this band has diagnostic probabilities 0.419, 0.063 and 0.239 for SFL, STG and ZEA, whereas EE gives 0.0053, 6.1e-17 and 0.0030.
STG EE retains acoustic structure with maximum absolute bin residuals 0.85% below 1000 and 0.784% over 1000–4000.
Below 1000, TT is consistent with the net reference, but STG EE is not.
`residual_controls.py` stores full-covariance tests and separate one-dimensional per-sky band-mean tests, including upper bands.

The production20-minus-independent260 difference below 1000 is consistent with the independent covariance: ZEA gives nominal p=0.44 for TT and 0.94 for EE over 100–1000.
This tests the difference between ensembles, not either ensemble against the simple model.
The independent reviewer empirically calibrated the ensemble-difference test using 2000 random 20-versus240 splits per configuration.
Thus most of the large acoustic pattern is repeatable, while the original 20 skies' additional excursions are compatible with sample variance.
The remaining EE/BB structure is not wholly sample variance.

The production BICEP-minus-ZEA contrast at 1000–4000 changes from +0.120±0.021% to +0.0585±0.0214% in TT and from +0.171±0.040% to +0.0022±0.0404% in EE.
The remaining TT contrast is unattributed.
The figures `low_ell.png` and `high_ell.png` show both ensembles and the reference operators.

## Original input spectra and the upper STG bands

`residual_input_spectra.py` measures full-sky T/E spectra from all 20 original alm files.
Passing their realized mean through the same response distinguishes an input-spectrum mismatch from finite-footprint sample variance.
The window kernels are held fixed during this small spectral-shape perturbation.

At ell=4500–4800, the production STG TT residual becomes +0.0446±0.1064%, compared with +0.4093±0.1068% against the earlier integer-bin reference.
STG EE becomes +0.1480±0.1028%, compared with +0.2782±0.1029%.
No new local alias term is required to account for the original positive TT excursion of conformal STG.
This is not a statement that every upper-band residual vanishes: independent STG TT/EE means in the same band are −0.047±0.028% and −0.073±0.029%.

After input-mean adjustment, the full-covariance production-versus-independent diagnostic over 4000–4800 gives p=0.970 for STG TT and 0.294 for STG EE; ZEA gives 0.307 and 0.322.
These use the unconditioned covariance, ignoring the small variance reduction associated with the correlated all-sky mean subtraction; they are nominal diagnostics, not exact conditional coverage claims.
The report also stores their one-dimensional tests.
The roughly −0.11% clean-band production EE shift shared by the four charts after conditioning is compatible with a finite-field realization difference, not a proven universal bias.

## Pure-E mask products and fixed-map refinement

`residual_pure_e_test` contains four periodic planar E-only realizations on each of the four production field grids.
The same native Fourier coefficients are interpolated onto a twice-finer grid before multiplying by the separately interpolated production mask and forming pure-B terms.
No new sky modes or reprojection are introduced.
Native and fine pseudo-spectra use the same original workspace only as a diagnostic normalization, not an exact refined-grid calibration.
Unmasked zero-B gates pass on both grids.

Native mask products generate a BB floor even though the input is pure E on the plane.
At ell=4500–4800 with cutoff6000, BB/periodicEE is (3.94±0.03)e-6 for STG and (5.75±0.27)e-6 for ZEA.
Refining only the mask products reduces these ratios to about 1e-12, removing more than 99.999%.
Cutoff4000 leaves nearly the same native floor at high ell, but not throughout 1000–4000.
This establishes numerical mask-product sampling leakage without spherical reprojection or power confined to the lmax6000 tail.
The pure-E report explicitly compares its dimensionful native power with production after restoring the production pixel window.

`residual_map_upsample` holds the actual projected, flattened production Q/U maps fixed and refines only their sampling before the mask and pure-B estimator.
All four seeds per projection reproduce the stored native spectra to the asserted 2e-9 relative tolerance, with observed differences near roundoff.
Periodic-crop and 64/128-native-pixel padded/tapered controls compare the paired change within each padding, rather than equating different Fourier lattices.
The taper is unity throughout the original crop; area normalization is restored before decoupling.
The 64/128-padding gate passes for every projection.
EE changes by no more than about 0.0002% in these controls.

For four-seed ZEA at ell=4500–4800, pad128 refinement changes the fractional BB residual by −2.589±0.173 percentage points, leaving +4.908±0.086% over window+alias.
Averaging dimensionful native C_ell, the removed fraction is about 30% of the STG/ZEA contamination in this band, about 12% at 4000–4500 and about 20% across 4000–4800.
The fraction depends on weighting and the reference denominator; the earlier approximately 26% ZEA figure omitted the alias term.
Prefer the direct paired percentage-point change and remaining residual for paper text.

At 4500–4800, the removed native powers are 1.32e-14 for STG, 1.87e-14 for ZEA and 3.27e-15 for SFL, respectively 0.94, 0.90 and 0.88 times their independent planar floors.
The SFL term is present but negligible relative to its geometric BB; BICEP's change is consistent with zero at this precision.
These controls explain a minority of ZEA's remaining contamination, not all of it.
The native STG C_ell floor is approximately flat over 4000–4800, while BB/EE rises because EE falls; after refinement its native C_ell decreases.
The plots `pure_e.png` and `production_bb.png` display these distinctions.

## Synthesis grid and finite averaging

`residual_ring_test` holds the original E alms, chart, footprint, output pixel centres and estimator fixed, synthesizing at Nside4096 with 8 or 16 subpixel samples per output-pixel axis.
The unchanged Nside8192/rebin8 production maps supply the baseline.
Both seeds in STG and ZEA pass native EE/BB roundtrips.
Production's tiny input B, omitted in this E-only intervention, is below 8e-14 of EE in the tested upper band and cannot explain the change.

For STG at ell=4500–4800, the two-sky mean BB/ownEE is:

| Synthesis and samples per axis | BB/ownEE | EE/production EE |
|---|---:|---:|
| Nside8192, 8 | (3.065±0.026)e-5 | 1 |
| Nside4096, 8 | (13.086±0.387)e-5 | 0.82965 |
| Nside4096, 16 | (8.847±0.242)e-5 | 0.82366 |

Doubling the sampling at Nside4096 lowers BB/ownEE by 32.39±0.15%, while EE changes by only 0.72%.
The independent reviewer checked the coherent EE change against the finite-subsample window, agreeing to roughly 1e-4.
A common multiplicative window cannot explain the B change in conformal STG.
Interpolation/finite averaging demonstrably contributes additive numerical B at Nside4096.
The ZEA two-sky mean BB/ownEE changes from 4.341e-4 in production to 5.552e-4 at 4096/8 and 5.119e-4 at 4096/16; it does not support a chart-independent floor subtraction.

The STG floor below 500 barely responds to either sampling intervention, and its 500–1000 value changes by at most about 4%.
These numerical sampling terms therefore do not explain the low-ell floor.
The plot `ring_controls.png` shows the low-ell stability and upper-band sensitivity.

No Nside8192/rebin16 control was run.
Changing Nside also changes the interpolation lattice and code path, and the high-ell spectral shapes of the Nside4096 term and refined production remainder differ.
Therefore these controls do not establish how much of the remaining Nside8192 production floor comes from finite averaging.
A paired 8192/16 intervention would be the smallest next test if a quantitative decomposition is needed; this report instead closes with that limitation explicit.

## Quoted ranges and remaining causes

Keep ell=4000 as the common upper edge for BB band-mean comparisons because of the already demonstrated projection-dependent aliases and the growing contamination relative to EE.
Do not call 1000–4000 a universally closed precision range.
ZEA's independent BB band mean against the radial-plus-spherical diagnostic is +0.273±0.025% there, with resolved bin structure; SFL's mean is −0.010±0.026%, but its bin-structure test also rejects the approximation.
STG BB is a floor diagnostic, and BICEP has no independent 260-sky test.
Do not extend the common BB lower edge below 1000 using a TT/EE result: ZEA retains strong low-ell BB structure, especially around EE troughs, that neither mask-product refinement nor the ring controls remove.
Inhomogeneous leakage and nonlocal projection/estimator response remain hypotheses for that structure, not established quantitative explanations.
TT/EE may be shown through 4800 as diagnostics, with the estimator conventions, input distinction and remaining per-bin errors stated.
No range is moved merely to improve agreement.

Ruled out as complete explanations are pure sample variance for the repeatable acoustic pattern, input cutoff changes for the lower additive ZEA/STG floor, a common coherent window for the controlled Nside4096 B term, and mask-product sampling for the entire production BB floor.
Established contributions to production residuals are the net radial/spherical reference correction, the original input-spectrum mismatch and native pure-B mask-product leakage.
A separate sampling-dependent additive B term is demonstrated only in the Nside4096 control.
Their distinction is essential: none licenses a universal production floor correction.

## Review, provenance and reproduction

Two independent Opus reviews are preserved beside the residual-anomalies fiber as `residual-review.json` and `residual-final-review.json`, with their briefs and attributed FFT/covariance scripts.
The final review accepts the FFT arithmetic, covariance tests, four-seed measurements and macro definitions, with the band-specific and native-power wording incorporated here.
Its canonical-ring follow-up, `residual-ring-review.json`, independently confirms both ZEA seeds, the final ring numbers and scoped interpretation, with no consequential blockers.

All 36 control units ran through `lc run --executor htcondor --profile htcondor9` and are canonical.
`lc verify -u highell` passed the materialized chain before the last ring gather; `check_residual_provenance.py` additionally checks all three final canonical hashes, all 36 unit hashes and upstream versions.
It compares explicit stage/theory inputs and imported stage helpers byte-for-byte with each unit's git pin, avoiding reliance on the external-file mtime/size fingerprints alone.
No manifests are rewritten by the check.
Each step is an output of `astra.yaml`, run by `lc run`:

1. `residual_input_spectra` (`analysis/residual_input_spectra.py`)
2. `residual_estimator_response` (`analysis/residual_estimator_response.py`)
3. `residual_controls` (`analysis/residual_controls.py`): the independent-sky means and tests, the FFT gate, the pure-E, map-refinement and ring controls
4. `residual_summary` (`analysis/summarize_residual_anomalies.py`)
5. `residual_numbers` (`analysis/paper_numbers/residual_numbers.py`)

The controls reproduce bit for bit only at one OpenMP/BLAS thread; more threads move the decoupled spectra and the FFT gate at the 1e-14 level and the p-values at the 1e-12 level.

The macro writer implements the bounded claims of `residual_numbers.tex`, including `residSTGEEFitP` alongside the STG EE mean and largest bin error.
The manuscript author should replace the original unexplained positive STG TT claim, qualify the clean-band EE agreement, explain the low-ell reference mismatch, and retain the numerical-BB limitation rather than claiming closure.
