# Residuals after the introduced simulation windows

The large low-multipole TT/EE structure is mostly a mismatch between the reference curve and the actual estimator, rather than an unmodelled projection distortion.
A second additive numerical BB term comes from sampling the masked pure-B estimator on the native grid.
Neither finding justifies a universal floor subtraction or a claim of sub-percent agreement in every band.
The production arrays, default loaders, geometric coupling matrix and paper source are unchanged by this diagnostic.

## Reference operators

The previous window analysis compares production to the geometric coupling matrix R, averaged at the integer multipoles in each width-50 bin.
The native measured spectra have already had the continuous square-pixel window divided out at the nominal bin centre.
The diagnostic here restores that window, applies NaMaster's actual `workspace.decouple_cell(workspace.couple_cell(k, C_native))` response, and only then divides out the same nominal window.
This accounts for discrete Fourier-cell weighting, variation of the spectrum within a bin, and the mask's radial bandpower response.
The independently controlled interpolation and finite-grid windows remain inside the coupled spectrum.
They are held fixed when testing the small spectral-shape perturbations below.

There is also a distinction between a spherical harmonic label and a local planar wavenumber.
At small separation, the leading large-ell Legendre/Bessel correspondence is `P_ell(cos theta) ≈ J_0[(ell+1/2) theta]`.
Changing variables in the spherical covariance, `(2ell+1) d ell = 2k dk`, gives `C_plane(k) ≈ C_sphere(k-1/2)`.
The diagnostic therefore shifts the input spectrum by minus one half before applying R; it does not shift the measured bandpowers, fit an offset, or double the area Jacobian.
The corresponding spin-2 corrections start beyond this leading large-ell correspondence.
The independent review establishes that the attribution between these stages is order-dependent: the original integer average already samples a half-multipole below the bin centre.
Consequently the radial-response-only stage introduces part of the convention difference that the spherical term then removes.
The net correction is physical; the two intermediate shifts must not be described as separately isolated pipeline mechanisms.
The reviewer independently reproduced the radial closure with a padded two-dimensional FFT forward model and the actual Fourier-cell counts.

`analysis/residual_estimator_response.py` keeps the simple integer-bin average, the radial estimator response, and the additional spherical correspondence as separate arrays.
`analysis/check_residual_fft_response.py` independently tests Fourier-cell counts, normalization and the radial response against a padded two-dimensional mask-covariance convolution on all four actual grids.
Its direct masked-map Fourier normalization agrees with NaMaster to the asserted1e-10 tolerance without fitting a normalization.
The maximum per-bin coupled-response discrepancy is0.30% over100–4800, with an asserted0.5% envelope; the rms discrepancies are0.015–0.049%.
This replaces a non-asserting constant-spectrum identity diagnostic, which did not test the physical response.
The production pixel-grid corner radius is below the diagnostic's input limit8000.
Its high-multipole radial continuation is only an approximation to the full anisotropic, sampled map.

Applying the radial response to a footprint-averaged projected spectrum is **not** the exact response of a spatially varying, anisotropic projected sky.
The approximation works particularly well for TT and EE, but it is not used to replace the BB forward model or calibrate its additive floor.
Remaining structured EE residuals and low-ell BB differences explicitly limit that interpretation.

## Independent means and sample variance

The independent260-sky ensemble is `results/highell/bin_covariance_pilot` seeds0–3 plus `bin_covariance_sims` seeds4–259.
Its E and T draws are independent, with zero true B, Nside8192, lmax6000, the production interpolation/rebin8/FlattenPol settings and the same field masks and pure-B estimator.
Its theory array was exported from `analysis/theory_spectra.npz`.
It is never pooled with the original20 skies, whose alms came from the pre-existing production simulations.
Unit-width pseudo-spectra are Fourier-cell weighted back to the production width50 and decoupled with the original production matrices.
No new science maps are synthesized locally.

For the independent260, ZEA TT over100–500 has residual−1.246±0.106% against the integer-bin reference.
The radial response alone leaves+0.421±0.108%; including the spherical correspondence leaves+0.003±0.108%.
Over1000–4000 the latter gives the following fractional residuals, in percent.
The uncertainties are standard errors of the per-seed mean over the specified band, not errors obtained by treating the bins as independent.

| Projection | TT [%] | EE [%] |
|---|---:|---:|
| SFL | +0.0228±0.0183 | +0.0069±0.0186 |
| STG | −0.0193±0.0132 | −0.0482±0.0128 |
| ZEA | −0.0103±0.0140 | −0.0316±0.0130 |

These are bounded band means, **not** a per-bin closure claim.
STG EE still has resolved acoustic structure, including a maximum absolute mean residual of roughly0.85% below1000 and0.78% over1000–4000 against this radial-plus-spherical diagnostic.
On the same260 skies, ZEA and STG TT below1000 are consistent with the full band covariance after these operators, whereas STG EE is not.
The residual below1000 is therefore neither wholly sample variance nor wholly explained by the radial model.
Over1000–4000 the full-covariance probabilities for TT are0.419,0.063 and0.239 for SFL,STG and ZEA, whereas EE gives0.0053,6.1e-17 and0.0030.
These diagnostic probabilities use the radial-plus-spherical reference, not an exact inhomogeneous model.
For example, STG EE has a band mean−0.048±0.013% and a maximum bin residual0.784%; its small mean does not imply closure.
The report also prints and stores one-dimensional per-sky band-mean tests, which need not agree with tests of bin structure.

The production20-minus-independent260 low-ell differences are consistent with the measured band covariance: for ZEA100–1000, the approximate covariance-corrected F-test gives p=0.44 for TT and0.94 for EE.
These are diagnostic probabilities, not exact finite-sample coverage claims for arbitrary non-Gaussian projected spectra.
They test a difference of ensembles, not whether either ensemble agrees with the original simple model.
`analysis/report_residual_means.py` prints both tests separately.
The independent-sky comparison shows that most of the large acoustic pattern is repeatable, while the extra excursions of the original20 are consistent with sample variance.

After conditioning on the original all-sky input mean, the production20-versus-independent260 full-covariance test over4000–4800 gives p=0.970 for STG TT and0.294 for STG EE; ZEA gives0.307 and0.322.
That conditions on the input *mean*, not on its finite-footprint realization, and does not prove that every original upper-band excursion is an input-spectrum effect.

The paired production BICEP−ZEA contrast over1000–4000 changes from+0.120±0.021% to+0.0585±0.0214% in TT, and from+0.171±0.040% to+0.0022±0.0404% in EE.
The remaining roughly0.06% TT projection contrast is not attributed here.

## Original input spectra and STG's upper bands

`analysis/residual_input_spectra.py` reads the T and E alms of **all20** original production skies and computes their full-sky power spectra.
Conditioning the reference on their realized mean spectrum is distinct from measuring power on the masked field.
It can remove a mismatch of the supplied input spectrum without removing finite-footprint sample variance.

With that input mean passed through the same radial-plus-spherical response, STG4500–4800 TT is+0.0446±0.1064%, compared with+0.4093±0.1068% against the earlier integer-bin reference.
STG EE is+0.1480±0.1028%, compared with+0.2782±0.1029%.
The upper-band TT excess is thus accounted for without adding a local alias term to conformal STG.
There is no evidence here that its remaining EE difference requires a new mechanism.
The roughly−0.11% lower-band EE shift shared by all four original charts after conditioning is compatible with a finite-field realization difference; it is not declared a proven bias.

## Pure-E mask-product controls

`residual_pure_e_test` is an additive lightcone/OSG output with four independent periodic planar E-only realizations on each of the four production field grids.
The periodic input has BB/EE below1e-30 in the tested circles before masking.
The same native Fourier coefficients are interpolated onto a twice-finer periodic grid before multiplying by the production mask and computing its pure-B terms.
The mask is interpolated separately; no new sky modes or projection are introduced.
An unmasked zero-B gate passes on both grids.
Native and refined pseudo-spectra use the **same original workspace only as a diagnostic normalization**, not a claimed exact refined-grid calibration.

Native mask products generate high-ell B even though the input is exactly pure E on the plane.
For cutoff6000, STG4500–4800 gives BB/periodicEE=(3.94±0.03)e-6 and ZEA gives(5.75±0.27)e-6.
Refining only those products reduces them to about1e-12, removing more than99.999%.
Truncating the input at4000 leaves almost the same native high-ell floor.
This isolates numerical mask-product/derivative sampling leakage rather than spherical reprojection or input power confined to the6000 tail.
It does not imply that every BB floor in production has this origin.

## Fixed production-map intervention

`residual_map_upsample` keeps the actual projected, flattened production Q/U maps fixed.
Their native spectra must reproduce the stored production bandpowers before interpretation.
Fourier refinement by two changes only their sampling before the mask and pure-B estimator.
Periodic-crop and64/128-native-pixel padded/tapered versions distinguish the response from an interpolation artefact at a discontinuous crop boundary.
The taper is unity throughout the original crop and hence wherever the production mask is nonzero.
A change in map area is normalized out before using the original workspace.
The refined/padded outputs remain diagnostic, not replacement production bandpowers.

The four-seed ZEA4500–4800 paired change is about−2.59 percentage points relative to the window-plus-alias model, leaving about+4.91%.
The same intervention barely changes SFL or BICEP BB.
All four projections pass the native production roundtrip and64/128-padding paired-change gates.
Using the mean native C_ell over4500–4800, the removed fraction is30.8% of STG's floor and32.5% of ZEA's excess over window+alias.
Other weightings give different fractions: in particular, the earlier independent review's approximate26% ZEA used a different excess definition.
The direct paired change and remaining fractional residual above are unambiguous and are the preferred paper numbers.
Thus the pure-B mask-product sampling effect explains about one third of this upper-band additive contamination, but it does not close ZEA.
The detailed final figures and generated summary carry the exact four-seed values and the padding checks.
Subtracting STG, either before or after refinement, is a partial comparison and is never treated as a universal floor calibration.

## Ruled out, and still open

The preceding window thread established the coherent interpolation and finite-subsample windows and independently verified the geometric sampling aliases with a paired spherical cutoff test.
Those effects are retained, not redone here.
Changing that cutoff does not remove ZEA's lower additive floor.
This thread additionally rules out pure sample variance as the origin of the repeatable large TT/EE acoustic pattern, resolves most of that pattern with the estimator and spherical correspondence, and accounts for the original upper STG TT excess through its actual input spectrum.
It isolates a separate additive pure-B mask-product sampling term, including its effect on the actual production maps.
It does **not** establish a universal STG subtraction, exact sub-per-mille TT/EE closure, or the mechanism of the remaining low-ell BB and high-ell additive ZEA residual.
Nonlocal projection terms and incoherent interpolation/subsample response remain candidates rather than measured explanations.

Retain4000 as the common upper edge for the paper's BB band-mean comparison, for the pre-existing reasons of projection-dependent aliases and the rising non-common additive floor.
Do not call1000–4000 a universally closed precision range: covariant tests still resolve sub-percent BB structure, STG is only a floor, and BICEP has no independent260-sky test.
Do not extend the lower edge below1000 on the strength of a TT/EE diagnostic: the BB local-affine model still has resolved low-ell structure.
TT/EE can be shown through4800 as diagnostics provided the estimator response, input-spectrum distinction and remaining per-bin deviations are stated.
No range is moved merely to improve agreement.

## Independent review and unfinished validation

The native Opus review is preserved beside the residual-anomalies fiber as `residual-review.json`, with its exploratory FFT and covariance scripts attributed to that review.
It accepts the TT/EE net closure and planar pure-E control with the qualifications above, independently verifies the production-map pairing, and rejects attribution of the entire BB floor to mask-product sampling.
The report already treats that term as a minority contribution.
The review confirms empirical calibration of the low-ell ensemble-difference test with2000 random20-versus240 splits per configuration.
It also finds statistically resolved EE and BB bin structure that small band means conceal.

Before scientific closure, finish the detached grid jobs and ring/subsample report, regenerate the compact summary from complete canonical outputs, and add executable gates.
The constant-response diagnostic currently records an error without asserting; replace or supplement it with an independently checked Fourier-cell/FFT forward gate rather than calling it a complete response validation.
The pure-E report now compares planar contamination with the production excess explicitly in native C_ell units, with the pixel window restored only on the production side.
The mean report uses `radial_model` rather than `exact`, tests the upper bands and stores both one-dimensional band-mean and full-covariance tests plus the realized-input control.
`paper_numbers.py --residual-only` generates the bounded claims in `paper/residual_numbers.tex`; the manuscript itself is unchanged.

## Reproduction

Run science under `~/code/python/spt3g_software/build/env-shell.sh` after activating CVMFS py3-v5, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for the local reports.
The additive OSG commands are `lc run --executor htcondor --profile htcondor9 residual_pure_e_test residual_map_upsample residual_ring_test -u highell`.
A fresh worktree must first verify and reuse the canonical upstreams and their existing archives, as documented in the OSG troubleshooting fiber; otherwise Snakemake can schedule legacy upstream units that no longer exist separately.
The tested controls carry explicit implementation/theory input fingerprints.

Local reports, in order:

- `analysis/residual_input_spectra.py`
- `analysis/residual_estimator_response.py`
- `analysis/report_residual_means.py`
- `analysis/check_residual_fft_response.py`
- `analysis/report_residual_pure_e.py`
- `analysis/report_residual_map_upsample.py`
- `analysis/report_residual_rings.py`
- `analysis/summarize_residual_anomalies.py`
- `analysis/plot_residual_anomalies.py`
- `analysis/paper_numbers.py --residual-only`

Large arrays and logs use `/sptlocal/user/cailmdaley/pol_project/residual_*`.
The compact summary and inspected plots live in `analysis/figures/residual_anomalies/`.
