# Masked E-template subtraction

The data-only test uses the SFL 1500d field, polar reflattening, a pure-E estimate, a scalar Wiener weight, subtraction at Q/U, and NaMaster pure-B spectra.
It uses no polarization data outside the analysis mask's support.
The sky remains the production unlensed E-only sky; this is a controlled leakage-plus-white-noise test, not an SPT likelihood or a forecast including lensing B sample variance.

## Estimator and masks

For grid-handed Q/U and a C2 E window W_E, the pure-E coefficient is

    E_pure(k) = -k^-2 integral [Q (d_x^2-d_y^2) + 2 U d_x d_y]
                               [W_E(x) exp(-i k.x)] dx.

The product rule gives `pure_e()` in `analysis/masked_template.py`.
In the continuum, integration by parts gives the exact response

    E_pure(k) = k^-2 FFT[W_E IFFT(k'^2 E(k'))].

A pure B input vanishes, but an E input does not have unit response near the boundary.
For wavelengths short compared with the taper, the response is approximately W_E E.
This approximation gives a leftover leakage proportional to (1-W_E) S_J[E].
Reusing the broad B-analysis taper for E therefore creates an avoidable residual; it does not measure the information limit of the mask edge.
The primary estimator uses a narrower E taper with the same observed support, and retains the broad-taper and ordinary pseudo-E estimates as controls.
There is no division by a small mask value and no use of an unobserved map extension.
The operator produces a nonlocal extension from observed coefficients, which is an estimator choice rather than extra data.

The test gates cover the E sign, pure-B annihilation, the exact product-rule E response, outside-support isolation, and regression of raw/oracle spectra to the earlier operator test.
The scalar C/(C+N) Wiener factor uses grid wavenumber and an isotropic theory prior with the flat-pixel window.
It is not the optimal position- and direction-dependent cut-sky posterior.

## Noise and sample variance

Each independent split has uniform white Q/U noise of 5 microkelvin-arcmin per component.
A template uses E estimated from its own split.
The split-cross BB power is noise-unbiased even though the template and B noise in an individual split can be correlated.
The ensemble scatter of that cross-power includes E leakage sample variance, direct B noise, and template noise.
The two splits correspond to a coadd noise level smaller by sqrt(2); they do not represent two independent 5 microkelvin-arcmin coadds.

There are two distinct residual statistics.
Writing f for the Wiener factor, L for the uncorrected leakage power and t for the local tan-squared misalignment coefficient, the ideal oracle gives

    isolated residual auto = (1-f)^2 L + f^2 t N_E,
    observed split-cross mean = (1-f)^2 L.

The isolated auto subtracts the noisy-E template from noise-FREE Q/U.
It tests the template-noise prediction but is not an observed-data BB spectrum.
For L=t C and f=C/(C+N_E), it becomes t C N_E/(C+N_E).
The cross mean lacks the template-noise term; comparing it directly with that auto noise limit would incorrectly suggest that the limit has been beaten.
The local prediction in `masked_template_report.py` uses the measured raw L for its signal term, the actual isotropic Wiener prior for f, and the independently computed geometry for t.
It omits finite-mask mode coupling and is a local approximation, not an exact bandpower forward model.

A fixed ensemble `R^{BB<-EE} C_EE` subtraction shifts the mean of BB but leaves its variance unchanged.
The primary variance comparison uses that ideal, perfectly known mean as its reference.
It also presumes that the template residual's mean can be calibrated and removed; neither arm is charged a calibration uncertainty.
The report separately gives sample RMS about zero without subtracting the template residual's mean.
A supplementary reference subtracts a fixed diagonal BB/EE ratio times each realization's noisy cross EE.
This allows EE-amplitude conditioning but is not the full non-diagonal R estimator, nor an optimal conditional estimator.
The ratio is set by the noiseless ensemble means, so its calibration uncertainty is also excluded.
Paired seed bootstraps give indicative confidence intervals on standard-deviation gains; twenty seeds do not resolve small gains precisely.

## Paper changes

The unmasked-grid test remains a test of S_J and should retain that qualification.
The masked experiment is a second validation, not a replacement for the noise-free operator demonstration.
The sentence saying subtraction comes after purification needs to specify E purification before construction and B purification after Q/U subtraction.
The present test does not establish that pure E is uniquely necessary; the pseudo-E control probes E-only leakage, while the pure-B annihilation gate explains why a pure-E estimate protects an independent B signal.
The blanket assertion that Wiener filtering never increases leakage is supported here only for the tested noise model and isotropic weight.
No statement about optimal mask-edge reconstruction, a cosmological B error bar, source-hole masks, anisotropic noise or real SPT data follows from these runs.

## Reproduction

Production is the additive `masked_template_realistic` ASTRA output in the highell universe.
Run it with `lc run masked_template_realistic -u highell --executor htcondor --profile htcondor9 --rerun-triggers code,input,params`.
Existing production outputs are unchanged.
Generate the report with `bash env.sh python analysis/masked_template_report.py --input results/highell/masked_template_realistic --out analysis/masked_template_results`.
The report refuses an incomplete or duplicated seed set; the pilot requires an explicit `--expected-seeds 1`.
Its summary stores the SHA256 of every numerical source file.
