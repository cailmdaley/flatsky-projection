# Projection geometry in a temperature lensing estimator

A flat-sky temperature quadratic estimator (TT QE) responds to the spherical sky's covariance expressed on the analysis grid.
An auxiliary reference projection, such as TAN, is unnecessary.
The tested quantities are the estimator's mean field and its response to controlled grid remappings; neither is a measurement of cosmological lensing-power bias.

## Metric and local covariance

Let $J$ map orthonormal sky displacements to east/north-oriented grid coordinates, as in `analysis/geometry.py`.
The sky metric on the grid is $G=J^{-T}J^{-1}$.
For small-scale modes within a locally affine patch,

$$
P(\boldsymbol x,\boldsymbol\ell)
=|\det J(\boldsymbol x)|\,
 C\!\left(\sqrt{\boldsymbol\ell^TJJ^T\boldsymbol\ell}\right).
$$

A rotation of the orthonormal sky frame leaves $JJ^T$ unchanged.
The relevant positive stretch in **grid axes** is $\Sigma_g=\sqrt{JJ^T}$, rather than the sky-axis factor $\sqrt{J^TJ}$.
Both have the same singular values, but a directional covariance needs the shear orientation as well as its magnitude.
Define

$$
\Sigma_g^{-1}=(1-\kappa_m)I-
\begin{pmatrix}\gamma_1&\gamma_2\\\gamma_2&-\gamma_1\end{pmatrix}.
$$

At first order, with $n_\ell=d\ln C_\ell/d\ln\ell$,

$$
\delta P/C_\ell=(2+n_\ell)\kappa_m
+n_\ell[\gamma_1\cos2\theta_\ell+\gamma_2\sin2\theta_\ell].
$$

The local rotation does not enter the covariance of an isotropic scalar sky.
The implementation retains the **exact nonlinear affine spectrum** for the mean-field prediction, then applies an approximate squeezed QE score.

The FFT assumes a homogeneous isotropic covariance on its Euclidean grid; it does not implicitly select an isotropic sky on TAN.
TAN itself is a counterexample to the old reference-chart interpretation: its TAN-relative displacement vanishes, while its unequal radial and tangential stretches produce an anisotropic local covariance.
STG has the opposite diagnostic: its intrinsic shear vanishes, although its displacement relative to TAN has shear.
The fields of $P_{\rm TAN}\circ P^{-1}-I$ remain legitimate chart-comparison quantities, but their rotation RMS is not a prediction for the curl estimator.

## Gradient, curl and the mask

Use $d=\nabla\phi+(-\partial_y,\partial_x)\Omega$.
Both estimator outputs are expressed as $L^2/2$ times the corresponding potential, so the gradient output is minus half the divergence of $d$, and the curl output is minus half its scalar curl.
For a compatible displacement on a periodic plane, the shear obeys

$$
\gamma_E=\frac{L^2}{2}\phi=\kappa,\qquad
\gamma_B=\frac{L^2}{2}\Omega.
$$

An arbitrary spherical metric need not be the pullback of one globally Euclidean displacement.
In the weak-distortion, squeezed limit, an isotropically filtered gradient QE measures

$$
\langle\hat\kappa\rangle=
\frac{I_0\kappa_m+I_2\gamma_E}{I_0+I_2},\qquad
\langle\hat\omega\rangle=\gamma_B,
$$

where $I_0=\int \ell d\ell\,(2+n_\ell)^2$ and $I_2=\tfrac12\int\ell d\ell\,n_\ell^2$ for the noise-free inverse-variance weights used here.
This is a filter-dependent contraction of the metric, not automatically the potential of a specified coordinate displacement.

The Fourier E/B decomposition on a cut field is not unique without boundary conditions.
Multiplication by a nonround mask mixes shear E into pseudo B.
A zenithal projection has radial intrinsic shear and no B under a rotationally symmetric extension, but this does not guarantee a zero curl reconstruction on the 1500d box.
For ZEA, the measured pseudo-B fraction of $W\gamma$ is 0.40134 on the field and $1.18\times10^{-9}$ on the round grid taper.
STG's shear is numerically zero; a B fraction of its finite-difference residual is meaningless.
SFL and BICEP's field pseudo-B fractions are 0.49420 and 0.20950, so the TAN-relative ranking of their rotations does not predict their reconstruction ranking.

## Estimator and simulation experiment

`analysis/lensing_qe.py` implements the usual symmetrized TT scores using real-space products and FFTs.
The normalization includes both Fourier-leg cuts and the full $2\times2$ finite-grid gradient/curl response matrix.
The independent direct Fourier sum agrees with the convolution response at numerical precision, and the 64-realization flat-sky perturbative test recovers unit diagonal response with off-diagonals consistent with zero.

`src/stages/lensing_qe_sims.py` generates twenty Gaussian isotropic spherical TT skies per chart, at Nside2048 and sky band limit 4000.
The four-arcminute grids cover a padded $90^\circ\times60^\circ$ canvas; the sky mask is the same C2-tapered $100^\circ$ RA by $28^\circ$ declination field as the paper.
The estimator uses noise-free TT legs $300\le\ell\le1200$, with no beam or instrumental filtering.
All charts share the same spherical realization for a seed.

For the response, a fixed potential has convergence RMS 0.003 in each of $30\le L<60$, $60\le L<120$, and $120\le L<240$.
Its gradient or curl defines a displacement on the analysis grid.
The corresponding lens on the sphere is $P^{-1}\circ(I+d)\circ P$, sampled on the spherical sky **before projection**.
Paired positive and negative injections cancel the baseline mean field and isolate the odd response.
These are controlled **grid-potential** injections; they are not realizations of physical full-sky gradient lensing, and different charts do not share one physical lens.

An independent flat Gaussian TT ensemble uses the identical grid, mask and estimator.
Its signed perturbations are computed from the Fourier derivatives of the flat field, avoiding interpolation error in the normalization control.
The reported diagonal response ratio is the spherical injection response divided by the flat diagonal for the same injected channel.
Off-diagonals are not fully matrix-calibrated leakage measurements; the flat off-diagonals are reported separately.

The projection-only mean field is the spherical ensemble mean minus the flat-control ensemble mean.
For each Fourier band its squared amplitude is estimated by the cross-seed U-statistic

$$
\widehat{|\mu|^2}
=\frac{|\sum_s q_s|^2-\sum_s|q_s|^2}{N(N-1)},
$$

where $q_s$ is the spherical-minus-flat reconstruction.
Band powers are sums over retained Fourier modes, divided by the canvas average of $W^4$; they are **not** $C_L$ estimates.
Errors use leave-one-seed-out jackknife resampling.
The report distinguishes spherical reconstruction noise from the larger variance of the difference with the independent flat control.

## Measured results and their scope

The twenty-seed diagonal response ratios at $30\le L<60$ are:

| Projection | Gradient / flat | Curl / flat |
|---|---:|---:|
| SFL | $0.472\pm0.016$ | $0.692\pm0.018$ |
| STG | $0.864\pm0.014$ | $0.886\pm0.018$ |
| ZEA | $0.960\pm0.016$ | $0.978\pm0.020$ |
| BICEP | $0.564\pm0.014$ | $0.741\pm0.018$ |

These are measured response changes in this estimator with its unprojected isotropic filters, not biases in an SPT analysis.
A Nside4096/two-arcminute control tests interpolation and output-grid sampling; its result belongs beside these numbers before precise continuum claims are made.
The response test does not include RDN0, cosmological-lensing sample variance, polarization, anisotropic filtering or instrumental noise.
One cannot square these ratios to infer a cosmological lensing-power bias.

The nonlinear affine/squeezed mean-field prediction has no fitted parameters.
For $2\le L<30$, measured/predicted curl template amplitudes are $0.961\pm0.006$ (SFL), $0.958\pm0.018$ (BICEP), and $0.997\pm0.074$ (ZEA).
The SFL correlation is 0.9984, but its four-percent amplitude discrepancy is real at the ensemble precision.
The predictor approximates both finite-$L$ leg support and filtering near the mask boundary; exact normalization alone does not make it an exact mean-field calculation.
STG's curl mean field is consistent with zero.

The curl mean field is concentrated at low $L$, but there is no cutoff at ten or thirty.
SFL has noise-subtracted curl pseudo powers $(4.03\pm0.32)\times10^{-4}$ at $30\le L<60$ and $(1.34\pm0.34)\times10^{-4}$ at $60\le L<120$.
ZEA has a detected curl mean field at $10\le L<30$, consistent with its windowed intrinsic shear.
BICEP has a smaller low-$L$ curl mean field than SFL, contrary to the TAN-rotation prediction.
No universal statement that a cylindrical grid “fails the curl null” follows: its significance depends on the estimator, mask, $L$ cuts, mean-field subtraction and calibration.

## Reproduction and paper changes

Run the scientific Python commands under the SPT build environment:

```bash
ENV=$HOME/code/python/spt3g_software/build/env-shell.sh
lc run --executor htcondor --profile htcondor9 lensing_qe_sims -u highell
lc run --executor htcondor --profile htcondor9 lensing_qe_resolution -u highell
$ENV python3 analysis/check_lensing_qe.py
$ENV python3 analysis/lensing_geometry.py --out results/lensing_qe_geometry
$ENV python3 analysis/lensing_mean_prediction.py --out results/lensing_qe_report/mean_prediction.npz
$ENV python3 analysis/lensing_report.py --prediction results/lensing_qe_report/mean_prediction.npz --out results/lensing_qe_report
```

The authoritative measured table is `results/lensing_qe_report/summary.json`.
`analysis/lensing_numbers.py:numbers()` supplies macros for the sole `numbers.tex` writer, `analysis/paper_numbers.py`.
No manuscript edit belongs in this thread.

The required manuscript changes are:

- Replace the assertion that no estimator has been run with the explicitly scoped noise-free TT experiment.
- Remove TAN as the QE's source plane, the identification of TAN-relative $\omega_{\rm rms}$ with curl response, and the claim that a symmetric composite Jacobian guarantees a masked curl null.
- Remove Table C.1 as an estimator prediction, including its comparison with post-Born rotation.
  A chart-comparison table would need a separate purpose to justify retaining it.
- Replace the BICEP-above-SFL curl ranking with the measured opposite ordering for this field and estimator.
- Describe direct mean-field power as low-$L$ concentrated with measurable tails, rather than confined below the science range.
- Retain and strengthen the recommendation that projection belongs in the lensing simulations: both the mean field and the response depend on it.

Map-level polarization template subtraction is a separate calculation in `analysis/template_subtraction.py`.
Claims about connected higher-point functions should use the exact covariance and Gaussianity arguments, not the retired TAN-potential identification.
