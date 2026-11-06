# A mean-spectrum response does not determine bandpower covariance

A deterministic linear reprojection preserves Gaussianity, but it generally destroys statistical homogeneity on the analysis plane.
Its Gaussian covariance is determined by the full two-point function, not by the radial mean-spectrum response $R(\ell,\ell')$ alone.
In particular, $R\,\operatorname{Cov}(\widehat C)\,R^T$ is not a valid general covariance prescription for the paper's $R$.

## Exact Gaussian calculation

Write all independent sky amplitudes as a real vector $a$ with covariance $S$.
Real and imaginary harmonic components can always be put in this representation, with their appropriate variances.
Let $L$ be the linear operator comprising synthesis, projection, polarization rotation, interpolation and pixel averaging.
Let $H$ contain the mask, pure-mode construction and Fourier transform, and put $A=HL$.
A coupled bandpower is a quadratic estimator $p_b=a^T A^T Q_b A a$, where $Q_b$ selects the Fourier annulus and carries its normalization.
Writing $K=ASA^T$, Wick's theorem gives

$$
\langle p_b\rangle=\operatorname{Tr}(Q_b K),\qquad
\operatorname{Cov}(p_b,p_c)=2\operatorname{Tr}(Q_b K Q_c K).
$$

The same expression holds for TT, EE and BB by choosing the corresponding output components in $Q_b$.
The factor of two refers to independent **real** amplitudes; a complex harmonic representation must include both its covariance and its reality constraint rather than reuse this factor blindly.
For fixed linear decoupling $\widehat C=Dp$, the exact covariance is $D\operatorname{Cov}(p)D^T$.
Here $D$ can include the inverse mask-coupling matrix, bandpower rebinning and deterministic pixel-window corrections.
This last propagation is valid because $D$ acts on each realization's actual measured quadratic estimators.

For an isotropic input $S=\sum_\lambda C_\lambda P_\lambda$, the mean response is

$$
R_{b\lambda}=\operatorname{Tr}(Q_b A P_\lambda A^T).
$$

Those traces do not specify $K$ or the quadratic traces needed for the covariance.
For an exact amplitude-derived response and full-sky sample powers $\widehat C_\lambda$, isotropy gives $\mathbb E[p\mid\widehat C]=R\widehat C$.
The law of total covariance then makes the omitted term explicit:

$$
\operatorname{Cov}(p)=R\operatorname{Cov}(\widehat C)R^T
+\mathbb E[\operatorname{Cov}(p\mid\widehat C)].
$$

The conditional covariance matrix is positive semidefinite, but its **off-diagonal entries can be negative** and cancel correlations in the first term.
Thus the propagated radial sample variance is a real contribution in this exact construction, not the complete answer; a broad $R$ does not establish the sign or size of the net bin correlation.
The counterexample below demonstrates both this incompleteness and the cancellation.
The paper's local-Jacobian $R$ makes further averages over footprint position and Fourier direction, so it contains still less information.
Composing this mean response with a mask mean-coupling matrix cannot reconstruct what was averaged out.
The actual mask must be included in the amplitude operator, or its equivalent two-point function, **before** the Wick contractions.

## A counterexample with the same response

Take four independent real Gaussian sky amplitudes with variances $(1,1,4,4)$.
Average the squares of the first two output amplitudes for band 1 and the last two for band 2.
Consider two orthogonal amplitude operators:

$$
A_1 a=(a_1,a_3,a_2,a_4),
$$

$$
A_2 a=\frac{1}{\sqrt2}(a_1+a_3,a_2+a_4,a_1-a_3,a_2-a_4).
$$

For the two input groups $(a_1,a_2)$ and $(a_3,a_4)$, both have exactly the same response and means,

$$
R=\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix},\qquad
\langle p\rangle=(2.5,2.5).
$$

Their covariances differ:

$$
\Sigma_1=\begin{pmatrix}8.5&0\\0&8.5\end{pmatrix},\qquad
\Sigma_2=\begin{pmatrix}6.25&2.25\\2.25&6.25\end{pmatrix}.
$$

Thus their inter-band correlations are 0 and 0.36 despite identical $R$.
Propagating the input group-power covariance through $R$ instead gives $4.25$ for **every** entry, hence correlation 1, agreeing with neither.
`analysis/projection_covariance.py:counterexample` constructs these operators and checks the exact numbers.
`analysis/check_projection_covariance.py` also checks the Wick formula against 300,000 Gaussian draws with a nontrivial rectangular amplitude operator.

## A projection counterexample

On an unbounded plane with a spatially constant nonsingular Jacobian $J=\partial\boldsymbol{x}_{\rm plane}/\partial\boldsymbol{x}_{\rm sky}$, a homogeneous Gaussian sky remains homogeneous after remapping.
The projected Fourier covariance is diagonal in its **two-dimensional** wavevector,

$$
\langle y(\mathbf k)y^*(\mathbf k')\rangle
=(2\pi)^2\delta_D(\mathbf k-\mathbf k')\,|\det J|\,
 C_{|J^T\mathbf k|}.
$$

For a sheared field, radial averaging gives a broad $R(\ell,\ell')$, even though distinct output Fourier annuli have zero covariance without a mask.
A broad mean response therefore does not, by itself, imply broad bin correlations.
The remapping does change the diagonal variance through its angular anisotropy; the variance uses an angular average of **squared** directional power, not the square of its average.
Spatially varying geometry and the survey window can introduce off-diagonal covariance, which requires their retained position and directional dependence.

## A leading local, finite-annulus prediction

A leading local approximation can retain more information than $R$ without constructing the full amplitude operator.
Write the position- and direction-dependent native map power as $P(\boldsymbol{x},\boldsymbol{k})$.
It includes the full local Jacobian, the signed polarization rotation, HEALPix interpolation at $J^T\boldsymbol{k}$, and pixel averaging in the map frame.
For locally stationary covariance and nearby Fourier modes,

$$
K(\boldsymbol{k},\boldsymbol{k}')\simeq
\mathcal F_{\boldsymbol{x}}\!\left[W^2(\boldsymbol{x})
P\!\left(\boldsymbol{x},\frac{\boldsymbol{k}+\boldsymbol{k}'}2\right)\right]
(\boldsymbol{k}-\boldsymbol{k}').
$$

Wick's theorem squares this Fourier kernel before the annular sums.
For narrow bins of common width $\Delta$, locally straight annulus boundaries reduce the radial pair-overlap factor to

$$
T_\Delta(d_\parallel-s)=\max\!\left(0,1-\frac{|d_\parallel-s|}{\Delta}\right),
$$

where $s$ is the separation of bin centres and $d_\parallel=\boldsymbol d\cdot\widehat{\boldsymbol k}$.
Thus the relevant pair sum is the direction average of $|\mathcal F[W^2P]|^2 T_\Delta$, not an average of $P$ propagated through $R$.
The wide-bin limit recovers $W^4\langle P^2\rangle$, while finite bins retain the redistribution of that variance among neighbouring annuli.
`analysis/local_bin_covariance.py` implements the pair sums and their ratios to a homogeneous control with the same radial mean.
Its spatial arrays are reflected from native westward pixel indices into the eastward coordinates used by the analytical Jacobian before their Fourier transform.
Missing that reflection changes the finite-bin answer even though the wide-bin limit is unchanged.

This is a **leading approximation for coupled spectra**, not an exact Gaussian covariance for the purified, mask-decoupled estimator.
It omits pure-B boundary terms, reciprocal-grid alias images, higher local-stationarity corrections and annulus curvature.
The control uses the locally constant radial spectrum approximation over a mode-coupling width.
STG BB has no geometrical signal in this model and is excluded.
The exact toy-operator check is scalar; no exact EE/BB Wick trace with the actual NaMaster pure-B amplitude operator has been computed.
The BB comparisons below validate only the stated broad-band statistics of the complete pipeline, not a precision full-matrix pure-B prediction.
ZEA BB above4500 also contains alias/floor contributions absent from this leading geometrical model, so its agreement after broad-band averaging must not be read as a geometrical closure of those top bins.
`analysis/check_local_bin_covariance.py` tests finite-bin overlap against exact discrete Wick annulus-pair counts for an amplitude-modulated white Gaussian field, the lag-sum/Parseval invariant, and the uniform-mask anisotropic limit.
`analysis/check_local_covariance_orientation.py` checks spatial/directional orientation against an exact Wick trace on a sheared chart, with a wrong-reflection negative control.
The complete projected/control comparison now tests this approximation on two broad bands:2000–3000 at width100 and3500–4800 at width50.
Across the eight non-floor auto-spectra in each band, the largest fractional difference in predicted variance ratio is5.5%, and the largest absolute difference in predicted adjacent correlation is about0.02.
These are bounds on those sixteen **band-averaged summaries**, not on every covariance-matrix element or every multipole.
For high-ell SFL TT/EE/BB, predicted coupled variance ratios are3.525/3.820/5.036 against measured3.670±0.088/3.945±0.098/5.185±0.128; predicted adjacent correlations are0.145/0.154/0.226 against0.130±0.013/0.142±0.014/0.226±0.013.
The wide-bin approximation instead gives4.336/4.779/7.110, demonstrating why finite annuli matter.
`analysis/summarize_bin_covariance.py` generates the full family table and maxima in `results/highell/bin_covariance_comparison/local_summary.{json,md}`.
The predictions use stride8/nchi128, with source-content hashes in `results/highell/bin_covariance_local_prediction/prediction.json`.
A stride16/nchi64 sweep changes TT/EE variances by at most0.14% and correlations by0.0011; SFLBB's worst per-bin changes are1.8% and0.003.

## Complete projected and homogeneous ensembles

The final measurement uses260 newly synthesized independent Gaussian skies per chart, not the old production20.
`bin_covariance_pilot` contains seeds0..3 and `bin_covariance_sims` contains4..259, all synthesized at Nside8192/lmax6000, reprojected with production rebin8/FlattenPol, and measured with the same1500d mask and pure-B estimator.
T and E are independent draws, so these runs test each auto-spectrum covariance, not TE or TT–EE cross-covariance.
All gathered and unit content hashes passed direct verification against their manifests: pilot4 units, extension256 units and controls16 batches.

`bin_covariance_flat_controls` contains256 independent homogeneous Gaussian skies per chart.
Their radial TT/EE/BB input means come from the production20, keeping the native sampling window; T, E and B are independent in each control.
The masks, pure-B estimator, unit-width pseudos and mask-response decoupling match the projected ensemble.
NaMaster1.5.1 initializes OpenMP workers with base-seed+thread, so consecutive base seeds reuse Fourier streams; `control_seed` separates bases by1024, and an executable four-thread gate rejects shared Fourier rows.
No grid control was drawn with the invalid adjacent-base-seed scheme.

`analysis/compare_bin_covariance.py` retains both coupled and mask-decoupled widths30/50/100 and lag separations of one, two and four bins.
Errors delete **whole skies** independently in the projected and control ensembles, retaining all dependence among band pairs.
Pointwise bin-pair curves are also saved; their errors are not simultaneous confidence bands.
All-off-diagonal averages cannot exclude local correlations: with $N$ bins and correlation $r$ only between immediate neighbours, that average is $2r/N$.

At width50 over3500–4800 the mask-decoupled results are:

| Chart / spectrum | Fractional-variance ratio to homogeneous | Adjacent rho, projected | Adjacent rho, homogeneous | Excess rho |
|---|---:|---:|---:|---:|
| ZEA TT |0.992±0.024|−0.052±0.012|−0.053±0.011|0.001±0.016|
| ZEA EE |1.021±0.026|−0.021±0.013|−0.042±0.011|0.021±0.017|
| ZEA BB |3.111±0.080|0.080±0.012|−0.020±0.014|0.100±0.018|
| SFL TT |3.612±0.086|0.060±0.013|−0.044±0.013|0.104±0.018|
| SFL EE |3.892±0.094|0.072±0.014|−0.021±0.012|0.093±0.019|
| SFL BB |5.056±0.123|0.159±0.013|−0.019±0.011|0.178±0.017|
| STG TT |1.008±0.025|−0.024±0.012|−0.042±0.011|0.018±0.016|
| STG EE |1.060±0.028|−0.043±0.013|−0.028±0.013|−0.015±0.018|

The fractional-variance ratio averages $(\operatorname{Var}\widehat C_b/\langle\widehat C_b\rangle^2)_{\rm projected}/(\operatorname{Var}\widehat C_b/\langle\widehat C_b\rangle^2)_{\rm homogeneous}$ over bins, rather than taking a ratio of range-averaged powers.
Its small finite-control inverse-variance bias is not removed (approximately0.8% in the Gaussian-bandpower limit, below these errors).
Positive deterministic per-bin power rescalings cancel exactly from correlations and fractional variance.
This matters because the high-ell projected/control mean ratios are0.9965/0.9982/1.0003 for SFL TT/EE/BB and0.9875 for ZEA BB; their small mean mismatches are measured, not assumed away.
The controls are matched radial-spectrum diagnostics, not perfectly identical mean spectra.

Mask decoupling changes the absolute adjacent correlations: high-ell coupled SFL TT/EE/BB give0.130±0.013/0.142±0.014/0.226±0.013, while coupled ZEA EE/BB give0.062±0.013/0.161±0.011.
The projected-minus-control excess is more stable across these two estimators.

STG BB's adjacent correlation is0.440±0.012 in controls **without any projection**, compared with0.380±0.013 in projected skies.
Its projected/control mean ratio is0.8844±0.0009, so it is not a clean matched-B signal comparison.
An estimator E-to-B residual is plausible but not isolated; exclude this numerical floor from geometrical BB conclusions.
The ensembles are unlensed and noise-free, without source holes or time-domain filtering.
They test the covariance consequences of position/direction dependence under the same simple mask, not SPT's complete pipeline.

Final artifacts are `results/highell/bin_covariance_comparison/{comparison.json,moments.npz,comparison.png,binwise.png}`.
The old20,13-sky and44-sky artifacts are superseded for final measurement claims.
Run `analysis/paper_numbers.py --covariance-only` for the generated `paper/covariance_numbers.tex`; it refuses incomplete or old seed sets.
The central writer emits only macros cited by `paper/main.tex`, so add the intended macro references there and regenerate; an unquoted covariance result is deliberately omitted from the LaTeX file.
`analysis/bin_covariance_numbers.py:numbers()` returns the complete validated dictionary independently of that publication filter.

## The SPT references

Balkenhol et al. (2023), arXiv:2212.05642, Appendix A.2, describes a 1000-sky CMB-only Monte Carlo measurement, with the HEALPix sky masked before projection.
The paper states that the correlation structure is dominated by projection and replaces its band-diagonal estimates with second-order polynomial fits.
The covariance revisions broaden cosmological parameter uncertainties by up to 15%.
Section IV.A says projection-induced correlations fall to at most 20% only for separations of at least $\Delta\ell=100$; it uses wider bins for the null tests.
Figure8's caption also states that the minimum-variance TT/TE/EE data bandpowers are correlated by **up to40% for neighboring bins**.
This provides a quantitative comparison target within its multipole range, not a universal amplitude for every bin, and does not identify $R$ as its source.
Table III uses bins of width100 above2000, whereas its lower-multipole bins have width50.
Thus the matched-binning comparison over2000–3000 uses our width100 results; our width50 extension to4800 is a separate diagnostic.
`analysis/compare_bin_covariance.py` explicitly includes2000–3000 alongside the extension to4800, with both widths retained.

Camphuis et al., arXiv:2506.20707, Section III.B, explicitly says excess high-$\ell$ bin correlations that were difficult to compute analytically motivated the move to full spherical harmonic transforms.
It does not provide a quantitative excess-correlation curve in that discussion.
The downloaded PDFs and searchable text are under `/sptlocal/user/cailmdaley/pol_project/bin_correlations/literature/`.

## Required paper qualification

Replace the sentence expecting the wide rows of $R$ to contribute to correlations with the distinction between a mean-spectrum response and a mode-amplitude response.
The quoted ordinary-sky conclusion from a disk-averaged off-diagonal statistic should not be read as an isotropic Gaussian covariance prescription for the 1500d field.
The all-off-diagonal average can be small even when adjacent bands are substantially correlated, so it should not support the phrase "no more correlated than those of an ordinary sky" without a matched control.
A defensible replacement is:

> Linear reprojection preserves the Gaussianity of the unlensed input, but its covariance on the plane need not be statistically homogeneous.
> The mean-spectrum response $R$ does not determine the bandpower covariance: that calculation requires the full anisotropic two-point function, with the mask and mode purification included before the Gaussian contractions.

The matched-binning2000–3000 comparison at width100 gives ZEA adjacent TT/EE correlations−0.016±0.020/−0.019±0.022 after mask decoupling, with projected-minus-control excess0.044±0.029/−0.008±0.031.
The corresponding coupled TT/EE values are0.031±0.020/0.027±0.022.
The largest individual adjacent-pair estimates in this range are0.083/0.067 for decoupled TT/EE and0.133/0.110 for coupled TT/EE, with per-pair errors around0.06; these maxima are descriptive, not simultaneous confidence limits.
Thus our simple-mask Lambert experiment does **not** reproduce an up-to40% neighbouring TT/EE correlation in that range.
This is not a contradiction of SPT's results: their maximum need not occur in this band; their mask, masking-before-projection, filtering and minimum-variance multifrequency/TE estimator differ, and our input is unlensed.
The exact source of their reported excess remains unassigned.
The leading local model reproduces the broad-band **coupled** variance ratios and neighbouring correlations measured here, not SPT's full covariance.

For the paper, replace the disk-averaged ordinary-sky assertion and its20-sky field follow-up with the260/256 comparison, stating the auto-spectrum and simple-mask scope.
Replace the inference from broad $R$ rows with the exact mean-versus-covariance distinction above.
If space permits, give the leading $\mathcal F[W^2P]$ finite-annulus calculation and its broad-summary accuracy as a separate approximation, not an exact $R$-based covariance prescription.
