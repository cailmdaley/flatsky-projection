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

## Existing field measurement

`analysis/projection_covariance.py` reads the finest-spectrum thread's exact rebinnings of the production 20 skies.
It measures average lag correlations at widths 30, 50 and 100 and separations of one, two and four bins.
Its uncertainty deletes **whole sky realizations**, preserving all dependence among bin pairs.
This does not make 20 realizations sufficient for a full covariance matrix; it supplies uncertainty on a specified average statistic.
These are decoupled, pure-B NaMaster bandpowers, not raw annular pseudo-powers.

At width 50 over $3500\leq\ell<4800$, the measured adjacent correlations and delete-one jackknife standard errors are:

| Projection | TT | EE | BB |
|---|---:|---:|---:|
| ZEA | −0.091 ± 0.042 | 0.010 ± 0.060 | 0.135 ± 0.055 |
| SFL | 0.033 ± 0.053 | 0.065 ± 0.047 | 0.196 ± 0.061 |
| STG | −0.153 ± 0.036 | −0.053 ± 0.039 | 0.382 ± 0.044 |

These numbers are preliminary, generated in `results/highell/bin_correlation_diagnostic/measured.json`; the table is a research note, not paper-number source.
STG's BB lies at the numerical floor and does not measure shear-induced B covariance.
The SFL BB trend is visible, but these decoupled TT/EE products do not establish the source of the large correlations described by SPT.
The table's extension above3500 is outside the2018 SPT spectra's range, which stops at3000; it must not be presented as a direct reproduction of that analysis.
Neither an all-off-diagonal average nor this preliminary table settles agreement with SPT's different estimator and mask.
An average over **all** off-diagonal entries is particularly unsuitable for excluding local band correlations: with $N$ bins and correlation $r$ only between immediate neighbours, that average is just $2r/N$.
For 86 bins, even adjacent correlation 0.2 contributes only 0.0047 to the all-off-diagonal average.

`analysis/check_coupled_correlations.py` checks the estimator distinction on the same 20 skies.
Without mask decoupling, ZEA's high-$\ell$ adjacent correlation is $0.092\pm0.060$ for EE and $0.213\pm0.053$ for pure BB; SFL gives $0.134\pm0.045$ and $0.261\pm0.058$.
Mask decoupling thus matters numerically, but these coupled products still do not establish a large positive ZEA EE excess over a matched Gaussian control.

## Matched homogeneous Gaussian controls

`bin_covariance_flat_controls` contains256 independent homogeneous Gaussian skies per chart, materialized through `lc` on OSG.
All canonical and sixteen unit content hashes were checked against their manifests.
Their radial TT/EE/BB input means come from the production20, keeping the native sampling window; T, E and B are independent in each control.
The masks, pure-B estimator, unit-width pseudos and mask-response decoupling match the projected ensemble.
NaMaster1.5.1 initializes OpenMP workers with base-seed+thread, so consecutive base seeds reuse Fourier streams; `control_seed` separates bases by1024, and the executable gate pins four actual OpenMP threads and requires all inter-sky Fourier-row overlaps below0.9.

The preliminary comparison of the production20 against these256 controls shows that a small mean off-diagonal correlation can coexist with substantially enlarged variances.
At width50 over3500–4800, the mean per-bin **fractional variance ratio**, $(\operatorname{Var}\widehat C_b/\langle\widehat C_b\rangle^2)_{\rm projected}/(\operatorname{Var}\widehat C_b/\langle\widehat C_b\rangle^2)_{\rm homogeneous}$, is3.79±0.31 for SFL TT,3.43±0.24 for SFL EE and4.36±0.37 for SFL BB.
ZEA BB gives2.98±0.23, while its TT and EE give0.95±0.05 and1.07±0.06.
Errors delete whole skies separately in each independent ensemble; the small finite-control-sample bias of a variance ratio is not removed.
These are research diagnostics in `/sptlocal/user/cailmdaley/pol_project/bin_correlations/comparison-production20/`, pending confirmation by the260 newly synthesized projected skies.
The controls' broad-band mean offsets of about0.2–0.4% for TT/EE and1% for ZEA BB are measured rather than assumed away; positive deterministic per-bin rescalings cancel exactly from correlation and fractional variance.
The control spectra's linear interpolation and the small difference between production-alm power and the new CAMB reference explain part of those offsets, as checked in the independent review.

STG BB's adjacent correlation is0.440±0.012 in controls **without any projection**, compared with0.382±0.044 in the production20.
Its projected/control mean ratio is0.883±0.002, so this is not a clean matched-B signal comparison.
An estimator E-to-B residual is a plausible cause, not yet isolated; STG's high-ell BB must be excluded from projection-covariance conclusions.
The projected and control ensembles are unlensed and noise-free, with no source holes or time-domain filtering.
They test whether radial spectra and the same simple mask determine the covariance, not whether they reproduce SPT's full estimator.

## The SPT references

Balkenhol et al. (2023), arXiv:2212.05642, Appendix A.2, describes a 1000-sky CMB-only Monte Carlo measurement, with the HEALPix sky masked before projection.
The paper states that the correlation structure is dominated by projection and replaces its band-diagonal estimates with second-order polynomial fits.
The covariance revisions broaden cosmological parameter uncertainties by up to 15%.
Section IV.A says projection-induced correlations fall to at most 20% only for separations of at least $\Delta\ell=100$; it uses wider bins for the null tests.
Figure8's caption also states that the minimum-variance TT/TE/EE data bandpowers are correlated by **up to40% for neighboring bins**.
This provides a quantitative comparison target within its multipole range, not a universal amplitude for every bin, and does not identify $R$ as its source.
`analysis/compare_bin_covariance.py` explicitly includes2000–3000 alongside the extension to4800.

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

A final quantitative statement about projection's contribution to the SPT excess requires the larger ensemble and a matched control estimator; it is not established by these 20 skies.
