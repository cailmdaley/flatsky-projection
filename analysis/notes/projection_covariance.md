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

On an unbounded plane with a spatially constant nonsingular Jacobian $J$, a homogeneous Gaussian sky remains homogeneous after remapping.
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
The SFL BB trend is visible, but these decoupled TT/EE products do not reproduce the large positive ZEA correlations described by SPT.
Neither an all-off-diagonal average nor this preliminary table settles agreement with SPT's different estimator and mask.

## The SPT references

Balkenhol et al. (2023), arXiv:2212.05642, Appendix A.2, describes a 1000-sky CMB-only Monte Carlo measurement, with the HEALPix sky masked before projection.
The paper states that the correlation structure is dominated by projection and replaces its band-diagonal estimates with second-order polynomial fits.
The covariance revisions broaden cosmological parameter uncertainties by up to 15%.
Section IV.A says projection-induced correlations fall to at most 20% only for separations of at least $\Delta\ell=100$; it uses wider bins for the null tests.
That statement does not provide a single high-$\ell$ correlation amplitude to match at every bin or identify $R$ as its source.

Camphuis et al., arXiv:2506.20707, Section III.B, explicitly says excess high-$\ell$ bin correlations that were difficult to compute analytically motivated the move to full spherical harmonic transforms.
It does not provide a quantitative excess-correlation curve in that discussion.
The downloaded PDFs and searchable text are under `/sptlocal/user/cailmdaley/pol_project/bin_correlations/literature/`.

## Required paper qualification

Replace the sentence expecting the wide rows of $R$ to contribute to correlations with the distinction between a mean-spectrum response and a mode-amplitude response.
The quoted ordinary-sky conclusion from a disk-averaged off-diagonal statistic should not be read as an isotropic Gaussian covariance prescription for the 1500d field.
A defensible replacement is:

> Linear reprojection preserves the Gaussianity of the unlensed input, but its covariance on the plane need not be statistically homogeneous.
> The mean-spectrum response $R$ does not determine the bandpower covariance: that calculation requires the full anisotropic two-point function, with the mask and mode purification included before the Gaussian contractions.

A final quantitative statement about projection's contribution to the SPT excess requires the larger ensemble and a matched control estimator; it is not established by these 20 skies.
