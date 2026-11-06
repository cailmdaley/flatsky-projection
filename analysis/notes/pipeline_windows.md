# Simulation windows in sky and map coordinates

The simulation synthesizes point samples on HEALPix, interpolates them onto an oversampled map, and averages eight samples per output-pixel axis.
There is no HEALPix pixel-average window in those point samples.
The two windows introduced by the subsequent operations act in different coordinate frames.
`analysis/pipeline_windows.py` implements them, and `spectra.load`, `load_field`, and `load_scalar` remove their effective response for the known simulated input spectrum.
`correct_windows=False` returns the unchanged production bandpowers.

## Ring interpolation

Write the local sky wavevector as \(q=(q_e,q_n)=J^T k\), in the orthonormal east/north frame.
If the local ring separations are \(h_e,h_n\), the coherent power response of bilinear interpolation is

\[
T_{\rm int}(q,\delta)=\operatorname{sinc}^4(q_e h_e/2)\operatorname{sinc}^4(q_n h_n/2),
\]

where \(\operatorname{sinc}x=\sin x/x\).
This is the square of the phase-averaged amplitude, not the mean squared interpolated value at one point.
The latter includes high-frequency interpolation images and is not the coherent low-frequency transfer.

For HEALPix Nside \(N\), let \(z=|\sin\delta|\).
In the equatorial zone, \(h_e=\cos\delta\,\pi/(2N)\) and \(h_n=2/(3N\cos\delta)\).
In the polar cap, let \(r=N\sqrt{3(1-z)}\); then \(h_e=\cos\delta\,\pi/(2r)\) and \(h_n=2r/(3N^2\cos\delta)\).
These are local asymptotic spacings, not a claim that curved finite-Nside rings are exactly rectangular.
The actual spt3g implementation interpolates between rings in \(z\), rather than in colatitude.
The controlled tests below include that implementation and bound the local approximation on the tested footprints and frequencies.

The window depends on declination and wave direction.
It must therefore be evaluated with the full Jacobian and its orientation, before averaging the footprint and Fourier azimuth.
The histogram of singular values alone discards the orientation relative to the HEALPix rings.
The previous equal-area-square approximation \(\exp[-q^2(4\pi/N_{\rm pix})/6]\) cannot capture that information.

## Finite grid average

For output-pixel width \(d\), the exact coherent amplitude of an eight-sample average on one axis is

\[
A_8(k)=\frac{\sin(kd/2)}{8\sin(kd/16)}.
\]

The two-dimensional power is \(T_8(k_x,k_y)=A_8(k_x)^2A_8(k_y)^2\).
This is not the continuous square top-hat \(T_\infty\), which production already divides out after bandpower estimation.
The per-band correction is

\[
F_{b,p}^{XY}=\frac{\langle R_p^{XY} [C_{\rm input}T_{\rm int}]\,T_8\rangle_b}
{\langle R_p^{XY} C_{\rm input}\rangle_b\,\overline T_\infty(\ell_{\rm eff,b})}.
\]

The numerator uses sky wavevectors for the ring response and map wavevectors for the grid response.
The denominator reproduces the production division at the band centre, rather than dividing the integrand by a varying top-hat within each band.
The correction removes only introduced simulation windows; it leaves the geometric spectrum distortion and E-to-B mixing in place.
It is calibrated to the specified input spectrum analytically, with no measured amplitudes or fitted parameters.
It is not a transferable diagonal response for an arbitrary unknown sky.

STG has no nonzero geometric BB spectrum against which to define this ratio.
Its BB loader uses its EE window only as a presentation convention; neither its estimator floor nor aliased B is claimed to be calibrated by that operation.
The EB leg uses a scalar response with the simulated E spectrum.

## Aliasing

The reciprocal lattice spacing of a 2 arcmin grid is \(2\pi/d=10800\), and the axis Nyquist is 5400.
A sampled output mode \(k\) also receives modes \(K=k+(10800n_x,10800n_y)\).
Each image reads the input spectrum at \(|J^T K|\), retains the grid average evaluated at \(K\), and contributes a polarization angle measured relative to the *observed* direction of \(k\).
Folding therefore creates B even for a conformal projection.

The familiar onset \(10800-L_{\rm sky}\) applies to a locally unit Jacobian.
Elsewhere the spherical cutoff becomes an ellipse in the map Fourier plane, so neither 4800 for a cutoff of 6000 nor 5800 for a cutoff of 5000 is a universal onset across the footprint.
`report_grid_aliases.py` sums the nearest reciprocal images for the field and compares both sky cutoffs.
It is a local-affine prediction, not an exact prediction of every purified-estimator bandpower or the HEALPix-interpolation sidebands.
`window_bandlimit_test` in `astra.yaml` supplies the independent paired spherical intervention.

An isotropic pre-deconvolution of the alms would invert neither the direction-dependent ring response nor the map-frame average exactly.
It would also increase the high-frequency power available to alias.
It offers no precision gain over this known-input effective correction; direct evaluation of the harmonic sky on output subpixels would remove the interpolation stage itself, at greater cost.

## Reproduction and statistical interpretation

Run scientific scripts with `~/code/python/spt3g_software/build/env-shell.sh python <script>` and bounded OpenMP/BLAS threads.

- `analysis/check_pipeline_windows.py` synthesizes monochromatic sky waves at HEALPix centres and measures the coherent response of the actual spt3g interpolator.
- `analysis/check_projected_windows.py` passes known plane-wave spectra through actual `maps.reproj_map` in every paper projection, off-centre on the field, varying interpolation and rebin independently.
- `analysis/check_window_quadrature.py` compares footprint/azimuth quadratures; convergence is assessed on the correction ratio rather than a coarse absolute forward spectrum.
- `analysis/cache_simulation_windows.py` warms the known-input correction cache; cache keys include the implementation, theory file and production mask metadata.
- `analysis/check_window_loaders.py` checks shapes, broadcasting, raw-data roundtrips and invalidation of the derived field cache.
- `analysis/report_pipeline_windows.py` uses the production-mask geometric R model evaluated at every integer multipole, reports corrected residuals and paired-seed projection differences, and writes `/sptlocal/user/cailmdaley/pol_project/pipeline_windows_production.npz`.
- `analysis/plot_pipeline_windows.py` renders those absolute and paired residuals.
- `analysis/paper_numbers.py --window-only` writes `paper/window_numbers.tex` from those outputs and the controlled tests.
- `lc run --executor htcondor --profile htcondor9 window_bandlimit_test -u highell` runs the additive paired spherical test; `analysis/report_bandlimit_test.py` verifies its cutoff-6000 leg against production and compares the result with the alias calculation.

Pointwise standard errors describe seed scatter at a specified multipole.
They are not independent across bins or projections.
For a projection comparison, form the per-seed difference first and estimate the error of its mean.
Twenty seeds cannot support an invertible covariance over 120 bins, so a sum of squared diagonal residuals is not a calibrated goodness-of-fit statistic.
The correction does not make aliasing, the STG floor, finite-estimator response or numerical forward-model errors disappear.
