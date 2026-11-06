**Verdict:** I found no errors in the Fourier normalization, noise variance, coordinate mapping, or amplitude prediction. There is one scientific blocker: the paper macros report amplitude loss, but the larger effect is S/N loss. No repo files were changed.

## Blocker: S/N loss is missing from the macros
A catalogue selects on S/N, and `local_source_numbers()` only reports `1-amplitude`. At the worst SFL location I checked (RA 45, Dec −68, g = 0.342), with the CMB+white filter:

| Profile | Amplitude | σ (g=0 → here, µK) | S/N relative to g=0 |
|---|---|---|---|
| Cluster | 0.952 | 17.02 → 19.82 | **0.817** |
| Point source | 0.930 | 21.83 → 22.72 | **0.894** |

- **Likely cause (inferred):** the sheared CMB moves low-ℓ power into the band of an isotropic filter. I did not isolate this directly.
- **ZEA:** S/N stays at or above 0.9975 at the corners I printed, so the effect is SFL/BICEP-specific there.
- **Evidence is thin:** the 16 seeds only weakly confirm this σ growth. Blank z-score std at g > 0.2 is 1.014 ± 0.073 for clusters, which is about 1.6σ from the no-inflation case by my rough estimate.
- **Open question:** if real pipelines use an isotropic CMB term, as I recall SPT's does (unverified), this loss applies to them. I also didn't check which projections the catalogues use.

## What passed

| Check | Result |
|---|---|
| Amplitude vs independent Fourier prediction `μ p̃(Jᵀk)` | Agrees to < 4e-6 for all 180 rows |
| Point source in white noise vs `2μ/√((1+s1²)(1+s2²))` | 6.6e-8; reduces to √(1−g²) for equal area |
| White-noise σ | 19.927 ± 0.070 vs analytic 19.928 µK (40,000 draws) |
| `cutout` chart vs J⁻¹ at small radius | Within about 7.5e-4 inside 5′, no sign or axis errors |
| Noise scaling | `P_map = μ P_sky(Jᵀk)` and pixel σ ∝ √μ are consistent across both files |
| BFGS peak finder | Never below the pixel-grid maximum, no jumps in 80 draws at the sim amplitudes |
| 16-seed blank z-score, all rows | Std 0.93–1.02 (± 0.046) by profile and projection |

## Smaller issues
- **Paired response is not a measurement:** the estimator is linear, so `fixed − blank = A·fit(source)` holds by algebra. The report's assert only checks bookkeeping, and 16 seeds add nothing to mean amplitude bias. They only test scatter and peak bias.
- **`local_measurements.py:122` comment is wrong:** the box's fundamental is ℓ = 422, so `ell<300` removes only the DC mode, not large-scale backgrounds.
- **Box is not periodic for real CMB:** cuts from a larger flat sky scatter 2.5–3% above `sigma_uK`. `check_local_source_window.py` says it doesn't cover background covariance, so this is ungated. A taper removed the excess, though it also changes the estimator.
- **`gaussian_prediction` is stored on every row:** it is only valid for point/white and is off by up to 17% elsewhere. The report filters correctly, but the JSON invites misuse.
- **`noisy_test` reuses `rng(4431)`:** the six Monte Carlo rows are not independent, which is why their means all run low together. With 64 draws, σ is only known to about 9%.
- **Peak-search bias:** it was +4.0 µK on the 100 µK point source in seed 0, so it isn't negligible next to the geometry effect. That's one seed; aggregate it and compare across projections.

## Scope
- **Supported:** noiseless template-mismatch response at 15 fixed points, plus a local-affine σ checked in pooled scatter.
- **Not supported:** finite-pixel or HEALPix source behaviour, field-averaged bias, or a catalogue-level size. There is also no filter-scale maximization, multi-band filtering, TOD filtering, or foregrounds.
- **Still open from the constitution:** the SPT-like catalogue size and the stacked-cutout item.
- **Noise level:** 18 µK-arcmin looks like SPT-SZ depth to me, so state which survey "SPT-like" means.

## Minimal tests to add
1. **S/N macro and gate:** add `(amp/σ)/(1/σ₀)`, and check point/white against `2√μ/√((1+s1²)(1+s2²))`. That formula already reproduces 0.9617 and 0.9844 for BICEP.
2. **CMB-only blank:** record `fit(cmb, template, ...)` on the common sky in `local_source_sims.py`, so white noise stops diluting the σ-growth test.
3. **Fixability invariance:** use the metric template with `power=actual`. This must give S/N ratio exactly 1 at every location, and it shows whether the loss can be fixed by the filter.
4. **Fourier prediction gate:** add the amplitude check above to `check_local_measurements.py` at 1e-5.
5. **Background box gate:** compare σ from a large-sky cut with `sigma_uK`, or apply a taper and state the convention.

## Review limits
- **Code version:** I read the 158-line version; the committed file has 162 lines. It adds `magnification_template_amplitude` and `sky_position_arcmin`, and I saw only the row lines for those. The numbers above that mention 16 seeds, 180 rows or σ come from the canonical `results/highell` outputs.
- **Stopped test:** another session stopped my oversized grid scan, so I replaced it with the 80-draw FFT check.
- **Handoff not delivered:** my message to the local-measurements session failed because it isn't reachable from here, so this report is the only copy. The S/N finding still needs filing in its fiber, which I left alone to stay read-only.
