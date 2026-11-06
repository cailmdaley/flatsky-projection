**Verdict: the numbers and scoped claims are approved, but fix two things before they reach the paper.** All 20 NPZ hashes match `summary.json`, and an independent recomputation matches every headline value.

## Must-fix before paper use
1. **The macros aren't exported.** `analysis/paper_numbers.py` never references `masked_template`, and there's no `maskedTemplate*` in `paper/`, including `numbers.tex`. `numbers()` runs and its output looks clean, but I didn't compile any LaTeX.
2. **`maskedTemplateVarianceGain` is misnamed.** It holds a standard-deviation ratio; the variance gain is about 399. Rename it to `...StdGain`, likewise `...HighVarianceGain` and the interval macro.

## Headline checks
| Claim | Result |
|---|---|
| Isolated residual, 900–1100 | Masked 1.5579%, oracle 1.5578%, SE about 0.0054 percentage points; quote as 1.558 ± 0.005% |
| Local prediction | 1.5507%, oracle/prediction = 1.0045, about 1.3σ |
| Cross BB std gain | 19.970 reproduced; bootstrap reruns gave about [11.5, 36.0] |
| Conditioned / RMS gain | 17.803 / 20.426 reproduced |
| 2000–3000 | Std gain 1.0362, RMS gain 0.1151 reproduced |

- **"Raw" in your headline:** I read 1.5579% as the masked estimator, expressed as a fraction of raw BB. The denominator is the noiseless raw BB.
- **Interval:** it's solid but not sharp. A Gaussian F interval gives [12.6, 31.7], leave-one-out runs 16.2–25.4, and the jackknife SE is 6.8. There's no single-seed outlier, and the band is only 4 bins (ℓ = 925–1075).
- **Rerun drift:** the lower bound moved by about 0.2 across bootstrap reruns, so print `[11, 36]` and say "≈20" in prose. I checked the interval statistically; I didn't replay the 4000-draw RNG stream.
- **Wiener bias:** confirmed in every band. The measured cross mean matches (1−f)²L at z = −0.01, −0.23, +0.86 and −0.49. At 2000–3000 it's 14.83 ± 0.37% against 14.51% predicted, about 40 SE from zero.

## Interpretation limits for the prose
- **The ≈20 gain doesn't measure template quality.** The wide-taper control gets 21.5, with wide/masked SD = 0.93 [0.85, 1.07]; pseudo-E gets 19.96. Masked and wide differ in the mean: noiseless residual 1.1×10⁻⁵ vs 0.92% of raw, RMS gain 20.4 vs 2.8.
- **The gain probably depends strongly on noise level.** A rough mode count suggests the post-subtraction scatter is near the noise×noise floor, within about 30%. That is my inference, not something these runs measure, so don't carry the number to other depths.
- **Don't rank 20.4 above 20.0.** The cross mean is 0.34 SE from zero, so that gap is a sample fluctuation. It isn't a `ddof` bug, because the mean of x² about zero is unbiased.
- **17.8 has no interval.** It's not sensitive to the ratio choice, since best-linear β gives 17.76. It also isn't a bound on full-R conditioning.
- **2000–3000 needs its interval.** The gain is 1.04 [0.74, 1.36], consistent with no gain, and scatter is noise-dominated (corr(cross_raw, signal) = 0.25). `maskedTemplateHighVarianceGain` has no interval macro, so add one.
- **Macro gaps:** both residual macros print `1.56\%`, and there's no local-prediction macro, so the three-way comparison can't be sourced from macros. `HighRMSGain` prints 0.12, not 0.115. `EstimatorCost` is 1.0001 (z = 4.5), but the same ratio is below 1 in the higher bands (0.99982 at 2000–3000, z = −12). Call it agreement at the 10⁻⁴ level.

## Caveats and figure
- **Caveats:** all are retained in the note: uniform 5 µK-arcmin splits, B_sky = 0, no source holes, scalar Wiener weight, no full R-conditioning and no true-B response calibration. Source holes and true-B response are absent from the `summary.json` scope string, so paper prose must carry them.
- **Note algebra:** the auto-noise and B-response derivations check out. "L = tC" is only approximate (t = 0.0817, BB/EE ≈ 0.07), but the prediction uses measured L, so the headline is unaffected.
- **Figure caption:** the left panel's noisy curves are isolated autos, not observed spectra. On the right, joined points imply a continuous curve, but the 900–1100 and 1000–2000 bands share two bins. The oracle curve is also hidden behind masked.

The earlier reviews are at `analysis/masked_template_results/review-*.md`, not `results/review*.md`, and I didn't read them. I wrote two scratch scripts to `/tmp` (`mt_review.py`, `mt_review2.py`). No repo or felt files changed, and I didn't check the manifest `git_sha` against the code.
