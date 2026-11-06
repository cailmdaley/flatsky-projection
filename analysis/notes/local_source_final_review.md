**Verdict: the science is approved, with one blocker on the record. The summary and note were not regenerated after the resolution gather, so they don't carry the check that validates the S/N forecast.** The mathematics holds. I edited nothing in the repo. I ran one throwaway read-only script on existing JSONs, with no simulation, and deleted it afterwards.

## Blocker
`local_source_summary.json` has `sampling_resolution: {}`, and the note doesn't mention the HEALPix finding. `results/highell/local_source_resolution` has now finished gathering (34 entries, all 32 units).
- **Regenerate:** rerun the report and note against the gathered output.
- **Record the 2048 bias:** at Nside 2048, the measured CMB variance is only 0.65–0.73 of the prediction, so the 64-seed variance check fails unless it is corrected by the paired gain.
- **Align the numbers:** the note quotes the periodic values 0.817 and 19.82 µK. The macro uses the finite-patch 0.8205, with σ of 19.73 µK.
- **Fix the silent pass:** `report()` writes `{}` when the sampling files are absent, which is how this slipped through. It should fail instead.

## What passed
| Check | Result |
|---|---|
| Kernel normalization | `weights` = `n²·ifft2(tf/P)/norm` matches the sims' `blank` estimator algebraically: it gives unit response to its own template. |
| Pad = 1 limit | Reduces analytically to the periodic formula; numerically 22.72 and 19.82 µK, matching the note. |
| White variance | Σw²·µ(18′)²/pix² matches the pixel noise the sims inject; by Parseval it's the same as the periodic white variance. |
| Map-plane covariance | `µP(Jᵀk)` is correct, and the axes match `fit`. |
| Zero-padding | Pads 2, 4 and 8 agree to 4 decimal places at the SFL corner. |
| Whole-seed jackknife | Deletes entire seeds across all locations, so correlations between locations are handled. |
| Pairing | Same alms by construction (`lmax=6000` at every Nside). White blanks agree to 1.3e-13, and CMB blanks correlate at 0.977–0.9999. |

## Fine/coarse results (16 paired seeds, relative to 2048)
| Chart, profile | Gain 4096 | Gain 8192 | 64-seed ratio × gain 8192 |
|---|---|---|---|
| SFL point | 1.301 ± 0.021 | 1.389 ± 0.025 | 0.936 ± 0.045 |
| SFL cluster | 1.232 ± 0.010 | 1.300 ± 0.013 | 0.952 ± 0.042 |
| ZEA point | 1.332 ± 0.021 | 1.435 ± 0.024 | 0.939 ± 0.050 |
| ZEA cluster | 1.270 ± 0.010 | 1.349 ± 0.014 | 0.944 ± 0.050 |
| BICEP point | 1.249 ± 0.018 | 1.334 ± 0.020 | 0.933 ± 0.050 |
| BICEP cluster | 1.234 ± 0.008 | 1.303 ± 0.011 | 0.942 ± 0.050 |

I propagated the last column's errors by hand, treating the 64-seed ratio and the gain as independent.

- **Variance model:** the 16-seed ratios at 8192 are 0.98–1.01 ± 0.10, so the forecast is supported.
- **The ~0.94 is one result, not six:** all six rows are about 1.3σ low, but they share CMB seeds. If the bias scales as pixel area squared, which I infer rather than measured, 8192 still loses about 2% and the true ratio is about 0.956.
- **Corrected corner S/N:** applying the per-location gains to the 64-seed variances gives about 0.90 for SFL points (forecast 0.895). SFL clusters give about 0.80 ± 0.035 (forecast 0.820), an error I estimated by hand. I'd keep this as the empirical comparison.

## Findings the note should state
- **2048 can't validate corner/centre S/N.** Its suppression depends on location: SFL point variance ratio (2048/8192) is 0.52 at the centre and 0.83 at the corner. The earlier agreement (0.898 vs 0.895) was partly luck, because the centre's CMB contribution is small. Other 2048 scatter numbers, such as astrometric scatter and peak bias, are affected a little too, and the note should say so.
- **Raw 16-seed S/N is not evidence of a larger loss.** SFL cluster gives 0.736 ± 0.059, and ZEA, which should be about 1, gives 0.959 ± 0.053. All three share one low centre draw (empirical 6.07 µK vs predicted 8.29 µK). The report should present the 64-seed × gain estimator, not this one.
- **The earlier 2.5–3% excess isn't supported.** The prior review said large-sky cuts scatter 2.5–3% above `sigma_uK`. The finite-patch correction is about −0.4% in the SFL cluster σ at the corner, and the opposite sign. It's exact for a stationary Gaussian sky, so either that excess came from something else or it was noise. I haven't seen that test.
- **"Metric plus covariance restores S/N" is still only a periodic forecast.** No finite-patch version was computed. The finite-patch shift elsewhere is about 0.4%, so this is a wording fix.

## Scope
- **Amplitude vs S/N vs catalogue:** the note keeps these separate well, and the report's `scope` string excludes catalogue calibration.
- **`local_source_numbers()` docstring:** it says only "exact spherical response", but the `SnrLoss` macros are forecasts. It should say so.
- **Paper:** `main.tex` quotes none of these macros yet, so `paper/local_source_numbers.tex` is an empty header. Nothing in the paper depends on them, and that file doesn't need committing.

## Approved claims
- SFL amplitude losses of 6.98% (point) and 4.85% (cluster), at the 15 sampled points.
- Finite-patch S/N forecast losses of 10.5% and 18.0%, labelled as forecasts. The empirical check is consistent, but only to about 3 percentage points.
- Unit response with a metric template.
- ZEA stays within 0.3%, and BICEP's amplitude gains aren't information gains.

None of this is approved as a catalogue bias, completeness or count statement.
