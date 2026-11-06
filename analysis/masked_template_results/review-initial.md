## Review of `bb48cf4`: masked E-template on SFL 1500d

**I found no bugs in the signs, U convention, support isolation or noise independence.** The main problem is one of interpretation. The masked result will be dominated by `(1−W)` attenuation, which is an estimator choice rather than an information limit, so fix this before spending the 20-seed run.

`bash env.sh python analysis/check_masked_template.py` passes. Pilot job 2734650 was still running with no output, so every number below is a prediction or a toy result. Predictions use the real mask and tan² geometry plus the cached 20-seed spectra, and I ran extra in-memory probes.

HEAD moved to `f29a781` during the review, but `masked_template.py` and its check are unchanged. I also read the new `masked_template_report.py`.

### Checked and sound

| Item | Evidence |
|---|---|
| Pure-E product rule | Hand derivation matches every term. Flipping any of 6 derivative-term signs fails the gate (residual 5.6–160, against 5e-13). |
| B annihilation on a compact mask | About 1e-12 on a toy 2.5° C² mask, so aliasing from spectral derivatives isn't a concern. |
| E estimator isolation | Exact by construction, and the gate confirms it. |
| NaMaster isolation | Data outside W>0 do reach purification, but only at BB/EE ≲1e-11 in a toy. Negligible. |
| U convention | Maps are COSMO, so `usign=-1`. NaMaster needs that flipped U: BB/EE is ~1e-32 flipped, ~1 unflipped. `pure_e`, `eb` and `qu_of_b` are consistent. |
| Splits | Independent, and each template comes from its own split, so template noise can't bias the cross. `N=Δ²` is correct. |

### Findings, most important first

**1. High: the masked result mostly measures W, not the edge's information cost.**
- Pure-E gives roughly `W·E`, so the leftover leakage is `(1−W)·S_J[E]`.
- The predicted noise-free residual over raw is ⟨W²(1−W)²t⟩/⟨W²t⟩ = **0.0095**, flat in ℓ. Here t = tan²2δψ, and without tan² weighting it would be 0.0056, because shear peaks at the edges.
- That is about 400× the oracle floor of 2.4e-5 at ℓ 500–1000.
- A toy confirms it: measured 0.011 against predicted 0.0107.
- A sharper template taper inside the same support removes it. In the toy, a 0.5° `W_E` gives about 1e-6 above ℓ=1000, 7e-6 at ℓ 500–1000, and 2.6e-3 at ℓ 200–500.
- Real-geometry predictions:

| Template taper | Residual/raw |
|---|---|
| 2° | 3.0e-3 |
| 1° | 4.2e-5 |
| 0.5° | 3.6e-7 |

- Dividing by W with a W>0.05 cut does worse: 1–3e-3, rising to 1.7e-2 at ℓ 3000–4000.
- I didn't test low-ℓ noise amplification for sharp tapers.
- **Action:** add a sharp-`W_E` variant before running `masked_template_realistic`. Also fix the comment at `masked_template.py:70` ("information cost") and the report's `edge_cost_*` key.

**2. High: the noise-limit formula doesn't apply to the cross mean.** With f the Wiener factor, N the noise power and L the raw leakage:

| Statistic | Oracle expectation |
|---|---|
| `bb_isolated` | (1−f)²L + f²tN = t·CN/(C+N) |
| `bb_cross` | (1−f)²L = t·CN²/(C+N)², no template noise in the mean |

- Masked versions use ⟨W²(1−fW)²t⟩ and ⟨W⁴t⟩, with ⟨W⁴t⟩/⟨W²t⟩ = 0.959.
- So comparing the cross with the noise-limit curve would look like beating the limit.
- Wiener filtering also trades bias for variance in the cross. The oracle at ℓ 2000–3000 shows it: R/L is 0.12 with a variance ratio of 0.85, against 0 and 0.89 unfiltered. At ℓ 3000–4000 it's 0.81 and 1.00, against 0 and 1.12.

**3. High: the power-subtraction reference is fair only if both sides are treated alike.** Subtracting a fixed mean does leave the variance unchanged, as you said, but three things qualify the comparison:
- **Debiasing:** if power subtraction gets a known mean, the template residual should get one too, and then you compare variances. If not, compare RMS about zero. At ℓ 500–1000 the masked mean (~1% of L) is similar in size to raw scatter (1.6% of L), so without debiasing the rms gain could fall to about 1.6. The report's `gain` is std-only, so it hides this.
- **Conditioning:** rescaling to each realization's EE is a stronger reference. Cached seeds show it cuts raw scatter by 7–30% (0.0275→0.0246 at ℓ 900–1100, 0.0133→0.0092 at ℓ 1000–2000).
- **Statistics:** one seed says nothing about variance. With 20 seeds, a single band's variance ratio is only known to about ±45% (1σ, my estimate). Leakage scatter is also 2.4× the Gaussian value (0.0159 vs 0.0065), so Gaussian forecasts are order-of-magnitude only.

**4. Medium: paper line 530 isn't supported as written.**
- The code's order is pure-E, then QU subtraction, then NaMaster pure-B. So subtraction comes after E purification but before B purification.
- The stated reason, that E must be free of mask ambiguity, also looks weak. In the toy, pseudo-E gives essentially the same residual (0.011–0.013, against 0.011–0.015 for pure-E) and is closer to `W·E`.
- That toy used a single-harmonic S_J and unpurified residuals. Add a `pseudo` variant on real maps before changing the sentence.

**5. Medium: the outputs can't test the masked prediction yet.**
- EE isn't saved, and `tan2` is only W²-weighted. The report's `pred` covers the oracle only.
- All of this is recomputable from geometry and cached EE.
- Add a cheap regression gate: `bb_signal` and `bb_signal_oracle` for seed 0 should reproduce cached `field_BB_raw[0]/w2` and `field_BB_sub[0]/w2`.

**6. Medium (inferred, not measured): the scalar Wiener weight uses grid ℓ.** Sky ℓ differs most at the sheared edges, and that matters on the steep tail near ℓ~2500. It's also blind to position and direction, so `wf` isn't the optimal cut-sky weight.

**7. Low:**
- `fft.set_workers(4)` without `with` does nothing; workers stayed at 1. It only affects speed.
- "Unbiased" in the docstring means noise-unbiased, not leakage-free.
- `units.projection` is unused, and `TEMPLATE_SIGN` is silently assumed to be +1.
- The gate only covers one k-mode, `res=1`, and E response at W=1. Add the exact identity Ê_k = Σ W_{k−k'}(k'²/k²)E_k'.

### Predictions to compare with the pilot

| ℓ band | L/N | Oracle cross R/L | Masked cross R/L | Oracle isolated (cached) | Masked variance ratio: B=0 | With lensing BB |
|---|---|---|---|---|---|---|
| 500–1000 | 8.7 | 7e-5 | 0.0099 | 0.0094 | 0.008 | 0.016 |
| 1000–1500 | 3.0 | 5e-4 | 0.011 | 0.026 | 0.048 | 0.060 |
| 1500–2000 | 0.83 | 0.006 | 0.018 | 0.090 | 0.29 | 0.30 |
| 2000–3000 | 0.13 | 0.12 | 0.14 | 0.34 | 0.85 | 0.85 |
| 3000–4000 | 0.007 | 0.81 | 0.82 | 0.88 | 1.00 | 1.00 |

Variance ratios are Gaussian forecasts. At ℓ 900–1100, the cached isolated residual is 0.0156, a factor of about 64. My rough hand estimate for masked is about 0.025, a factor of about 40.

### What each statistic can claim

| Statistic | Can claim | Cannot claim |
|---|---|---|
| `bb_signal_masked` / `bb_signal` | Noise-free cost of this W-attenuated estimator, for that realization or seed mean | Information cost of the edge |
| `bb_isolated_*` | Template operator residual, comparable to the noise limit | Observed BB, or anything about variance |
| `bb_cross_*` mean | Noise-unbiased leakage residual, (1−f)²L-type | The noise-limit value, or a B measurement free of signal residual |
| `bb_cross_*` scatter across seeds | Sample variance including signal and noise, 20 seeds only | Anything from the one-seed pilot |
| `bb_auto_*` | Noise-biased diagnostic | Debiased BB |
| `gain` | Template versus **fixed, perfectly known** mean | Superiority over EE-conditioned or imperfect-mean subtraction, or a bias-inclusive gain |

**All of these results apply only to this setup:**
- Unlensed B=0 sky, with SFL polar reflattening and an exactly known Jacobian.
- A 2.5° C² rectangle with no source holes.
- White isotropic uniform noise, with no beam or timestream filtering.
- Two 5 µK-arcmin splits, equivalent to about a 3.5 µK-arcmin coadd.

They don't show the template works on real SPT data. Lensing BB would need to enter both arms.

### One side effect

Running `env.sh` as you asked appears to have extracted `spt3g_build.tar.gz` into an untracked 94 MB `build/` in the worktree. Its birth time is 22:06:29 and it wasn't in my first status check, though another worker in this worktree could also have triggered it. I left it there because you said not to edit anything. It's harmless locally, but don't stage it. No other files or fibers were changed.
