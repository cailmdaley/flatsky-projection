## Approved: no must-fix issues in `cfce23a`

**The 20-seed run can continue as a scoped measurement.** One correction to your summary: the sharp taper wasn't what your quoted numbers measured.

- **`0.0148806`:** this is the **oracle** isolated residual from the `bb48cf4` pilot, whose manifest confirms that code.
- **`0.0239794` and `0.00926`:** these are the wide taper, stored under the old `masked` key. They do confirm the wide forecast.

The first real sharp-taper data are the two finished 20-seed units, seeds 9 and 15, both with `cfce23a` manifests.

### First sharp-taper measurement (two seeds, ranges span seeds 9 and 15)

| ℓ band | Noiseless residual/raw: oracle | Wide | Sharp | Pseudo | Isolated: oracle / sharp |
|---|---|---|---|---|---|
| 200–500 | 0.9–1.0e-4 | 0.018 | 5.5–6.0e-4 | 2.3–2.4e-4 | 0.0046–0.0049 / 0.0050–0.0053 |
| 500–900 | 2.5–2.7e-5 | 0.009–0.011 | 3.2–3.3e-5 | 3.2–3.3e-5 | 0.0086 / 0.0086 |
| 900–1100 | 1.0e-5 | 0.0095–0.0100 | 1.1–1.2e-5 | 1.2e-5 | 0.0160–0.0162 / same |
| 1000–2000 | 6.4–6.5e-6 | 0.0091–0.0092 | 7.2–7.4e-6 | 7.4–7.6e-6 | 0.0402–0.0405 / same |
| 2000–3000 | 2.5–2.6e-6 | 0.0093–0.0099 | 3.0e-6 | 3.2e-6 | 0.340–0.348 / same |
| 3000–4000 | 8.7e-6 | 0.0105–0.0107 | 9.2–9.3e-6 | 9.4–9.5e-6 | 0.883–0.884 / same |

From ℓ≈500 up, the sharp taper is close to the oracle, so the `(1−W)` penalty is gone. Its geometric cost is 3.9e-7, against 0.0095 for wide. Sharp and oracle also agree in the cross means.

### What I checked

| Item | Result |
|---|---|
| Sharp-mask support | On the full real grid, `we>0` and `w>0` match pixel for pixel (1,395,182 each), and `we≥w` everywhere. No unobserved data enter. |
| Coordinates | `xy_to_angle` matches the mask's angles exactly, and the FIELD order and taper form are the same. |
| Taper resolution | Slanted SFL edges narrow it to 0.37°, about 11 pixels, which is still resolved. Pure-B annihilation stays around 1e-9–1e-10, well inside tolerance. |
| Per-split operators | Each estimator uses its own split. `ee` is shared correctly between the cross and isolated arms, and `usign` is handled in every path, including `pseudo`. |
| Noiseless controls | Correctly unfiltered. |
| Noise | Auto minus cross gives 1.0 N for raw and 1.07–1.09 N for sharp near ℓ≈1000. That matches added template noise of t·f²·N with t≈0.08. |
| Regression | The gate passes on the old pilot. Seeds 9 and 15 reproduce the cache to ≤2.5e-13 from ℓ=500 up. |
| Report math | Gain, RMS, the conditioned arm, paired bootstrap and `local_taper_cost` are all correct. |

### Tidy-ups before citing results (not blocking the run)

1. **Commit the report.** The `--expected-seeds` and hash changes are still uncommitted. I reviewed them and they're sound, but the summary should come from committed code.
2. **Fix the report docstring.** Lines 3–5 still say this isn't a conditioning test, which contradicts the new supplementary arm.
3. **Keep the old pilot apart.** In `results/highell/masked_template_pilot`, `masked` means the 2.5° taper, so don't mix that file with the new units.

### Claim boundaries

- **Uniform depth is the main assumption.** The sharp taper gives full E weight to Q/U almost up to the support edge. That's fine for these uniform-noise maps, but real field edges have rising noise and filtering effects. So this shows the attenuation isn't an inherent limit, not that a 0.5° taper is safe on SPT data. This is the sentence most likely to be overstated.
- **Claim nothing below ℓ≈500.** Report bands start at 900. At 200–500 the sharp noiseless residual is about 6× the oracle. A single real-grid noise draw also shows pure-E noise up 1.06× at ℓ 200–500 and 4.3× at ℓ 50–200.
- **Paper line 530 now looks wrong in its reasoning.** Pure-E shows no benefit over pseudo-E from ℓ≈500 up, and pseudo is about 2.4× better at ℓ 200–500. This is only two seeds, so rewrite the sentence after all 20 land.
- **Variance:** two seeds say nothing about it yet. Treat `gain` as known-mean-in-both-arms, and `sample_RMS_gain` as the undebiased case.
- **Bands and intervals:**
  - The conditioned gain and RMS gain have no interval.
  - The conditioned arm is a diagonal amplitude reference, not the full R operator.
  - The 900–1100 and 1000–2000 bands overlap, so their points aren't independent.
- **Cross means at high ℓ:** these are Wiener bias plus scatter (0.14–0.19 at ℓ 2000–3000), so they aren't the noise limit.
- **Everything from the first review still applies:** B=0 sky, white noise, no beam or filtering, no source holes, and two 5 µK-arcmin splits.

Job state at my last check: 2 of 21 steps done, and 18 jobs running with no restarts. I edited no files or fibers. `build/` from the first review is still there.

That background check has finished, and it was the same one whose seed 9 and 15 table I already used, so the approval and findings above stand. No action needed.
