# Independent Opus review of the initial spherical-grid experiment

This is the attributed review of the initial fixed-cutoff campaign, before the C2 correction and expanded input-bandwidth sweep.
The accepted changes and final scope are in `spherical_grid_tests.md`; review experiments that remain scratch-only are not promoted into production claims.
In particular, a single-realization sum need not have a monotonic B power when another input band is added, so "lower bounds" below describes ensemble coupling power, not a deterministic inequality for every sky.

**Verdict: the transforms, parallel transport, CAR centring and inverses are correct, so the point-sample null stands. The pixel-average and aliasing numbers aren't ready for the paper as limits: two test-design choices bias them.** I edited nothing in the repo, filed no fiber and delegated nothing; scratch scripts are in `/tmp/sgt`. Unless noted, the stored results use 3 seeds, and my new experiments use seed 0 only, with field-to-seed scatter of about 20–30%.

BB/EE below means field-selected B² over input E² in the same ℓ band.

## Scientific issues

**1. The CAR alias test can't see aliasing, so the HEALPix vs CAR comparison is invalid.** Input stops at ℓ=512, which folds to exactly 2·384−512 = 256, the edge of the analysis band. Moving that cutoff changes the result completely:

| Input ℓ max | CAR E/B, ℓ 128–192 | CAR E/B, ℓ 192–256 | HEALPix E/B, ℓ 192–256 |
|---|---|---|---|
| 384 | 2e-12 / 1e-12 | 5e-11 / 1e-11 | 1.5e-4 / 3.2e-5 |
| 512 | 9e-8 / 6e-8 | 1.5e-3 / 2.4e-5 | 0.12 / 0.13 |
| 640 | 0.20 / 0.18 | 0.22 / 0.31 | 0.39 / 0.44 |
| 768 | 0.50 / 0.40 | 0.49 / 0.39 | 0.95 / 0.89 |

- **Fix:** report aliasing as a function of input bandwidth, or use a realistic beamed spectrum.
- **Keep:** "CAR is clean up to ℓ=383" is a real sampling result. "CAR aliasing is small" is not.

**2. `--lmax 256` biases HEALPix pixel-average leakage low and creates a false resolution scaling.** HEALPix leakage is broadband, mostly from higher-ℓ E into lower-ℓ B. CAR is close to band-diagonal.
- **Missing input:** at ℓ 192–256 (Nside 128), input from ℓ 257–383 alone gives about 3× the B of all in-band input. Input from ℓ 384–512 gives about 500×.
- **False scaling:** the stored ~16× drop per Nside doubling at fixed ℓ looks like a⁴, but both Nsides share the same ℓ≤256 cut.
- **With `lmax=2Nside`:** Nside 256 gives 6.8e-6 at ℓ 128–256, about 2.5× the stored 2.6–2.9e-6. At fixed ℓ/Nside, HEALPix falls only 2–4× per doubling, and CAR about 4×.
- **Consequence:** the a⁴ picture could underestimate production-scale leakage by orders of magnitude. Extrapolation needs a third Nside, and the "Dimensionless resolution tests" description needs correcting.

**3. A white-input BB/EE isn't a limit for a CMB spectrum.** In HEALPix, B at ℓ 128–192 gets 3.9e-2 from input E at ℓ 192–256, against 1.4e-2 from E in its own band. Any bound should come from a coupling matrix applied to the real EE spectrum, or from a direct realistic run.

**4. The mixing explanation in the fiber needs changing.** It says mixing comes from spatial variation of the window, but CAR contradicts that:
- **φ average:** in coordinates its physical width varies with declination, yet it gives BB ≈ 1e-28, because it is an exact m filter.
- **θ average:** its width is constant, yet it reproduces essentially all coordinate-average BB (8.5e-7 vs 8.3e-7 in the field).
- **Latitude scaling:** BB at dec −30:−56:−75 is 0.155:1:6.44, against cot²θ at 0.152:1:6.33.
- **Ruled out:** centroid offset is not the cause. Removing the sin θ weight leaves BB unchanged, and sampling at centroids gives only about 3e-9.
- **My inference (not derived):** the θ operator couples through the spin connection, which goes as cot θ.

**5. Transport isn't a uniform correction.** It adds B to the CAR φ average, taking it from about 1e-28 to 5e-7 in the field. Overall it lowers field BB from 8.4e-7 to 5.4e-7 but raises dec −30 from 1.2e-7 to 3.0e-7. The paper has to say which operator models a real mapmaker; I didn't check what spt3g does.

**6. Smaller points**
- **Mask comment:** `regions()` says C2, but the taper is C1², so the comment is wrong. Production's map-level C2 gives a C2² power weight, which matches to about 0.3%, and BB changes by under 3% against plain C2.
- **Not the same sampling:** CAR has 1.5× as many pixels, and at dec −56 its cells have about 0.59× the HEALPix area. That inflates part of the ~77× HEALPix/CAR gap (4.16e-5 vs 5.42e-7, ℓ 192–256).
- **Weak gates:** in `check_spherical_grid.py`, the longitude-phase check only asserts finite values. The transport check compares two formulas and never tests the healpy/ducc convention. Tests A and C below would make good gates.

## Checked and correct
- **A. Transport sign and formula:** on a single-ℓ field, the transported ring average is a constant real multiple of the field, with scatter 7e-14, and matches `d^ℓ_22` (−0.393140 for both). The flipped sign scatters by 3e-2. healpy and ducc agree to 5e-12.
- **B. Rotated-frame formula:** the Rodrigues expression matches a hand derivation.
- **C. CAR subpixel centring:** the code trace is correct, and measured phases are at most 2.4e-5 rad, where a half-pixel error would give 0.07–0.92 rad.
- **D. Inverses:** point-sample BB is about 1e-28 for CAR and 1e-21 for HEALPix (Nside 128). HEALPix pixel-average results are unchanged across 2, 8 and 24 iterations, and inverting to ℓ=383 shifts them by less than 1%.
- **E. Subpixel convergence:** BB rises about 8% from factor 4 to 8, close to the 10% expected from (1−1/f²)².
- **F. CAR window prediction:** it agrees to 5.4e-4 in the field, and at worst 2.6e-3 (dec −75, top band).
- **G. Locality:** about 97% of HEALPix field B comes from sources within a roughly 10° buffer around the field.
- **H. ASTRA:** the additions remove no lines from `astra.yaml`, and `lc status` reports both outputs ok.

During the review, results moved from `__units` to the gathered `spherical_grid_test` directory, and HEAD gained commit `7474894`. The gathered 128-8 values match what I analysed.

## Claims that are safe
- Spherical transforms don't leak for band-limited point samples, down to numerical precision.
- For band-limited ℓ≤256 white input, field-selected BB/EE is about 8e-6 to 5e-5 for HEALPix and 5e-8 to 5e-7 for CAR (Nside 128). These are lower bounds for a sky with power above ℓ=256 on HEALPix, and they aren't cut-sky pseudo-Cℓ.
- The tangent-plane CAR window, including its declination dependence, is verified to about 3e-3.
- CAR coordinate averaging in φ alone produces no E/B mixing.

## Claims to avoid
- "No leakage" on HEALPix or CAR for finite pixels.
- "CAR aliasing is negligible" without stating the input bandwidth.
- Any a⁴ scaling or extrapolation to production resolution.
- "Transport removes leakage."
- "Mixing comes from spatial variation of the window."
