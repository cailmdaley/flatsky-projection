# Independent Opus follow-up on the expanded cutoff sweep

This review read an intermediate note while final numerical and wording corrections were in progress.
Its required distinction between aliasing and finite-pixel coupling, all-cutoff table, pairing clarification, and 4x4 quadrature scope are incorporated in `spherical_grid_tests.md`.
The tautological chart/sky algebra comparison was removed; the actual pixell API and spherical roundtrip gates remain.
The citation was checked against the paper itself: equation (B.5), not a subsection named B.5.

**The scoped conclusion is safe once one thing changes: the note's "aliasing" and suppression language. Finite averaging doesn't always suppress the out-of-band signal, and the difference statistic isn't purely aliasing.** Otherwise I found no remaining code errors. All 12 sweep units are already gathered, so I read those numbers directly.

## Must fix
The note says "Finite averaging suppresses this particular alias injection", and that's false at the lowest cutoff. B/in-band-E at ℓ 192–256 in the 1500d footprint, mean of 3 skies:

| Input ℓ max | HEALPix point | HEALPix 4×4 | CAR point | CAR 4×4 |
|---|---:|---:|---:|---:|
| 384 | 2.2e-5 | 1.1e-4 | 1.9e-11 | 5.3e-8 |
| 512 | 0.12 | 0.020 | 3.6e-5 | 6.5e-6 |
| 640 | 0.36 | 0.047 | 0.33 | 0.033 |
| 768 | 0.88 | 0.066 | 0.41 | 0.037 |

- **What 384 shows:** averaging raises B on both grids. CAR point sampling is clean there (2e-11), so CAR's 5e-8 can't be sampling aliasing. It has to be the pixel operator coupling ℓ 257–384 into the band.
- **What to change:** the difference measures the total out-of-band contribution, meaning sampling aliasing plus finite-pixel coupling. Fix that wording in the note, in the `astra.yaml` description and in the stage docstring ("isolate(s)… aliasing").
- **Replacement claim:** averaging raises the contribution at cutoff 384 and lowers it from 512 upward. At 640 and 768 the two grids are within a factor of two, so the large gap at 512 and below reflects the CAR cutoff and sampling.
- **Table:** replace the note's single-cutoff alias table with all four cutoffs.

## Minor
- **Scope sentence:** "This isolates intrinsic grid/pixel leakage" should say "from input ℓ≤256". The sweep shows that this isn't the whole finite-pixel response.
- **Comparing cutoffs:** `hp.synalm` probably assigns its random draws differently at each `lmax`. If so, the ℓ≤256 sky isn't shared between cutoffs, which I inferred rather than tested. Pairing is exact within a cutoff, but compare across cutoffs only as ensemble means.
- **Quadrature label:** keep "4×4" on alias averages at ℓ≈768, where they aren't continuum pixel averages. Main-stage factor-8 convergence doesn't carry over.
- **pixell gate:** the `chart`/`sky` comparison in `check_spherical_grid.py` compares identical algebra, so it can't fail. Note line 99 shouldn't lean on it, although the variable change itself is correct.
- **Citation:** the note cites Hall & Tessore equation (B.5), while the HEALPix fiber says Appendix B.5. I didn't open the paper, so check the label.

## Checked
- **Gates:** `check_spherical_grid.py` passes. The new ring, convention, CAR-centring, m-filter and C2 gates all test what they claim, and the centring gate would catch a half-pixel error.
- **C2 mask:** it now uses production `make_masks.taper`, squared for power, and the gate confirms the taper.
- **Gathered alias output:** it holds exactly the 12 new units, with no stale `0`–`2` files that could double-count at 512.
- **CAR synthesis above grid bandwidth:** this was the one extra check I ran, because I relied on it last time without validating it. `synthesis_2d` matches direct `synthesis_general` to 8e-10 at ℓ=512 and 6e-10 at ℓ=768, against a map rms of about 100–150, so the CAR sweep can be trusted.
- **Parallelogram coefficient:** I rederived `A/24 · (1+g²)/(1−g²) = Δ²(1+cos²δ)/48`, and it holds.

Nothing was edited, filed or delegated. The only additions are scratch scripts in `/tmp/sgt`.

## Safe after your final sweep
With the wording above fixed, these all stand:
- The point-sample null for band-limited input.
- Small, operator-dependent finite-pixel B, measured for fixed ℓ≤256 white input.
- Transport not being a universal correction.
- The CAR window to about 3e-3.
- A cutoff-dependent out-of-band stress test at Nside 128 and 384×768 CAR only.

None of these supports a CMB-specific or production-resolution bound, or a cut-sky leakage bound, and the note already says so.
