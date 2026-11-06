**Approved: there are no mathematical or code blockers, and the covariance treatment is adequate for the quoted approximate errors.** SFL 0.899±0.011 (points) and 0.800±0.040 (clusters) stand as statistical errors. I edited and filed nothing. I only ran small read-only scripts on the existing JSONs, with no Monte Carlo.

## Cross-check (Nside 8192)
I rebuilt the estimator and jackknife independently from `sources_*.json` and `resolution_8192_*.json`. S and the jackknife error match the summary to all four printed decimals in all six rows. I then compared them with a closed-form Gaussian delta-method covariance (Isserlis, using the measured seed correlations and explicit 16/64 overlap):

| | S | Stratified jackknife | Paired part | Coarse-only part | Unstratified jackknife | Gaussian, with overlap | Gaussian, gain and coarse independent | Forecast |
|---|---|---|---|---|---|---|---|---|
| SFL point | 0.8987 | 0.0105 | 0.0058 | 0.0088 | 0.0107 | 0.0093 | 0.0094 | 0.8946 |
| SFL cluster | 0.7999 | 0.0397 | 0.0171 | 0.0358 | 0.0402 | 0.0337 | 0.0335 | 0.8205 |

- **Agreement:** across all six projection/profile rows, jackknife/Gaussian is 0.85–1.18. That is the scatter expected from fourth moments estimated on 16 and 48 skies. Both SFL quoted errors sit on the conservative side, and both estimates are within about 0.5σ of their forecasts.
- **Overlap barely matters:** you were right not to assume independence, but assuming it would have given almost the same answer. To first order the coarse-variance covariance enters as cov(a,f) − cov(a,p), with a the 64-sky coarse variance, p the paired coarse variance and f the paired fine variance. That is proportional to ρ²−1, and ρ is 0.977–0.999 at these two locations.
- **Error budget:** the coarse-only stratum dominates, not the gain.

## Estimator and code
- **Stratification:** changing weights in the pooled variance when a sky is deleted is harmless. It only shifts a stratum's leave-one-out mean, and within-stratum centering removes that shift. The strata treat 16/48 as fixed, which matches the design.
- **Ratio bias:** the O(1/n) bias in V is 2(1−ρ²)(1/16−1/64), which is about 1e-4 on S.
- **Indexing:** `paired` holds seed values used as indices. That is safe because `base` is sorted and `report()` asserts seeds 0–63 first. The `b`, `f`, `pred`, `white` and `amps` arrays all share the `gr` ordering, and the mask keeps `b[paired]` aligned with `f`.
- **Completeness gates:** these work. Line 19 and the assertions at lines 62, 104 and 109 fail on missing data, and a missing location raises `KeyError`. The paired white and source differences are 1e-13 and 5e-13 µK.
- **Data:** I saw 90 unique rows per seed with no NaNs in the 64 coarse and 16 Nside 8192 files. I didn't scan the 4096 files, but their summary values are finite.

## Non-blocking notes
- **Precision:** "0.040" implies more precision than the error has. Given the jackknife/Gaussian spread, ±0.04 is the honest statement, and ±0.011 is about 0.01.
- **What the ± covers:** it is statistical only, for this centre/corner pair. Residual pixelization isn't included, but the estimate is stable from 4096 to 8192 for SFL (0.8993→0.8987, 0.8007→0.7999). The largest shift among all six rows is BICEP cluster, +0.0044, within its error.
- **Dead guards:** `if sampling else {}` at line 89 and `if sampling and` at line 90 can never take their fallback after the line 19 assertion. They are cosmetic, but they read like the silent pass the last review flagged.

The note's quoted values (variance 0.94–0.96 ± 0.04–0.05, S/N 0.899±0.011, 0.800±0.040) match the summary. I didn't review the rest of the note or the 4096 errors beyond the printed table, and I didn't run any test suite.
