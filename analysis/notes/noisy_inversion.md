# Inverting the projection response on noisy bandpowers

`noisy_inversion_sims` (astra.yaml, `src/stages/noisy_inversion_sims.py`) draws 260 lensed Gaussian skies (TT, EE, BB, TE), applies a 1.2′ Gaussian beam on the sphere, projects them with the production pipeline onto the 1500d field for ZEA, STG and SFL, and adds two independent splits of white grid noise: 5 μK-arcmin coadded in polarization, 5/√2 in temperature.
Per-pixel noise variance scales with the pixel's area magnification (plane area over sky area), a fixed instrument depth per unit sky area: uniform on the equal-area ZEA and SFL, rising as sec⁴(θ/2) on STG (checked against `geometry.jacobian` to 1e-7).
The stage keeps split-cross and noise-free coupled pseudo-spectra of the same skies.

`analysis/noisy_inversion.py` decouples them at Δℓ = 50, divides the flat top-hat and simulation windows, and fits one amplitude per 500-wide sky band over 500–4000 through R (TT scalar block; joint EE/BB with E bands × beamed lensed EE and B bands × beamed lensed BB), exterior held at its model.
Bins are weighted by their measured scatter; the variance cost is quoted for the inverse-variance combination of the bands, because with noise the plain band mean is dominated by the noisiest band.
Equal fractional weighting is also computed; it is a poor estimator with noise (SFL EE ×464, BB ×2700 over ZEA) and is not quoted.

Results (weighted, combined, 260 skies; noise-free on the same skies in brackets):

| | TT | EE | BB |
|---|---|---|---|
| STG / ZEA | 1.03 ± 0.03 (1.07) | 1.03 ± 0.04 (1.10) | 1.08 ± 0.10 |
| SFL / ZEA | 6.9 ± 0.4 (9.5) | 2.9 ± 0.2 (8.4) | 54 ± 4 (1900) |

Residual bias: BB 0.05 ± 0.05 (ZEA), 0.03 ± 0.05 (STG), 0.06 ± 0.11 (SFL) of lensed BB; TT and EE agree with the 0.19–0.22% offset shared by all charts without noise (the known window/pipeline offset, not projection-specific).
SFL's corrected BB over 500–1000 scatters 10× ZEA's, 90% of that variance present without noise: the leaked E's sample variance.
SFL's uncorrected leaked BB: 111σ over 500–4000, 7σ over 2500–3000, 1.8σ over 3000–3500, though above lensed BB to ℓ ≈ 4300. The forward model matches the measured leaked BB through ℓ ≈ 3000 with lensing and noise present.

Choose-not-correct survives: correcting STG is nearly free at SPT-3G depth, correcting SFL is not, and for B modes the cost is a factor ~50 in variance.
Caveats: Gaussian lensed sky (no lensing non-Gaussianity), white noise with no scan-synchronous or 1/f component, no filtering, diagonal bin weighting.

Reproduce: `lc run --executor htcondor --profile htcondor9 noisy_inversion_sims -u highell`; `bash env.sh python3 analysis/noisy_inversion.py`; `bash env.sh python3 analysis/paper_numbers.py --noisy-only`.
Plots: `/sptlocal/user/cailmdaley/pol_project/noisy_inversion/{inversion,sfl_detectability}.png`.
