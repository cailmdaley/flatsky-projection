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

Residual bias of the inverse-variance-combined amplitude (the convention of the variance ratios): BB −0.001 ± 0.002 (ZEA), −0.001 ± 0.002 (STG), +0.030 ± 0.009 (SFL) of lensed BB.
SFL's comes from 500–1000, +0.075 ± 0.021 of lensed BB (0.080 ± 0.020 without noise), where the leak is 20× lensed BB: R underpredicts SFL's leaked BB there by 0.4%.
TT and EE keep offsets of 0.09–0.23%, present without noise (the known window/pipeline offset). The plain band mean of the seven bands is dominated by the noisiest band and hides the SFL bias; it is not quoted.
STG's cost falls from 1.07/1.10 (noise-free) to 1.03/1.03 (noisy), but a paired jackknife puts the drop at only 1.2σ/1.5σ, so the paper does not claim noise dilutes STG's cost; for SFL the dilution is clear.
SFL's corrected BB over 500–1000 scatters 10× ZEA's, 90% of that variance present without noise: the leaked E's sample variance.
SFL's uncorrected leaked BB: 111σ over 500–4000, 7σ over 2500–3000, 1.8σ over 3000–3500, though above lensed BB to ℓ ≈ 4300. The forward model matches the measured leaked BB through ℓ ≈ 3000 with lensing and noise present.

Choose-not-correct survives: correcting STG is nearly free at SPT-3G depth, correcting SFL is not, and for B modes the cost is a factor ~50 in variance.
Caveats: Gaussian lensed sky (no lensing non-Gaussianity), white noise with no scan-synchronous or 1/f component, no filtering, diagonal bin weighting.

Reproduce: `lc run --executor htcondor --profile htcondor9 noisy_inversion_sims -u highell`; `bash env.sh python3 analysis/noisy_inversion.py`; `bash env.sh python3 analysis/paper_numbers.py --noisy-only`.
Plots: `/sptlocal/user/cailmdaley/pol_project/noisy_inversion/{inversion,sfl_detectability}.png`.
