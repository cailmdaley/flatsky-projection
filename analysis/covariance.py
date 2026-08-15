"""Recompute the bandpower covariance/correlation structure from the 2021 spectra.

Why this exists: the 2021 analysis's covariance and correlation-matrix figures
survive only as images pasted into Overleaf -- no code that produced them is
in this checkout. `analysis/spectra.py` already knows how to load the raw
bandpowers (`Cl_flatsky_namaster_unlensed_{EE,BB}_{proj}_mask{i}.txt`, each
20 seeds x 400 bandpowers of width dl=10 out to l=4000); this module reuses
that loader and recomputes the seed-to-seed covariance/correlation directly
from the surviving text files, so the figures are reproducible again.

CAVEAT -- N=20 is few. A 400x400 sample correlation matrix from 20 seeds has
only 19 degrees of freedom per entry; off-diagonal entries have sampling
noise of order 1/sqrt(19) ~ 0.23 even when the true correlation is zero, so
any single configuration's correlation matrix is dominated by that noise, not
by real structure. Do not over-read one panel. Where possible this module
reports and plots the mean over the 5 projections x 7 masks = 35 configurations
(per spectrum). Note the averaging gain is much less than sqrt(35): all 35
configurations are built from the SAME 20 seeds on nested concentric disks, so
their sampling fluctuations are strongly correlated -- the mean suppresses
config-specific structure but not the shared seed noise. Treat any single
matrix as illustrative, and the mean as qualitative rather than definitive.

Usage:
    env-shell.sh python analysis/covariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spectra as S

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "figure.autolayout": False,
})

SPECS = ("EE", "BB")

# ell ranges (inclusive, in multipoles) for the off-diagonal-correlation
# summary -- chosen to sample the low-l, acoustic-peak, and high-l/band-limit
# regimes separately.
ELL_RANGES = [
    ("low", 0, 500),
    ("mid", 500, 2000),
    ("high", 2000, 4000),
]


def covariance_and_correlation(bandpowers):
    """Seed-to-seed covariance and correlation of one (proj, mask) config.

    `bandpowers` is (NSIMS, NBINS). Returns (cov, corr), each (NBINS, NBINS).
    NaN bandpowers (masked/undefined bins, if any) propagate to NaN rows/cols.
    """
    cov = np.cov(bandpowers, rowvar=False)
    diag = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(diag, diag)
    return cov, corr


def offdiag_mean(corr, lo_idx, hi_idx):
    """Mean of the off-diagonal correlation within a square sub-block."""
    block = corr[lo_idx:hi_idx, lo_idx:hi_idx]
    n = block.shape[0]
    if n < 2:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    return np.nanmean(block[mask])


def ell_to_bin(ell_lo, ell_hi):
    """Bandpower-index slice [lo, hi) covering the multipole range [ell_lo, ell_hi)."""
    lo = int(np.searchsorted(S.BIN_LO, ell_lo, side="left"))
    hi = int(np.searchsorted(S.BIN_LO, ell_hi, side="left"))
    return lo, hi


def plot_correlation_matrix(corr, title, path):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(
        corr, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
        extent=[S.ELL[0], S.ELL[-1], S.ELL[0], S.ELL[-1]],
    )
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"multipole $\ell$")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("correlation")
    fig.savefig(path)
    plt.close(fig)


def plot_diagonal_error_comparison(all_cov, spec, path):
    """Bandpower std dev vs ell, one line per projection, averaged over masks.

    all_cov: dict[(proj, mask_idx)] -> cov matrix, for this spec.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(S.PROJS)))
    for proj, color in zip(S.PROJS, colors):
        stds = np.array([
            np.sqrt(np.diag(all_cov[(proj, i)]))
            for i in range(len(S.PERC))
            if (proj, i) in all_cov
        ])
        mean_std = np.nanmean(stds, axis=0)
        ax.plot(S.ELL, mean_std, color=color, label=S.PROJ_LABEL[proj])
    ax.set_yscale("log")
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"bandpower std.\ dev.\ [$\mu K^2$] (mean over masks)")
    ax.set_title(f"{spec}: bandpower error vs.\\ projection")
    ax.legend(ncol=2)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    summary_lines = []

    for spec in SPECS:
        # (n_masks, n_projs, NSIMS, NBINS)
        data = S.load(spec)

        cov_by_config = {}
        corr_by_config = {}
        for i, perc in enumerate(S.PERC):
            for j, proj in enumerate(S.PROJS):
                bp = data[i, j]
                # The 10% (mask6) runs for SFL, SIN and BICEP are broken on
                # disk (all-NaN or overflowed to ~1e246; the disk swallows the
                # south celestial pole -- see shear_leakage.py). Drop them
                # entirely, matching the shear-law fit, rather than nan-meaning
                # around partial garbage.
                if np.isnan(bp).any() or np.nanmax(np.abs(bp)) > 1e100:
                    continue
                cov, corr = covariance_and_correlation(bp)
                cov_by_config[(proj, i)] = cov
                corr_by_config[(proj, i)] = corr

        # Per-configuration correlation-matrix figures.
        spec_dir = OUT / "covariance" / spec
        spec_dir.mkdir(parents=True, exist_ok=True)
        for (proj, i), corr in corr_by_config.items():
            perc = S.PERC[i]
            title = (
                f"{spec} correlation -- {S.PROJ_LABEL[proj]}, "
                f"$f_{{\\rm sky}}$={perc:.2f}\\%"
            )
            path = spec_dir / f"corr_{proj}_mask{i}.png"
            plot_correlation_matrix(corr, title, path)

        # Mean correlation matrix over all 35 configurations -- averages down
        # the N=20 sampling noise by ~sqrt(35), see module docstring.
        mean_corr = np.nanmean(np.stack(list(corr_by_config.values())), axis=0)
        plot_correlation_matrix(
            mean_corr,
            f"{spec} correlation -- mean over {len(corr_by_config)} configs",
            OUT / f"covariance_{spec}_mean_corr.png",
        )

        # Diagonal-error comparison across projections.
        plot_diagonal_error_comparison(
            cov_by_config, spec, OUT / f"covariance_{spec}_diag_error.png",
        )

        # Numerical summary: mean off-diagonal correlation per ell range,
        # for the mean-over-configs matrix (the noise-suppressed estimate)
        # and for the spread across individual configurations.
        summary_lines.append(f"\n{spec}:")
        for name, ell_lo, ell_hi in ELL_RANGES:
            lo, hi = ell_to_bin(ell_lo, ell_hi)
            mean_val = offdiag_mean(mean_corr, lo, hi)
            per_config = np.array([
                offdiag_mean(c, lo, hi) for c in corr_by_config.values()
            ])
            summary_lines.append(
                f"  ell in [{ell_lo:5d}, {ell_hi:5d}): "
                f"mean-over-configs off-diag corr = {mean_val:+.3f}   "
                f"per-config range = [{np.nanmin(per_config):+.3f}, "
                f"{np.nanmax(per_config):+.3f}]  "
                f"(std across configs = {np.nanstd(per_config):.3f})"
            )

    print("\n".join(summary_lines))
    print(f"\nfigures written under {OUT}")


if __name__ == "__main__":
    main()
