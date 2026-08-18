"""One-page figure for the SPT thread: proj0/proj5 in TT and EE.

Two panels. Left TT, right EE. In both: Crawford's measured full-mask ratio
(one realization, from his pickles), and the continuous zero-parameter
prediction -- `crawford_model.geometry_of` for his real mask geometry, the
Planck 2018 lensed spectrum as input, evaluated unbinned on a fine ell grid
rather than only on his 100-wide bandpowers.

The EE panel additionally carries our own 2021 NaMaster measurement: mask 5
(the 5%-sky apodized disk at dec -56, radius 0.45 rad, whose W^2-weighted
<g^2> = 0.0262 -- one of the two 2021 disks that bracket the 1500d
footprint's 0.0442), SFL over ZEA, mean over the 20 paired seeds, rebinned
from dl = 10 to dl = 100 by summing each leg inside the coarse bin. No
transfer function is divided out anywhere: pixel windows cancel in the ratio.

Run inside the project container:
    systemd-run --user --scope -p MemoryMax=4G bash env.sh python3 onepager/plot_onepager.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(HERE.parent / "src" / "stages"))

import crawford as C            # noqa: E402
import crawford_model as CMOD   # noqa: E402
import geometry as G            # noqa: E402
import spectra as S             # noqa: E402
import paper_style as PS        # noqa: E402

RESAMPLING_L = 5400.0     # only 1-2 clean bandpowers remain above this
BANDLIMIT_L = 4000.0      # our 2021 sims are band-limited here
SUSPECT_L = 3500.0        # below this, comfortably clear of the band limit
MASK_INDEX = 5            # the 5%-sky disk, R = 0.45 rad, <g^2> = 0.0262

DATA = PS.SECONDARY[0]     # Crawford's points: near-black
MODEL = "#A05638"          # the prediction curve
OURS = "#2E6F6F"           # this work: a teal that is no projection's colour
BAND = "#C7924A"


def prediction_curve(spec, ell_max=6000.0, step=2.0):
    """Continuous predicted ratio: unbinned, on a fine ell grid."""
    s1, s2, w = CMOD.geometry_of("full")
    ell_th, cl_th = CMOD.theory_input(spec)
    ell = np.arange(step, ell_max + step, step)
    proj0 = G.predicted_spectrum(ell, s1, s2, w, ell_th, cl_th,
                                 band_limit=CMOD.BAND_LIMIT,
                                 n_bins=64, n_psi=96)
    proj5 = np.interp(ell, ell_th, cl_th)
    return ell, proj0 / proj5


def our_disk_prediction(ell_max=4000.0):
    """The same zero-parameter model, evaluated over *our* disk and our input.

    Radius 0.45 rad about (0, -56), C2 taper 2.5 deg -- mask 5 of the paper
    suite; the sims' own unlensed EE spectrum; and the sims' hard band limit
    at ell = 4000 rather than Crawford's 7999. Nothing is fitted here either.
    """
    import healpy as hp
    from make_masks import apodize_disk_analytic

    center = hp.ang2vec(0.0, -56.0, lonlat=True)
    w_full = apodize_disk_analytic(2048, center, 0.45, "C2", 2.5)
    pix = np.nonzero(w_full > 0)[0]
    w = w_full[pix].astype(float)
    del w_full
    lon, lat = hp.pix2ang(2048, pix, lonlat=True)
    a = G.relative_jacobian("ProjSFL", "ProjZEA", lon, lat,
                            to_kw=dict(delta_center_deg=-56.0),
                            from_kw=dict(delta_center_deg=-56.0))
    s1, s2 = G.singular_values(a)
    g = (s1 - s2) / (s1 + s2)
    w2 = w ** 2
    g2 = float(np.average(g ** 2, weights=w2))

    with np.load(S.THEORY_NPZ) as data:
        ell_th = data["ell"].astype(float)
        cl_th = data["unlensed_EE"]

    ell = np.arange(1.0, ell_max + 1.0)
    proj0 = G.predicted_spectrum(ell, s1, s2, w2, ell_th, cl_th,
                                 band_limit=4000, n_bins=64, n_psi=96)
    proj5 = np.interp(ell, ell_th, cl_th)
    # Binned to dl = 100 exactly as the measurement is, so the comparison is
    # like for like rather than a fine curve read against coarse points.
    n_coarse = int(ell_max // C.DELTA_L)
    b0 = CMOD.bin_like_crawford(ell, proj0, n_bins=n_coarse)
    b5 = CMOD.bin_like_crawford(ell, proj5, n_bins=n_coarse)
    return C.ELL[:n_coarse], b0 / b5, g2


def our_ee_ratio():
    """SFL/ZEA EE ratio from the 2021 NaMaster bandpowers, rebinned to dl=100.

    Sums each leg over the ten dl = 10 bins inside a dl = 100 bin, then
    divides -- the estimator's own weighting, and identical to an
    ell-weighted mean to well under a percent.
    """
    ee = S.load("EE")                       # (masks, projs, seeds, bins)
    i_sfl = S.PROJS.index("ProjSFL")
    i_zea = S.PROJS.index("ProjZEA")
    sfl = np.nanmean(ee[MASK_INDEX, i_sfl], axis=0)   # (400,)
    zea = np.nanmean(ee[MASK_INDEX, i_zea], axis=0)
    n_coarse = S.NBINS // 10                          # 40 bins of dl = 100
    sfl_c = sfl.reshape(n_coarse, 10).sum(axis=1)
    zea_c = zea.reshape(n_coarse, 10).sum(axis=1)
    return C.ELL[:n_coarse], sfl_c / zea_c


def main():
    plt.rcParams.update(PS.rc(font_size=9.0))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.fontsize": 8.0,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.155,
                        wspace=0.07)

    ell_ours, ratio_ours = our_ee_ratio()
    ell_dp, ratio_dp, g2_disk = our_disk_prediction()
    report = {}

    for ax, (spec, label) in zip(axes, [("tt", "TT"), ("ee", "EE")]):
        ell_p, ratio_p = prediction_curve(spec)
        ell_m, ratio_m = CMOD.measured("", spec)
        report[spec] = (ell_m, ratio_m, ell_p, ratio_p)

        ax.axvspan(RESAMPLING_L, 6000, color=BAND, alpha=0.13, lw=0, zorder=0)
        ax.axhline(1.0, color=PS.RULE, lw=0.7, zorder=0)

        ax.plot(ell_p, ratio_p, "-", color=MODEL, lw=1.9, zorder=3,
                label="prediction (zero parameters)")
        ax.plot(ell_m, ratio_m, "o", color=DATA, ms=3.0, mew=0, zorder=4,
                label="Crawford (cl_flatsky, 1500d masks)")

        if spec == "ee":
            ax.plot(ell_dp, ratio_dp, "--", color=OURS, lw=1.1, alpha=0.85,
                    zorder=3, label="same model, our disk and band limit")
            clean = ell_ours < SUSPECT_L
            edge = (ell_ours >= SUSPECT_L) & (ell_ours <= BANDLIMIT_L)
            ax.plot(ell_ours[clean], ratio_ours[clean], "D", color=OURS,
                    ms=3.6, mew=0.9, mec=OURS, zorder=5,
                    label="this work (NaMaster, 5 per cent sky disk)")
            ax.plot(ell_ours[edge], ratio_ours[edge], "D", color="none",
                    ms=3.6, mew=0.9, mec=OURS, zorder=5)
            ax.axvline(BANDLIMIT_L, color=OURS, lw=0.7, ls=":", zorder=1)
            ax.text(BANDLIMIT_L - 90, 0.98, "our band limit", rotation=90,
                    ha="right", va="bottom", fontsize=6.8, color=OURS)

        ax.set_title(label, pad=5)
        ax.set_xlim(0, 6000)
        ax.set_xlabel(r"multipole $\ell$")

    axes[0].set_ylabel(r"$C_\ell(\mathrm{proj0})\,/\,C_\ell(\mathrm{proj5})$")
    axes[0].text(5950, 1.02, "resampling\nregime", ha="right", va="bottom",
                 fontsize=6.8, color="#8a5a1e")
    axes[1].legend(loc="upper left", handlelength=1.6, borderaxespad=0.3,
                   labelspacing=0.35)

    path = HERE / "onepager_ratio.pdf"
    fig.savefig(path)
    fig.savefig(HERE / "onepager_ratio.png", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")

    print(f"\nour disk <g^2> = {g2_disk:.4f}")
    print("--- this work, EE, SFL/ZEA (mask %d, dl=100) ---" % MASK_INDEX)
    for l0 in (1050, 2050, 3050, 3550, 3950):
        i = int(l0 // C.DELTA_L)
        j = int(np.argmin(np.abs(ell_dp - ell_ours[i])))
        print(f"  l={ell_ours[i]:6.0f}  ratio={ratio_ours[i]:.4f}"
              f"  disk-pred={ratio_dp[j]:.4f}")
    print("\n--- Crawford measured / prediction (binned to his bins) ---")
    for spec in ("tt", "ee"):
        ell_m, ratio_m, ell_p, ratio_p = report[spec]
        binned = CMOD.bin_like_crawford(ell_p, ratio_p)
        for l0 in (1050, 2050, 3050, 4050, 5050):
            i = int(l0 // C.DELTA_L)
            print(f"  {spec.upper()} l={ell_m[i]:6.0f}  meas={ratio_m[i]:6.3f} "
                  f" pred={binned[i]:6.3f}  meas/pred={ratio_m[i]/binned[i]:6.3f}")


if __name__ == "__main__":
    main()
