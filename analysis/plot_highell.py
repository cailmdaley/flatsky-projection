"""The high-ell extension: both channels of the distortion, out to the
2 arcmin pixel wall.

Two panels, one figure, mirroring the paper's two channels on the same
footprint (the 5% disk), the same leg (QU) and the same 20-seed average.

Left, the scalar channel. Measured EE divided by the input EE. The sheared
projections ring with the acoustic peaks -- the scalar remap moving power in
l -- and SFL and BICEP climb past 1.5 in the fourth peak region. ZEA, equal
area and barely sheared, holds within a few percent of unity almost to the
band limit; STG, conformal and shear-free, loses power monotonically to its
magnification.

Right, the leakage channel. BB/EE against each projection's own 2<g^2>. The
law is flat and near prediction for SFL, SIN and ZEA to l ~ 4800. Above that
every geometry -- including STG, which has no shear at all -- rises onto a
common curve, which is the 2 arcmin pixel scale (l_Nyq = 5400) and not the
projection: that band is shaded and excluded. STG's slowly rising floor below
the wall is the flattening / resampling floor and is shown rather than hidden.

Data. `results/highell/namaster_spectra/bandpowers_{proj}_seed{s:04d}.npz`,
5 projections x 20 seeds x 7 masks, dl = 50 bins to l = 6000, two legs
(0 = QU, 1 = EB). The run divided by a HEALPix pixel window it should not
have; that division is undone here by multiplying back the binned
hp.pixwin(8192)^2, exactly as `analysis/highell_firstlook.py` does.

Run:  systemd-run --user --scope -p MemoryMax=8G bash env.sh python3 analysis/plot_highell.py
Out:  figures/highell.pdf, figures/highell.png, figures/highell_numbers.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pymaster as nmt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectra as S            # noqa: E402
import shear_leakage as SL     # noqa: E402
import theory as T             # noqa: E402
import paper_style as PS       # noqa: E402

ROOT = HERE.parent
FIG_PDF = ROOT / "figures" / "highell.pdf"
FIG_PNG = ROOT / "figures" / "highell.png"
NUMBERS = ROOT / "figures" / "highell_numbers.json"

MASK_INDEX = 5           # PERC[5] = 5.0%, the paper's reference footprint
N_SEEDS = 20
DL = 50.0
LMAX = 6000.0
QU = 0                   # leg index

L_WALL = 4800.0          # where every geometry starts onto the common curve
L_NYQ = 5400.0           # 2 arcmin pixel Nyquist
L_MIN_FIT = 500.0
L_PLOT_MIN = 300.0

SHEARED = ["ProjSFL", "ProjSIN", "ProjZEA", "ProjBICEP"]

# The highell sims carry sky power to l = 6000, not the 4000 of the 2021 run
# that `shear_leakage.predicted_spectra` defaults to.
BAND_LIMIT = 6000.0

MS, MEW = 2.6, 0.8       # marker convention of the projection-spectra figures


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load():
    """Seed-resolved EE, BB on the 5% disk, pixel-window artefact undone."""
    nb = int(LMAX / DL)
    ee = np.full((len(S.PROJS), N_SEEDS, nb), np.nan)
    bb = np.full_like(ee, np.nan)
    ell = None
    for j, proj in enumerate(S.PROJS):
        for s in range(N_SEEDS):
            f = (ROOT / "results" / "highell" / "namaster_spectra"
                 / f"bandpowers_{proj}_seed{s:04d}.npz")
            with np.load(f) as z:
                ell = z["ell_eff"]
                ee[j, s] = z["ee"][QU, 0, MASK_INDEX, 0, :]
                bb[j, s] = z["bb"][QU, 0, MASK_INDEX, 0, :]

    b = nmt.NmtBinFlat(np.arange(0, LMAX, DL).astype(float),
                       np.arange(DL, LMAX + DL, DL).astype(float))
    pw = hp.pixwin(8192)[: int(LMAX) + 1] ** 2
    hpc = b.bin_cell(np.arange(int(LMAX) + 1, dtype=float), pw)
    ee = ee * hpc
    bb = bb * hpc

    # Input EE on the same bandpowers, for the scalar-channel ratio.
    th = T.load()
    cl = np.zeros(int(LMAX) + 1)
    src = th["unlensed_EE"][: int(LMAX) + 1]
    cl[: len(src)] = src
    cl_ee = b.bin_cell(np.arange(int(LMAX) + 1, dtype=float), cl)

    mom = SL.load_moments(verbose=False)
    g2 = mom["g2"][MASK_INDEX]         # (n_proj,)

    ell_th = th["ell"].astype(float)
    cl_th = th["unlensed_EE"].astype(float)
    m_bb = np.empty((len(S.PROJS), len(ell)))
    m_ee = np.empty_like(m_bb)
    for j in range(len(S.PROJS)):
        m_bb[j], m_ee[j] = model_curves(mom, j, ell, ell_th, cl_th)

    return ell, ee, bb, cl_ee, g2, m_bb, m_ee


def model_curves(mom, j, ell, ell_theory, cl_theory):
    """Forward-modelled BB(l), EE(l) as absolute C_l, for projection `j`.

    Same normalisation as `plot_projection_spectra.model_curves`: the
    numerator carries W^2 |det J|^2 per unit sky area, the pseudo-C_l
    denominator the same integral one power of |det J| lower.
    """
    i = MASK_INDEX
    h = mom["hist"][i, j]
    occ = h > 0
    s1, s2, w = mom["cell_s1"][i, j][occ], mom["cell_s2"][i, j][occ], h[occ]
    norm = (w * s1 * s2).sum()
    bb, ee = SL.predicted_spectra(mom, i, j, ell, ell_theory, cl_theory,
                                  band_limit=BAND_LIMIT)
    return bb / norm, ee / norm


def rebin100(ell, y):
    """dl = 50 -> dl = 100. Bin edges already align."""
    y = np.asarray(y, dtype=float)
    n = (y.shape[-1] // 2) * 2
    return (ell[:n].reshape(-1, 2).mean(1),
            y[..., :n].reshape(y.shape[:-1] + (-1, 2)).mean(-1))


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def at(ell, y, l0):
    return float(np.interp(l0, ell, y))


def smooth(y, n=5):
    """Running mean over `n` bandpowers, edges held."""
    k = np.ones(n) / n
    return np.convolve(np.asarray(y, float), k, mode="same") / np.convolve(
        np.ones(len(y)), k, mode="same")


def departure_ell(ell, ratio, target, frac=0.10):
    """First l > L_MIN_FIT where the measurement leaves `target` (the forward
    model, same binning) by more than `frac`, and stays out for the next bin
    too -- one bin of acoustic ringing is not a departure. Both curves are
    passed in already smoothed."""
    out = np.abs(ratio / target - 1.0) > frac
    for k in range(len(ell) - 1):
        if ell[k] <= L_MIN_FIT:
            continue
        if out[k] and out[k + 1]:
            return float(ell[k])
    return None


def band_stats(per_seed, model, ell, lo, hi):
    """mean(measured/model) over a band, and the seed-to-seed scatter of it.

    `per_seed` is (n_seeds, n_bins); the seed-averaged mean divides the
    seed-averaged measurement by the model, which is what the quoted ratio
    means, while the scatter comes from the per-seed band means.
    """
    sel = (ell > lo) & (ell < hi)
    seedwise = (per_seed[:, sel] / model[sel]).mean(1)
    mean = float((per_seed[:, sel].mean(0) / model[sel]).mean())
    sd = float(seedwise.std(ddof=1))
    return {"mean": mean, "seed_scatter": sd,
            "error_on_mean": float(sd / np.sqrt(per_seed.shape[0]))}


def numbers(ell, ee, bb, cl_ee, g2, m_bb, m_ee, ours100):
    d = {"mask": "5% disk (PERC index 5)", "leg": "QU", "n_seeds": N_SEEDS,
         "l_wall": L_WALL, "l_nyquist": L_NYQ,
         "model_band_limit": BAND_LIMIT,
         "model": "shear_leakage.predicted_spectra, restored-terms forward "
                  "model, normalised as in plot_projection_spectra",
         "projections": {}}

    ratio_seed = bb / ee                      # (proj, seed, nb)
    ratio_mean = bb.mean(1) / ee.mean(1)
    scalar_seed = ee / cl_ee                  # (proj, seed, nb)
    scalar = ee.mean(1) / cl_ee
    m_ratio = m_bb / m_ee                     # model BB/EE
    m_scalar = m_ee / cl_ee                   # model EE / input EE

    bands = {"500_4800": (L_MIN_FIT, L_WALL), "3000_4800": (3000.0, L_WALL)}
    for j, proj in enumerate(S.PROJS):
        e = {"g2": float(g2[j]), "two_g2": float(2 * g2[j]),
             "bb_over_ee_1500": at(ell, ratio_mean[j], 1500.0),
             "bb_over_ee_4500": at(ell, ratio_mean[j], 4500.0),
             "ee_over_input": {f"{L:.0f}": at(ell, scalar[j], L)
                               for L in (1000.0, 3000.0, 4500.0, 5500.0)},
             "ee_over_input_model": {f"{L:.0f}": at(ell, m_scalar[j], L)
                                     for L in (1000.0, 3000.0, 4500.0, 5500.0)},
             "ee_over_model": {}, "bb_over_ee_over_model": {}}

        for name, (lo, hi) in bands.items():
            e["ee_over_model"][name] = band_stats(
                scalar_seed[j], m_scalar[j], ell, lo, hi)
        e["ee_over_model_departure_ell_10pct"] = departure_ell(
            ell, smooth(scalar[j]), smooth(m_scalar[j]))

        if proj in SHEARED:
            for name, (lo, hi) in bands.items():
                e["bb_over_ee_over_model"][name] = band_stats(
                    ratio_seed[j], m_ratio[j], ell, lo, hi)
            e["bb_over_ee_over_model_departure_ell_10pct"] = departure_ell(
                ell, smooth(ratio_mean[j]), smooth(m_ratio[j]))
        d["projections"][S.PROJ_LABEL[proj]] = e

    d["sfl_over_zea_EE"] = {f"{L:.0f}": at(rebin100(ell, ell)[0], ours100, L)
                            for L in (4050.0, 4550.0)}
    return d


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def make_figure(ell, ee, bb, cl_ee, m_bb, m_ee):
    plt.rcParams.update(PS.rc(usetex=True, font_size=9.0))
    plt.rcParams.update({"axes.labelsize": 9.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))

    scalar = ee.mean(1) / cl_ee
    ratio_mean = bb.mean(1) / ee.mean(1)
    m_scalar = m_ee / cl_ee
    m_ratio = m_bb / m_ee
    keep = ell > L_PLOT_MIN
    band = dict(color=PS.SECONDARY[2], alpha=0.30, lw=0, zorder=0)

    # ---- left: the scalar channel ----------------------------------------
    ax = axes[0]
    ax.axvspan(L_WALL, LMAX, **band)
    ax.axhline(1.0, color=PS.RULE, lw=0.8, zorder=1)
    handles = []
    for j, proj in enumerate(S.PROJS):
        col = PS.PROJ_COLORS[proj]
        h, = ax.plot(ell[keep], scalar[j][keep], PS.PROJ_MARKERS[proj],
                     color=col, ms=MS, mfc="none", mew=MEW,
                     label=PS.PROJ_LABEL[proj])
        handles.append(h)
        ax.plot(ell[keep], m_scalar[j][keep], "-", color=col, lw=1.3,
                alpha=0.9)
    ax.set_xlim(L_PLOT_MIN, LMAX)
    ax.set_ylim(0.0, 4.6)
    ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000])
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"measured $C_\ell^{EE}$ / input $C_\ell^{EE}$")
    ax.set_title(r"The scalar remap grows through the damping tail",
                 fontsize=9.5, pad=6)

    # ---- right: the leakage channel --------------------------------------
    ax = axes[1]
    ax.axvspan(L_WALL, LMAX, **band)
    for j, proj in enumerate(S.PROJS):
        col = PS.PROJ_COLORS[proj]
        ax.plot(ell[keep], ratio_mean[j][keep], PS.PROJ_MARKERS[proj],
                color=col, ms=MS, mfc="none", mew=MEW)
        if proj in SHEARED:
            ax.plot(ell[keep], m_ratio[j][keep], "-", color=col, lw=1.3,
                    alpha=0.9)
    ax.set_yscale("log")
    ax.text(0.985, 0.045, r"2$'$ pixel wall" "\n" r"$\ell_{\rm Nyq}=5400$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=PS.INK_SOFT)
    ax.set_xlim(L_PLOT_MIN, LMAX)
    ax.set_ylim(1e-8, 5.0)
    ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000])
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$C_\ell^{BB}/C_\ell^{EE}$")
    ax.set_title(r"The leakage tracks the forward model to the pixel wall",
                 fontsize=9.5, pad=6)

    # One legend for the figure: five projections plus the model convention.
    model_line = matplotlib.lines.Line2D([], [], color=PS.INK_SOFT, lw=1.3)
    fig.legend(handles + [model_line],
               [PS.PROJ_LABEL[p] for p in S.PROJS] + ["forward model"],
               fontsize=8.0, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               handletextpad=0.5, handlelength=1.6, frameon=False)

    fig.tight_layout(pad=0.9, w_pad=1.6, rect=(0, 0, 1, 0.925))
    FIG_PDF.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote {FIG_PDF}\nwrote {FIG_PNG}")


def main():
    ell, ee, bb, cl_ee, g2, m_bb, m_ee = load()
    jS, jZ = S.PROJS.index("ProjSFL"), S.PROJS.index("ProjZEA")
    _, ee100 = rebin100(ell, ee.mean(1))
    ours100 = ee100[jS] / ee100[jZ]

    make_figure(ell, ee, bb, cl_ee, m_bb, m_ee)

    d = numbers(ell, ee, bb, cl_ee, g2, m_bb, m_ee, ours100)
    NUMBERS.write_text(json.dumps(d, indent=2))
    print(f"wrote {NUMBERS}\n")

    def fmt(x):
        return "none below 6000" if x is None else f"{x:.0f}"

    print("5% disk, QU leg, 20-seed mean; ratios are measured / forward model")
    for label, e in d["projections"].items():
        s, ms_ = e["ee_over_input"], e["ee_over_input_model"]
        print(f"  {label}")
        print("         EE/input  at 1000/3000/4500/5500 = "
              f"{s['1000']:.3f} {s['3000']:.3f} {s['4500']:.3f} {s['5500']:.3f}"
              "   model = "
              f"{ms_['1000']:.3f} {ms_['3000']:.3f} {ms_['4500']:.3f} "
              f"{ms_['5500']:.3f}")
        a = e["ee_over_model"]["500_4800"]; b = e["ee_over_model"]["3000_4800"]
        print(f"         EE / model:    500-4800 {a['mean']:.3f} +/- "
              f"{a['seed_scatter']:.3f} | 3000-4800 {b['mean']:.3f} +/- "
              f"{b['seed_scatter']:.3f} | 10% departure at l = "
              + fmt(e["ee_over_model_departure_ell_10pct"]))
        print(f"         2<g2>={e['two_g2']:.4e}  "
              f"BB/EE(1500)={e['bb_over_ee_1500']:.3e}  "
              f"(4500)={e['bb_over_ee_4500']:.3e}")
        if e["bb_over_ee_over_model"]:
            a = e["bb_over_ee_over_model"]["500_4800"]
            b = e["bb_over_ee_over_model"]["3000_4800"]
            print(f"         BB/EE / model: 500-4800 {a['mean']:.3f} +/- "
                  f"{a['seed_scatter']:.3f} | 3000-4800 {b['mean']:.3f} +/- "
                  f"{b['seed_scatter']:.3f} | 10% departure at l = "
                  + fmt(e["bb_over_ee_over_model_departure_ell_10pct"]))
    print("\nSFL/ZEA EE ratio, 5% disk, dl=100 bins")
    for l0, v in d["sfl_over_zea_EE"].items():
        print(f"  l={l0}: {v:.3f}")


if __name__ == "__main__":
    main()
