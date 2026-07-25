"""What the projection does over SPT's real 1500d footprint.

The question this answers
-------------------------
The 2021 suite measured the SFL/ZEA (= proj0/proj5) EE ratio on seven
concentric apodized disks and found it growing with sky area. SPT's actual
survey field is not a disk: 1500d spans about +-50 deg in RA and 28 deg in
declination. Since SFL's shear is set by RA offset from the reference meridian,
a wide-in-RA field is far more sheared than a disk of the same area, and the
2021 area trend cannot be interpolated to it.

So this script does two things:

1. **Validates.** It predicts the SFL/ZEA ratio for each 2021 disk from
   projection geometry alone and puts it beside the measured bandpowers. The
   forward model has no free parameters -- if it tracks the measurements, the
   mechanism is established and the model can be trusted off the training set.

2. **Extrapolates, correctly.** It then applies the same model to the 1500d
   rectangle. The answer does not lie between the 3% and 5% disk values; it
   lies above both, because area is the wrong abscissa.

Caveats that travel with every number here
------------------------------------------
The model is geometry only. It contains no mask mode-coupling, no bandpower
window, no purification, and -- importantly -- no pixel-window or interpolation
aliasing, which Caterina Umilta measured separately and which adds power at
high l on top of this. So these are predictions for the *shear* contribution
alone, and should be read as a floor rather than a total.

Ratios are quoted as band averages. The forward model smears acoustic peaks,
so the single-l ratio swings violently between peaks and troughs; a ratio at
one multipole of an unlensed spectrum is not a meaningful number.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import footprints as F
import geometry as G
import spectra as S

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# SPT-3G 1500d ("winter") field. Half-width in RA, declination limits.
FIELD_1500D = dict(ra_half_deg=50.0, dec_min_deg=-70.0, dec_max_deg=-42.0)

BANDS = [(1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def theory():
    """The sims' own unlensed EE input, as raw C_l on integer l."""
    cl = np.loadtxt(ROOT / "Clee.dat")
    cl[0] = 0.0
    return np.arange(len(cl)), cl


def band_mean(ell, y, lo, hi):
    sel = (ell >= lo) & (ell < hi)
    return float(np.nanmean(y[sel])) if sel.any() else np.nan


def _running_band(ell, y, width=400.0):
    """Running mean over a fixed multipole width.

    Not a smoothing filter for cosmetics: the ratio's oscillation is a real,
    acoustic-locked feature, and averaging over a width wide compared to the
    peak spacing is what recovers the underlying trend the excess is about.
    """
    ell = np.asarray(ell, dtype=float)
    centres = np.arange(ell.min() + width / 2, ell.max() - width / 2 + 1, 50.0)
    out = [np.nanmean(y[np.abs(ell - c) <= width / 2]) for c in centres]
    return centres, np.asarray(out)


def ratio_for(mask, ell, ell_th, cl_th, band_limit=None):
    """Predicted C_l^EE(SFL) / C_l^EE(ZEA) over a mask, plus the shear stats."""
    got = {}
    for proj in ("ProjSFL", "ProjZEA"):
        s1, s2, w = F.mask_geometry(mask, proj)
        got[proj] = (
            G.predicted_spectrum(ell, s1, s2, w, ell_th, cl_th,
                                 band_limit=band_limit),
            s1 / s2, w,
        )
    q, w = got["ProjSFL"][1], got["ProjSFL"][2]
    return got["ProjSFL"][0] / got["ProjZEA"][0], q, w


def main():
    ell_th, cl_th = theory()
    ell_fine = np.arange(50.0, 5001.0, 20.0)

    # ---- 1. the 2021 disks: prediction against measurement -----------------
    measured = S.load("EE").mean(axis=2)
    i_sfl, i_zea = S.PROJS.index("ProjSFL"), S.PROJS.index("ProjZEA")
    meas_ratio = measured[:, i_sfl] / measured[:, i_zea]

    disks = []
    for radius, perc in zip(F.DISK_RADII_RAD, F.DISK_PERC):
        mask = F.disk_mask(radius)
        # Band-limited at 4000 to match the sims these are validated against.
        r, q, w = ratio_for(mask, S.ELL, ell_th, cl_th,
                            band_limit=G.SIM_BAND_LIMIT)
        disks.append(dict(perc=perc, fsky=F.f_sky(mask), ratio=r,
                          q_mean=float(np.average(q, weights=w))))

    # ---- 2. the real footprint ---------------------------------------------
    field = F.rect_mask(**FIELD_1500D)
    r_field, q_field, w_field = ratio_for(field, ell_fine, ell_th, cl_th)
    qq = F.weighted_quantiles(q_field, w_field, (0.05, 0.5, 0.95))

    print(f"1500d footprint: f_sky = {F.f_sky(field)*100:.2f}% "
          f"({F.f_sky(field)*41253:.0f} deg^2)")
    print(f"  SFL shear ratio s1/s2: mean {np.average(q_field, weights=w_field):.3f}, "
          f"5/50/95% {qq.round(3)}, max {q_field.max():.3f}")
    for lo, hi in BANDS:
        print(f"  predicted SFL/ZEA over {lo}-{hi}: "
              f"{band_mean(ell_fine, r_field, lo, hi):.2f}")
    print()
    print(f"{'disk %':>7} {'f_sky%':>7} {'<q>':>6} {'measured':>9} {'predicted':>10}")
    for d in disks:
        m = band_mean(S.ELL, meas_ratio[F.DISK_PERC.index(d['perc'])], 3000, 4000)
        p = band_mean(S.ELL, d["ratio"], 3000, 4000)
        print(f"{d['perc']:7.2f} {d['fsky']*100:7.2f} {d['q_mean']:6.3f} "
              f"{m:9.3f} {p:10.3f}")

    # ---- 3. the figure ------------------------------------------------------
    # Three panels: the model is right; here is why 1500d differs; here is the
    # prediction. Raw curves oscillate hard because the sims' input is
    # *unlensed* EE, whose troughs are deep -- shear smears peaks into troughs,
    # so the ratio swings. The running band mean carries the trend; the thin
    # curve is kept so the oscillation is not hidden.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    colours = plt.cm.viridis(np.linspace(0, 0.85, len(disks)))

    ax = axes[0]
    for d, colour in zip(disks, colours):
        if d["perc"] == 10.0:
            continue  # no surviving measurement to compare against
        i = F.DISK_PERC.index(d["perc"])
        ax.plot(S.ELL, meas_ratio[i], lw=1.4, alpha=0.75, color=colour)
        ax.plot(S.ELL, d["ratio"], lw=0.9, ls="--", color="k", alpha=0.8)
    ax.plot([], [], lw=1.4, color="0.4", label="measured (2021 sims)")
    ax.plot([], [], lw=0.9, ls="--", color="k", label="geometry model")
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set(xlim=(200, 4000), ylim=(0.6, 2.6), xlabel=r"$\ell$",
           ylabel=r"$C_\ell^{EE}(\mathrm{SFL})\,/\,C_\ell^{EE}(\mathrm{ZEA})$",
           title="A model with no free parameters")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for d, colour in zip(disks, colours):
        if d["perc"] not in (0.75, 3.0, 5.0):
            continue
        mask = F.disk_mask(F.DISK_RADII_RAD[F.DISK_PERC.index(d["perc"])])
        s1, s2, w = F.mask_geometry(mask, "ProjSFL")
        ax.hist(s1 / s2, bins=np.linspace(1, 2.4, 90), weights=w / w.sum(),
                histtype="step", lw=1.6, color=colour,
                label=rf"{d['perc']:g}\% disk ({d['fsky']*100:.1f}\% sky)")
    ax.hist(q_field, bins=np.linspace(1, 2.4, 90),
            weights=w_field / w_field.sum(), histtype="stepfilled", lw=1.8,
            color="crimson", alpha=0.35, edgecolor="crimson",
            label=rf"1500d field ({F.f_sky(field)*100:.1f}\% sky)")
    ax.set(xlabel=r"local shear $s_1/s_2$ of SFL", ylabel="mask$^2$ weight",
           title="1500d is sheared like a much larger disk")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(ell_fine, r_field, lw=0.7, color="crimson", alpha=0.18)
    ax.plot(*_running_band(ell_fine, r_field), lw=2.4, color="crimson",
            label="1500d footprint")
    for d, colour in zip(disks, colours):
        if d["perc"] not in (3.0, 5.0):
            continue
        ax.plot(*_running_band(S.ELL, d["ratio"]), lw=1.6, ls="--",
                color=colour, label=rf"{d['perc']:g}\% disk")
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.axvspan(4000, 5000, color="0.85", alpha=0.5, zorder=0)
    ax.text(4500, 0.72, "past the 2021 band limit", fontsize=7,
            color="0.35", ha="center")
    ax.set(xlim=(200, 5000), ylim=(0.6, 3.4), xlabel=r"$\ell$",
           title="Prediction for the real survey field")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Projection shear over SPT's 1500d footprint "
                 "(EE, Sanson–Flamsteed vs Lambert azimuthal)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "footprint_shear.png")
    print(f"\nwrote {OUT / 'footprint_shear.png'}")


if __name__ == "__main__":
    main()
