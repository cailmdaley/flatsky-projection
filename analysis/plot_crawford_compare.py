"""Predict Tom Crawford's 2026 proj0/proj5 measurement from projection geometry.

This is the test that matters. Everything else in the revival is validation on
2021 simulations we also made; this is the model meeting somebody else's data,
taken five years later on a different footprint with a different pipeline.

The setup has no free parameters and no fitting. Inputs: the 1500d field
geometry (RA +-50 deg, dec -70..-42, ZEA centred at 0, -57.5), and Tom's own
measured proj5 bandpowers standing in for the sky spectrum -- legitimate
because the model puts ZEA's own bias well under a percent, so his proj5 is the
sky as seen through his beam, pixel window and binning, all of which then
cancel in the ratio. Output: his proj0.

What comes out
--------------
The model reproduces the size of the excess, the fact that EE is affected more
than TT, and the turnover at the top of his range -- all three from spectral
slope and sky geometry alone. It sits about 20% low at the peak (23% for EE, 19%
for TT), which is the right sign and roughly the right size for the
reprojection aliasing Caterina Umilta measured separately and which this model
does not contain.

Where it does not work
----------------------
His cut-corner and iso-shrink variants drop much further than a change in shear
can explain (see the third panel). Either those masks are far more aggressive
than the shapes guessed here, or something beyond shear responds to mask
surgery. Resolving it needs his actual masks; this script states the gap rather
than tuning a shape until it closes.

One plotting artifact to know about: below l ~ 2500 the model curve oscillates
harder than the data. That is the input spectrum, not the physics. His
bandpowers are 100 wide, the model stretches them by up to 1.5 in l, and a
peak stretched onto a trough of a coarsely binned spectrum exaggerates the
swing. The high-l regime, where the whole question lives, is unaffected.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawford as C
import footprints as F
import geometry as G

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

ELL_FINE = np.arange(50.0, 6001.0, 20.0)
DEC_MID = -56.0          # centre of the dec range, for the variant shapes
RA_HALF, DEC_HALF = 50.0, 14.0

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def predict(mask, cl_proj5):
    """Predicted proj0/proj5, binned onto Tom's bandpowers, plus mean shear."""
    ell_th, cl_th, _ = C.as_input_spectrum(cl_proj5)
    got = {}
    for proj in ("ProjSFL", "ProjZEA"):
        s1, s2, w = F.mask_geometry(mask, proj, **C.FIELD_CENTER)
        got[proj] = G.predicted_spectrum(ELL_FINE, s1, s2, w, ell_th, cl_th)
    s1, s2, w = F.mask_geometry(mask, "ProjSFL", **C.FIELD_CENTER)
    ratio = C.bin_to_his_bandpowers(ELL_FINE, got["ProjSFL"] / got["ProjZEA"])
    return ratio, float(np.average(s1 / s2, weights=w)), F.f_sky(mask) * 41253


def superellipse(power=None, shrink=None):
    """Mask family: power -> rounded corners, shrink -> isotropic contraction.

    Guesses at what Tom's 'cut corner' and 'iso shrink' variants are. They are
    labelled as guesses everywhere they appear.
    """
    import healpy as hp
    nside = F.NSIDE
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    ra = np.where(ra > 180.0, ra - 360.0, ra)
    u, v = np.abs(ra) / RA_HALF, np.abs(dec - DEC_MID) / DEC_HALF
    if power is not None:
        keep = u ** power + v ** power <= 1.0
    else:
        keep = (u <= shrink) & (v <= shrink)
    m = np.zeros(len(ra))
    m[keep] = 1.0
    return F.apodize(m)


def main():
    full = F.rect_mask(**C.FIELD_1500D)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # ---- panels 1 and 2: EE and TT on the full footprint -------------------
    for ax, spec, colour in zip(axes[:2], ("ee", "tt"), ("crimson", "steelblue")):
        p0, p5 = C.load(spec)
        pred, qbar, area = predict(full, p5)
        meas = p0 / p5
        sel = (C.ELL > 300) & (C.ELL < 5800)
        ax.plot(C.ELL[sel], meas[sel], "o", ms=4, color=colour,
                label="Crawford 2026, measured")
        ax.plot(C.ELL[sel], pred[sel], "-", lw=2, color="k",
                label="shear model, no free parameters")
        ax.axhline(1.0, color="0.3", lw=0.8)
        ax.set(xlim=(300, 5800), ylim=(0.7, 2.8), xlabel=r"$\ell$",
               title=f"{spec.upper()}")
        ax.legend(fontsize=8, loc="upper left")
        peak = np.nanmax(meas[sel]) / np.nanmax(pred[sel])
        ax.text(0.97, 0.05, f"model low by {100*(peak-1):.0f}\\%\nat the peak",
                transform=ax.transAxes, ha="right", fontsize=8, color="0.35")
    axes[0].set_ylabel(r"$C_\ell(\mathrm{proj0})\,/\,C_\ell(\mathrm{proj5})$")

    # ---- panel 3: the mask variants, where the model does not close --------
    ax = axes[2]
    p0, p5 = C.load("tt")
    scan = []
    for power in (4, 3, 2):
        scan.append(predict(superellipse(power=power), p5))
    for shrink in (0.9, 0.8, 0.7, 0.6, 0.5):
        scan.append(predict(superellipse(shrink=shrink), p5))
    scan.append(predict(full, p5))
    areas = np.array([s[2] for s in scan])
    vals = np.array([s[0][40] for s in scan])
    order = np.argsort(areas)
    ax.plot(areas[order], vals[order], "o-", color="k", ms=4,
            label="shear model vs. mask size")

    styles = {"": ("full mask", "crimson"),
              "_cutcorner": ("his cut-corner", "darkorange"),
              "_iso_shrink": ("his iso-shrink", "seagreen")}
    for variant, (label, colour) in styles.items():
        a, b = C.load("tt", variant)
        ax.axhline((a / b)[40], color=colour, lw=1.6, ls="--",
                   label=f"{label}, measured")
    ax.set(xlabel="mask area (deg$^2$)", ylim=(1.0, 2.2),
           ylabel=r"$C_\ell(\mathrm{proj0})/C_\ell(\mathrm{proj5})$ at $\ell=4050$",
           title="TT variants: the model does not close")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("A zero-parameter geometry model against Crawford's 1500d "
                 "proj0/proj5 measurement", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "crawford_compare.png")
    print(f"wrote {OUT / 'crawford_compare.png'}")

    # ---- the numbers, for the record ---------------------------------------
    for spec in ("ee", "tt"):
        p0, p5 = C.load(spec)
        pred, qbar, area = predict(full, p5)
        meas = p0 / p5
        print(f"\n{spec.upper()}  (footprint {area:.0f} deg^2, mean shear {qbar:.3f})")
        print(f"{'ell':>6} {'measured':>9} {'predicted':>10} {'ratio':>7}")
        for L in (1050, 2050, 3050, 4050, 5050):
            i = int(L // 100)
            print(f"{C.ELL[i]:6.0f} {meas[i]:9.2f} {pred[i]:10.2f} "
                  f"{meas[i]/pred[i]:7.2f}")


if __name__ == "__main__":
    main()
