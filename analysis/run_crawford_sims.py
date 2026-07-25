"""Rerun Crawford's simulation and flip the one switch the model does not model.

The residual this closes
------------------------
The zero-parameter geometry model in `crawford_model` predicts Crawford's
full-mask proj0/proj5 ratio to 0.6-3% out to l = 4050, in both TT and EE. One
thing survives: at l = 5050 all six of his configurations sit 7-8% *above* the
model, including the full-mask control. Because the excess is common to every
mask it is not about masks, and because it is common to TT and EE it is not
about the spectrum's shape either. What is left is his map-making.

The asymmetry that makes it a ratio bias
----------------------------------------
His script draws a Gaussian field directly on the ZEA grid, calls that the
proj5 map, and reprojects it once to SFL with `interp=False, rebin=1`. So the
denominator of his ratio is the field as drawn -- never resampled -- while the
numerator is the field resampled by nearest neighbour. Any power resampling
adds lands only in the numerator, and the ratio comes out high.

Nearest-neighbour resampling replaces f(x) by f(x + d) with d a sub-pixel
displacement that varies pixel to pixel. Fourier mode by Fourier mode, the
coherent part survives with amplitude <exp(i k.d)> and the rest of the variance
is scattered broadband. The scattered piece is a *tiny fraction of the total*
but it is spread over the whole Nyquist square, so at l ~ 5000 -- where the
input spectrum has fallen four orders of magnitude and his band limit at
l = 7999 is about to cut it off entirely -- a small additive term is a large
fractional excess. That is the shape of the residual: 0.990 at l = 3050,
1.026 at 4050, 1.071 at 5050.

What this script does
---------------------
It reproduces his map-making exactly -- same grid, same resolution, same
`camb2cl` input, same band limit, same `cmb_flatsky` call -- for one
realization, then reprojects that one field to SFL several times with different
resampling settings and takes spectra his way, with `cl_flatsky` through the
full mask rebuilt by `crawford_masks`. The proj5 spectrum is measured once,
because the proj5 map is the same object in every case.

One realization is enough, and it is also all he has. His pickles are shape
(4, 60) with rows 1-3 left at zero, which is the signature of the single-config
ancestor script `proj0_distortion_check_cmbtt_20jul26.py`: it allocates four
realizations and ends the loop with a deliberate `print(notavariable)`.
(`crawford.load` averages over the four rows, but the same factor of 4 divides
numerator and denominator, so the ratio it returns is exact.) The eight-way
loop in the 24 July script writes shape (2, 2, 2, 4, 60) and has that break
commented out, so it is not what produced the numbers we compare against.

Which pixel size he used
------------------------
This matters more than anything else here, because the resampling term scales
steeply with pixel size -- roughly as p^4 in the small-lp limit, measured at
p^3 between 1' and 0.5' where the l = 7999 band limit bites. The 20 July script
now reads `ngrid = 20000, reso_arcmin = 0.25`, but its own backup copy, saved
midway through 24 July, reads `10000, 0.5`. His full-mask pickles are timestamped
before that edit and his `iso_shrink` pickles after it, so the two halves of his
suite may not share a pixel size. `--ngrid`/`--res-arcmin` therefore have to be
scanned rather than assumed; both configurations are worth running.

What it found
-------------
The hypothesis holds, and the residual is his resampling.

Nearest neighbour puts a high-l excess above the geometry model at every pixel
size, rising steeply with l exactly as his residual does. Bilinear removes it
and overshoots to the low side, because it applies a sinc^2 window where
nearest neighbour applies sinc. The model is straddled from both sides, which
is the signature that the difference is sampling and not geometry.

The excess is *additive*, not multiplicative. At 1 arcmin the difference
C_l(nn) - model is flat at 1.3-1.6e-18 across l = 3050-5550 while the spectrum
underneath it falls by an order of magnitude. That is scattered aliasing power,
and its level agrees with `scatter_level` -- a closed-form integral with no
free parameters -- to 15% at 1 arcmin.

The amplitude goes as p^2.9 in pixel size, measured over 1, 0.5 and 1/3 arcmin
and stable to +-0.1 in the exponent for l >= 4050. (The closed-form scattered
level goes as p^4 exactly; the extra factor of 1/p is that the aliased power
does not stay spread flat over a Nyquist square that grows as p shrinks.)
Extrapolated to the 0.25 arcmin of his ancestor script that law predicts his
own full-mask residual to 1-2% in the ratio: 1.009/1.026/1.060/1.124 against
his 1.026/1.029/1.071/1.146 at l = 4050/4550/5050/5550.

Not sample variance: three seeds at 1 arcmin give 4.362, 4.326, 4.390 for the
l = 5050 ratio over the model, a 0.7% spread against a factor-of-four effect.

A note on cost
--------------
His own 20000^2 grid does not fit. `cl_flatsky` materialises a dozen full-grid
arrays and peaks near 44 GB there, which is the per-user cgroup limit on
amundsen; the run is OOM-killed partway through. Reproducing his exact pixel
size needs a batch node, and the p^2.9 law is what stands in for it here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import crawford as C          # noqa: E402
import crawford_masks as CM   # noqa: E402
import crawford_model as M    # noqa: E402

from spt3g import core, maps                                    # noqa: E402
from spt3g.simulations import quick_flatsky_routines as qfr      # noqa: E402

ARCMIN = core.G3Units.arcmin
DEG = core.G3Units.deg

# `reproj_map`'s two sub-pixel knobs. "nn" is his; the rest are the controls.
# rebin=n averages n^2 samples inside each output pixel, which is sub-pixel
# averaging on top of whichever point-sampling rule `interp` selects.
RESAMPLINGS = {
    "nn":      dict(interp=False, rebin=1),
    "nn4":     dict(interp=False, rebin=4),
    "interp":  dict(interp=True,  rebin=1),
    "interp4": dict(interp=True,  rebin=4),
}

# Bandpower indices at l = 1050, 2050, 3050, 4050, 5050.
SHOW = [10, 20, 30, 40, 50]


def input_spectrum(spec):
    """His `camb2cl(8000)` dict, with the EE substitution he makes for "EE"."""
    cld = qfr.camb2cl(M.BAND_LIMIT + 1)
    if spec == "ee":
        cld["cl"]["TT"] = cld["cl"]["EE"].copy()
    return cld


def draw_proj5(cld, ngrid, res_arcmin, seed):
    """His proj5 map: a Gaussian field drawn straight onto the ZEA grid.

    He does not seed, so the realization differs from his; the ratio is a
    same-sky quantity and does not care. Seeding here makes reruns comparable.
    """
    arr = qfr.cmb_flatsky(cld, ngrid, res_arcmin * ARCMIN,
                          seed=seed, use_seed=True, T_only=True)
    m = maps.FlatSkyMap(
        np.ascontiguousarray(arr), res=res_arcmin * ARCMIN, weighted=False,
        alpha_center=0.0, delta_center=CM.DELTA_CENTER_PROJ5 * DEG,
        proj=maps.MapProjection.ProjZEA)
    del arr
    return m


def reproject(m5, ngrid, res_arcmin, interp, rebin):
    """His reprojection onto the SFL grid, with the sub-pixel rule exposed."""
    m0 = maps.FlatSkyMap(
        ngrid, ngrid, res=res_arcmin * ARCMIN, weighted=False,
        alpha_center=0.0, delta_center=CM.DELTA_CENTER_PROJ0 * DEG,
        proj=maps.MapProjection.ProjSFL)
    maps.reproj_map(m5, m0, interp=interp, rebin=rebin)
    return m0


def spectrum(m, mask, res_arcmin):
    """His estimator: `cl_flatsky` through the mask, on his 60 bandpowers."""
    out = qfr.cl_flatsky(np.asarray(m), res_arcmin, delta_ell=C.DELTA_L,
                         apod_mask=mask)
    return np.asarray(out["cl"]["TT"][:C.N_BINS], dtype=float)


def _axis_transfer(kind, theta, n_u=256):
    """<T> over sub-pixel offset u, one axis, for a mode of phase theta = k p.

    Sample points sit at 0 and p; the field is wanted at u p. Nearest
    neighbour returns one of the two samples, so its transfer is a pure phase
    e^{-i theta d} with d the signed distance to the pixel taken, and the
    average over u is sinc(theta/2). Bilinear returns the weighted sum, and
    averages to sinc^2(theta/2) -- twice the suppression, which is why turning
    interpolation on is not a neutral fix but an overcorrection.
    """
    u = (np.arange(n_u) + 0.5) / n_u
    th = np.asarray(theta)[..., None]
    if kind == "nn":
        d = np.where(u < 0.5, u, u - 1.0)
        return np.exp(-1j * th * d).mean(axis=-1)
    return (((1 - u) + u * np.exp(1j * th)) * np.exp(-1j * th * u)).mean(axis=-1)


def coherent_window(kind, ell, res_arcmin, n_psi=192):
    """Fraction of a mode's power that survives resampling coherently.

    |<T_x><T_y>|^2 with the two axes' sub-pixel offsets independent, averaged
    over the direction of l because the estimator bins in |l|. Whatever this
    does not retain is either scattered (nearest neighbour, which moves samples
    but never averages them, so total variance is conserved) or destroyed
    (bilinear, which averages neighbours and genuinely smooths).
    """
    p = np.radians(res_arcmin / 60.0)
    psi = (np.arange(n_psi) + 0.5) * (np.pi / 2 / n_psi)
    tx = _axis_transfer(kind, np.outer(ell, np.cos(psi)) * p)
    ty = _axis_transfer(kind, np.outer(ell, np.sin(psi)) * p)
    return (np.abs(tx * ty) ** 2).mean(axis=1)


def rebin_window(ell, res_arcmin, rebin, n_psi=192):
    """Extra coherent suppression from `rebin`: an n x n box average per pixel.

    Each of the n^2 sub-samples carries its own random sub-pixel offset, so the
    averaging factorizes: the coherent transfer picks up the Dirichlet kernel
    of the sub-sample positions on top of whichever point-sampling rule was
    used. Its incoherent part partly averages down too, which this does not
    model -- for the bilinear runs there is little to average down.
    """
    ell = np.asarray(ell, dtype=float)
    if rebin <= 1:
        return np.ones_like(ell)
    p = np.radians(res_arcmin / 60.0)
    off = (np.arange(rebin) + 0.5) / rebin - 0.5
    psi = (np.arange(n_psi) + 0.5) * (np.pi / 2 / n_psi)

    def axis(th):
        return np.exp(1j * th[..., None] * off).mean(axis=-1)

    dx = axis(np.outer(ell, np.cos(psi)) * p)
    dy = axis(np.outer(ell, np.sin(psi)) * p)
    return (np.abs(dx * dy) ** 2).mean(axis=1)


def scatter_level(cld, res_arcmin):
    """White C_l the coherent power lost to nearest neighbour lands at.

    Nearest-neighbour resampling conserves the map's variance exactly -- every
    output pixel is *some* input pixel -- so the power the window removes from
    each mode has to reappear somewhere. Spread flat over the Nyquist square of
    area (2 pi / p)^2, it is this constant. Zero free parameters.
    """
    p = np.radians(res_arcmin / 60.0)
    ell = np.asarray(cld["ell"], dtype=float)[1:]
    cl = np.asarray(cld["cl"]["TT"], dtype=float)[1:]
    lost = (1.0 - coherent_window("nn", ell, res_arcmin)) * cl
    return float(np.trapz(2 * np.pi * ell * lost, ell) / (2 * np.pi / p) ** 2)


def row(label, vals, fmt="{:9.3f}"):
    print(f"  {label:<18}" + "".join(fmt.format(vals[i]) for i in SHOW))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ngrid", type=int, default=10000,
                    help="his grid is 10000 (default)")
    ap.add_argument("--res-arcmin", type=float, default=0.5,
                    help="his resolution is 0.5 (default)")
    ap.add_argument("--spec", default="tt", choices=("tt", "ee"))
    ap.add_argument("--settings", default="nn,interp",
                    help="comma-separated keys of " + ",".join(RESAMPLINGS))
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--out", default=None, help="JSON summary path")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    bad = [s for s in settings if s not in RESAMPLINGS]
    if bad:
        ap.error(f"unknown resampling {bad}")

    ngrid, res = args.ngrid, args.res_arcmin
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    log(f"grid {ngrid}^2 at {res} arcmin, spec {args.spec}, "
        f"settings {settings}")

    m5arr, m0arr = CM.cached_arrays("full", ngrid, res)
    m5arr = m5arr.astype(float)
    m0arr = m0arr.astype(float)
    log(f"masks built; <w^2> proj5 {np.mean(m5arr**2):.5f} "
        f"proj0 {np.mean(m0arr**2):.5f}")

    cld = input_spectrum(args.spec)
    m5 = draw_proj5(cld, ngrid, res, args.seed)
    log("proj5 field drawn")

    cl5 = spectrum(m5, m5arr, res)
    del m5arr
    log("proj5 spectrum measured")

    cl0 = {}
    for name in settings:
        kw = RESAMPLINGS[name]
        m0 = reproject(m5, ngrid, res, **kw)
        log(f"reprojected ({name}: {kw})")
        cl0[name] = spectrum(m0, m0arr, res)
        del m0
        log(f"proj0 spectrum measured ({name})")

    del m5, m0arr

    # The model is a footprint average and does not depend on the sim grid, so
    # it stays on the cheap cached geometry regardless of what was simulated.
    _, model = M.predict("full", args.spec)
    _, his = M.measured("", args.spec)

    # Zero-parameter resampling model, layered on the geometry model:
    #   nearest neighbour   W_nn  * P_geom + C_scat   (variance conserved)
    #   bilinear            W_bil * P_geom            (variance removed)
    w = {"nn": coherent_window("nn", C.ELL, res),
         "interp": coherent_window("interp", C.ELL, res)}
    c_scat = scatter_level(cld, res)
    pred = {}
    for name, kw in RESAMPLINGS.items():
        kind = "nn" if not kw["interp"] else "interp"
        win = w[kind] * rebin_window(C.ELL, res, kw["rebin"])
        pred[name] = win * model + (c_scat / cl5 if kind == "nn" else 0.0)
    w_nn, w_bil = w["nn"], w["interp"]

    print(f"\n=== {args.spec.upper()}  full mask  {ngrid}^2 @ {res}' ===")
    print("  l                 " + "".join(f"{C.ELL[i]:9.0f}" for i in SHOW))
    row("geometry model", model)
    row("his meas", his)
    row("his/model", his / model)
    print()
    ratios = {}
    for name in settings:
        ratios[name] = cl0[name] / cl5
        row(f"sim {name}", ratios[name])
        row(f"sim {name}/model", ratios[name] / model)
    print()
    print("  resampling windows and the scattered term they imply")
    row("W_nn", w_nn)
    row("W_bilinear", w_bil)
    row("C_scat/C_proj5", c_scat / cl5, fmt="{:9.3f}")
    for name in settings:
        row(f"pred {name}", pred[name])
        row(f"sim/pred {name}", ratios[name] / pred[name])

    # The discriminant: is the extra power multiplicative or additive?
    if "nn" in cl0 and len(settings) > 1:
        base = [s for s in settings if s != "nn"][0]
        d = cl0["nn"] - cl0[base]
        print(f"\n  resampling term, C_l(nn) - C_l({base}); "
              f"flat-scatter prediction {c_scat:.3e}")
        row("dC_l", d, fmt="{:9.2e}")
        row("dC_l/C_scat", d / c_scat)
        row("dC/C(nn)", d / cl0["nn"])

    out = Path(args.out) if args.out else (
        CM.CACHE / f"sims_{args.spec}_n{ngrid}_r{res:g}.json")
    payload = dict(
        ngrid=ngrid, res_arcmin=res, spec=args.spec, seed=args.seed,
        ell=list(C.ELL), cl_proj5=list(cl5), model=list(model),
        his_measured=list(his), c_scat=c_scat,
        w_nn=list(w_nn), w_bilinear=list(w_bil),
        cl_proj0={k: list(v) for k, v in cl0.items()},
        settings={k: RESAMPLINGS[k] for k in settings},
    )
    out.write_text(json.dumps(payload, indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
