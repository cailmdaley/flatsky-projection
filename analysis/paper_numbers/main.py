"""The `numbers` macros: the leakage law over the sweep, the 1500d field, and the theory they rest on.

Named main.py because a module called `numbers` shadows the standard library.

  the scalar expansion   nEEtail, coefEEtail: the mean slope n of unlensed EE
                         over 3000 < l < 4500 and n(n+2)/4; gsqSFLfield and
                         liftEEtailField: SFL's <g^2> on the field and the
                         lift of the damping tail it implies at leading order.
  the run                lmaxrun, fieldfsky (the field mask's binary sky
                         fraction), sweepfskylo/sweepfskyhi (the smallest and
                         largest sweep rectangle's), cleanlo/cleanhi (the band where the model
                         is judged), hannwidth (the Hann window's full width).
  the law                fitcoef, fitrms, fitworst, fitoffset, decades: the
                         coefficient of BB/EE = k <g^2> fitted over every
                         sheared (footprint, projection) pair of the sweep,
                         its scatter, and the span in <g^2>; bbEePiresTAN, the
                         law's BB/EE for a 10 deg gnomonic square, in closed form.
  band by band           band<SFL|BICEP>ratio: measured over forward-modelled
                         BB/EE in the narrow bands, mean and spread, where the
                         measurement is clear of the STG floor.
  the 1500d field        tfdratio (SFL's 2<g^2> over BICEP's), tfdratiomeas
                         (the measured ratio of their floor-subtracted
                         leakage); EEbox* and EEswing* (measured EE over the
                         input, Hann-averaged, and its swing near the first
                         peaks); floorField, bbdecadesField; lensedBBEE,
                         tfdSFLoverlens, ZEAoverlens (leakage against lensed
                         BB/EE at l = 1000).
  power conservation     powerconserveAll: the worst |(EE + BB)_QU / EE_EB - 1|
                         over every footprint and projection.
  the flattening angle   flattenSFLfield: how much spt3g's flatten_pol angle
                         would add to SFL's modelled leakage on the field over
                         the polar angle, from the field mask's moments under
                         both options; a property of the two options, the
                         same whichever the run used.

Run:  bash env.sh python3 analysis/paper_numbers/main.py
          --bandpowers results/fiducial/bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra results/fiducial/theory_spectra
          --shear-moments results/fiducial/shear_moments
          --healpix-masks results/fiducial/healpix_masks
          --field-spectra <dir> --out <dir>
Out:  <out>/numbers.tex and <out>/numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import field_spectra as FS  # noqa: E402
import shear_leakage as SL  # noqa: E402
import shear_moments as SM  # noqa: E402
import spectra as S  # noqa: E402
import theory as T  # noqa: E402
from macros import WORDS, Macros, f, pct, sci  # noqa: E402

# Where the forward model is judged cleanly: above the low-l mode-coupling
# regime and below the l at which near-Nyquist aliasing and the STG floor
# reach the sheared measurements.
L_CLEAN = (1000, 4000)
# Measured EE is quoted Hann-averaged: a single dl = 50 band resolves the
# acoustic oscillation of a ratio to the input, and a window of half width
# L_HANN nulls a period of L_HANN, one acoustic spacing.
L_HANN = 300.0
L_LENSED = 1000                # the multipole of the lensed BB/EE reference


def hann(y, ell, centre):
    d = ell - centre
    s = np.abs(d) < L_HANN
    w = np.cos(np.pi * d[s] / (2 * L_HANN))**2
    return (w * y[s]).sum() / w.sum()


def flattening_cost(healpix_masks, th, ell, band_limit):
    """SFL's modelled BB/EE on the field under spt3g's angle over the polar angle, minus one.

    The mean over L_CLEAN of the ratio of the two forward models, each from the
    field mask's moments under its own flattening residual.
    """
    nside, pix, mask = SM.read_mask(healpix_masks / f"mask_{S.FIELD}.g3.gz")
    j = S.PROJS.index("ProjSFL")
    ratio = []
    for flattening in ("spt3g", "polar"):
        mom = SM.footprint_moments(nside, pix, mask, flattening)
        bb, ee = SL.predicted_spectra({k: v[None] for k, v in mom.items()}, 0, j, ell,
                                      th["ell"].astype(float), th["unlensed_EE"],
                                      band_limit=band_limit)
        ratio.append(bb / ee)
    clean = (ell > L_CLEAN[0]) & (ell < L_CLEAN[1])
    return float((ratio[0] / ratio[1])[clean].mean() - 1)


def macros(run, healpix_masks, fd):
    M = Macros()
    P = {p: j for j, p in enumerate(S.PROJS)}
    th = T.load(run.theory_spectra)
    ell_th = th["ell"].astype(float)
    ell = run.ell

    # ---- the scalar expansion and the run --------------------------------
    # For an equal-area projection and C ~ l^n the direction average lifts
    # power by n(n+2)/4 <g^2>. Unlensed EE oscillates too strongly for a
    # local slope to mean much, so n is the mean slope on the damping tail,
    # a least-squares fit of ln C on ln l over 3000 < l < 4500.
    tail = (ell_th > 3000) & (ell_th < 4500)
    n_tail = np.polyfit(np.log(ell_th[tail]), np.log(th["unlensed_EE"][tail]), 1)[0]
    coef = n_tail * (n_tail + 2) / 4
    g2_field = fd["pipeline_g2"]
    M.add("nEEtail", f"${n_tail:.1f}$")
    M.add("coefEEtail", f"{coef:.0f}")
    M.add("gsqSFLfield", f(g2_field[P["ProjSFL"]]))
    M.add("liftEEtailField", pct(coef * g2_field[P["ProjSFL"]], 0))
    M.add("lmaxrun", f"{run.lmax}")
    idx = json.loads((healpix_masks / f"index_{S.FIELD}.json").read_text())
    M.add("fieldfsky", pct(idx["masks"][0]["fsky_binary"], 1))
    sweep = sorted(
        json.loads(p.read_text())["masks"][0]["fsky_binary"]
        for p in healpix_masks.glob("index_rect_*.json"))
    M.add("sweepfskylo", pct(sweep[0], 2))
    M.add("sweepfskyhi", pct(sweep[-1], 1))
    M.add("cleanlo", f"{L_CLEAN[0]}")
    M.add("cleanhi", f"{L_CLEAN[1]}")
    M.add("hannwidth", f"{2 * L_HANN:.0f}")

    # ---- the law over the sweep -------------------------------------------
    mom = run.moments()
    g2 = mom["g2"]
    meas = SL.measured_ratios(run)
    band0 = meas[0]
    keep = SL.law_mask(band0)
    k, rms, worst, _ = SL.fit_constant(g2[keep], band0[keep])
    M.add("fitcoef", f(k, 4))
    M.add("fitrms", pct(rms))
    M.add("fitworst", pct(worst))
    M.add("fitoffset", pct(k / 2 - 1))
    # Spelled as a word: the prose reads "across four decades in <g^2>".
    M.add("decades", WORDS[int(round(np.log10(g2[keep].max() / g2[keep].min())))])

    # ---- a small gnomonic patch, in closed form ------------------------------
    # The gnomonic projection stretches radially by sec^2(theta) and
    # tangentially by sec(theta), so its shear is g = tan^2(theta/2). Averaged
    # over a 10 deg square on the tangent plane, as used for weak-lensing mass
    # maps, the law gives BB/EE = 2<g^2>.
    half = np.radians(5.0)
    x = np.linspace(-half, half, 2001)
    theta = np.arctan(np.hypot(*np.meshgrid(x, x)))
    M.add("bbEePiresTAN", sci(2 * np.mean(np.tan(theta / 2) ** 4)))

    # ---- band by band ------------------------------------------------------
    # The narrow bands up to the clean edge (BANDS[0] is their union), every
    # pair that tests the law, and only where the measurement is more than
    # 20 times STG's floor in that band and footprint.
    pred = SL.predicted_ratios(run, mom)
    floor = meas[:, :, P[SL.FLOOR_PROJ]]
    tested = np.array([b > 0 and hi <= L_CLEAN[1] for b, (lo, hi) in enumerate(SL.BANDS)])
    clean = (tested[:, None, None] & keep[None] & np.isfinite(pred)
             & (meas > 20 * floor[:, :, None]))
    for p in ("ProjSFL", "ProjBICEP"):
        j = P[p]
        r = (meas[..., j] / pred[..., j])[clean[..., j]]
        M.add(f"band{S.PROJ_LABEL[p]}ratio", rf"{r.mean():.4f} \pm {r.std():.4f}")

    # ---- the 1500d field ---------------------------------------------------
    lab = {S.PROJ_LABEL[p]: j for p, j in P.items()}
    M.add("tfdratio", f"{g2_field[lab['SFL']] / g2_field[lab['BICEP']]:.1f}")
    M.add("tfdratiomeas", f"{fd['sfl_bicep']:.2f}")
    lensed = float(th["lensed_BB"][L_LENSED] / th["lensed_EE"][L_LENSED])
    M.add("lensedBBEE", f"${sci(lensed)}$")

    cl, ee, bb = fd["cl_ee"], fd["ee_mean"], fd["bb_mean"]
    for label, centres in (("SFL", (3000., 4000.)), ("STG", (3000., 4000.)), ("BICEP", (4000.,))):
        for centre in centres:
            word = {3000.: "threeK", 4000.: "fourK"}[centre]
            M.add(f"EEbox{label}{word}",
                  f(hann(ee[lab[label]], ell, centre) / hann(cl, ell, centre), 2))
    swing = (ell > 850) & (ell < 1450)
    ratio = ee[lab["SFL"]] / cl
    M.add("EEswingSFLlo", f(ratio[swing].min(), 2))
    M.add("EEswingSFLhi", f(ratio[swing].max(), 2))
    centres = np.arange(800., 4500., 50.)
    zea = [hann(ee[lab["ZEA"]], ell, c) / hann(cl, ell, c) for c in centres]
    M.add("EEboxZEAdev", pct(np.max(np.abs(np.array(zea) - 1)), 0))

    def band(y, lo, hi):
        return y[(ell > lo) & (ell < hi)].mean()
    stg = lab["STG"]
    M.add("floorField", f"${sci(band(bb[stg], *SL.BANDS[0]) / band(ee[stg], *SL.BANDS[0]))}$")
    low = (ell > 500) & (ell < 3000)
    levels = [bb[j][low].mean() for j in lab.values()]
    M.add("bbdecadesField", WORDS[int(round(np.log10(max(levels) / min(levels))))])
    # Both leaks are measured BB/EE over the same band, against lensed BB/EE.
    def leak(name):
        return band(bb[lab[name]], *SL.BANDS[0]) / band(ee[lab[name]], *SL.BANDS[0])
    M.add("tfdSFLoverlens", f"{leak('SFL') / lensed:.0f}")
    M.add("ZEAoverlens", pct(leak("ZEA") / lensed, 0))

    # ---- power conservation ------------------------------------------------
    # The shear only rotates E into B, so (EE + BB) on the QU leg equals EE on
    # the E/B leg. Per (footprint, projection), seed-mean spectra, the per-bin
    # ratio averaged over 500 < l < 4800; the worst deviation. A band mean:
    # single bins deviate by up to ~1e-3.
    wide = (ell > 500) & (ell < 4800)
    total = (run.load("EE").mean(2) + run.load("BB").mean(2)) / run.load("EE", leg=S.EB_LEG).mean(2)
    M.add("powerconserveAll", f"${sci(np.abs(total[..., wide].mean(-1) - 1).max())}$")

    # ---- the flattening angle ----------------------------------------------
    M.add("flattenSFLfield", pct(flattening_cost(healpix_masks, th, ell, run.lmax), 1))
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "bandpowers", "simulation_windows", "theory_spectra", "shear_moments",
                 "healpix_masks", "field_spectra")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run = S.Run.from_args(args)
    macros(run, args.healpix_masks, FS.load(args.field_spectra)).write(args.out, "numbers")


if __name__ == "__main__":
    main()
