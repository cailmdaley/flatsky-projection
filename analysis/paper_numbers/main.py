"""Every macro of paper/numbers.tex: the footprint sweep, the 1500d field, and the threads that report into it.

Not yet a recipe: reads its inputs from fixed paths and writes paper/numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import shear_leakage as SL  # noqa: E402
import spectra as S  # noqa: E402
import theory as T  # noqa: E402
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
MASK = S.SWEEP.index(S.FIELD)  # the reference footprint, the 1500d field
L_WALL = 4800.0               # top of the plotted range, not a universal limit
# Where the forward model is judged cleanly: above the low-l mode-coupling
# regime and below the l at which near-Nyquist aliasing and the STG floor
# reach the sheared measurements.
L_CLEAN = (1000, 4000)
L_ALIAS = 4000.0              # EE aliasing quoted over L_ALIAS < l < L_WALL
L_RISE = 4250.0               # BICEP's BB/EE leaves the model above this
BAND_LIMIT = S.LMAX

OUT = ROOT / "paper" / "numbers.tex"


def main():
    M = Macros()
    P = {p: j for j, p in enumerate(S.PROJS)}
    short = {p: S.PROJ_LABEL[p] for p in S.PROJS}
    jstg = P[SL.FLOOR_PROJ]

    mom = SL.load_moments()
    g2 = mom["g2"]
    ee = S.load("EE")
    bb = S.load("BB")
    ell = S.ELL
    cl_ee = S.theory("unlensed_EE")
    th = T.load()
    ell_th, cl_th = th["ell"].astype(float), th["unlensed_EE"]

    # Forward model on the reference footprint, absolute C_l.
    m_bb = np.empty((len(S.PROJS), len(ell)))
    m_ee = np.empty_like(m_bb)
    for j in range(len(S.PROJS)):
        m_bb[j], m_ee[j] = SL.predicted_spectra(mom, MASK, j, ell, ell_th, cl_th,
                                                band_limit=BAND_LIMIT)

    # ---- 4.1 the scalar channel -------------------------------------------
    ratio = ee[MASK].mean(1) / cl_ee              # (proj, nbins), measured/input
    mratio = m_ee / cl_ee                         # model/input
    def at(y, L):
        return float(y[np.argmin(abs(ell - L))])

    # Only what the prose quotes. Measurement-over-model is quoted as a range
    # across projections, not per projection, so no per-projection macro for it.
    for p in ("ProjSFL", "ProjBICEP"):
        M.add(f"EEin{short[p]}oneK", f(at(ratio[P[p]], 1000)))
    for p in ("ProjSFL", "ProjSTG", "ProjBICEP"):
        M.add(f"EEin{short[p]}threeK", f(at(ratio[P[p]], 3000)))
    # ZEA is quoted as a deviation from unity across the band, not at points.
    for p in ("ProjSFL", "ProjBICEP"):
        M.add(f"EEin{short[p]}fourfive", f(at(ratio[P[p]], 4500), 2))

    # ---- 2.2 the scalar expansion, on the spectrum the results use --------
    # For an equal-area projection and C ~ l^n the direction average lifts
    # power by n(n+2)/4 <g^2>. Unlensed EE oscillates too strongly for a
    # local slope to mean much, so `nEEtail` is the mean slope on the damping
    # tail, a least-squares fit of ln C on ln l over 3000 < l < 4500;
    # `coefEEtail` the coefficient it implies; `gsqSFLfive` SFL's <g^2> on
    # the reference footprint; `liftEEtail` their product, the expansion's lift.
    tail = (ell_th > 3000) & (ell_th < 4500)
    n_tail = np.polyfit(np.log(ell_th[tail]), np.log(cl_th[tail]), 1)[0]
    coef_tail = n_tail * (n_tail + 2) / 4
    g2_sfl = float(g2[MASK][P["ProjSFL"]])
    M.add("nEEtail", f"${n_tail:.1f}$")
    M.add("coefEEtail", f"{coef_tail:.0f}")
    M.add("gsqSFLfive", f(g2_sfl))
    M.add("liftEEtail", pct(coef_tail * g2_sfl, 0))

    band = (ell > 500) & (ell < L_WALL)
    M.add("EEinZEAdev", pct(np.max(np.abs(ratio[P["ProjZEA"]][band] - 1)), 0))
    r_one = [at(ratio[j] / mratio[j], 1000) for j in P.values()]
    r_three = [at(ratio[j] / mratio[j], 3000) for j in P.values()]
    M.add("EEmodelRangeOneK", rng(min(r_one), max(r_one), 3))
    M.add("EEmodelRangeThreeK", rng(min(r_three), max(r_three), 3))

    # ---- the pipeline floor, measured on the conformal projection ---------
    meas = SL.measured_ratios()
    floor = meas[:, :, jstg]                      # (band, mask)
    M.add("floorFive", f"${sci(floor[0, MASK])}$")
    M.add("floorFiveTop", f"${sci(floor[-1, MASK])}$")
    M.add("floorTopBand", f"{SL.BANDS[-1][0]} < \\ell < {SL.BANDS[-1][1]}")

    # ---- 4.2 the leakage channel ------------------------------------------
    sel = (ell > 500) & (ell < 3000)
    for p, j in P.items():
        if p == SL.FLOOR_PROJ:
            continue
        mm = bb[MASK, j].mean(0)[sel].mean()
        M.add(f"BBmodel{short[p]}", f(mm / m_bb[j][sel].mean()))
    orders = np.log10(bb[MASK, jstg].mean(0)[sel].mean() / m_bb[jstg][sel].mean())
    M.add("STGmodelorders", f"{orders:.0f}")
    # How far the projections' BB actually spreads on this footprint, rather than
    # a remembered figure.
    lvl = [bb[MASK, j].mean(0)[sel].mean() for j in P.values()]
    M.add("bbdecades", _WORDS[int(round(np.log10(max(lvl) / min(lvl))))])

    # BB/EE against the model over the full usable range, both channels.
    wide = (ell > 500) & (ell < L_WALL)
    wr = []
    for p, j in P.items():
        if p == SL.FLOOR_PROJ:
            continue
        r = ((bb[MASK, j].mean(0) / ee[MASK, j].mean(0))[wide]
             / (m_bb[j] / m_ee[j])[wide])
        wr.append(r.mean())
    M.add("wideBBrange", rng(min(wr), max(wr), 3))
    we = [float((ratio[j] / mratio[j])[wide].mean()) for j in P.values()]
    M.add("wideEErange", rng(min(we), max(we), 3))
    M.add("lwall", f"{L_WALL:.0f}")

    # The clean statement: floor-subtracted (BB/EE)_meas / (BB/EE)_model, mean
    # over bins, on L_CLEAN, where neither the STG floor nor near-Nyquist
    # aliasing reaches the sheared measurements.
    floor_bin = bb[MASK, jstg].mean(0) / ee[MASK, jstg].mean(0)
    cband = (ell > L_CLEAN[0]) & (ell < L_CLEAN[1])
    cr = []
    for p, j in P.items():
        if p == SL.FLOOR_PROJ:
            continue
        r = ((bb[MASK, j].mean(0) / ee[MASK, j].mean(0) - floor_bin)
             / (m_bb[j] / m_ee[j]))
        cr.append(float(r[cband].mean()))
    M.add("cleanBBrange", rng(min(cr), max(cr), 3))

    # What FlattenPol's residual rotation is worth in the model: BB/EE with
    # Delta over BB/EE with the polar angle, mean over L_CLEAN, reference footprint. SFL
    # carries nearly all of it, ZEA a sliver.
    nod = dict(mom, cell_sin2_2delta=np.zeros_like(mom["cell_sin2_2delta"]))
    for p, nd in (("ProjSFL", 1), ("ProjZEA", 2)):
        j = P[p]
        b0, e0 = SL.predicted_spectra(nod, MASK, j, ell, ell_th, cl_th,
                                      band_limit=BAND_LIMIT)
        d = float(((m_bb[j] / m_ee[j]) / (b0 / e0))[cband].mean() - 1)
        M.add(f"flatten{short[p]}", pct(d, nd))

    M.add("cleanlo", f"{L_CLEAN[0]}")
    M.add("cleanhi", f"{L_CLEAN[1]}")

    # The near-Nyquist rise: floor-subtracted BB/EE over the model in the last
    # bin below the wall. BICEP climbs above L_RISE; SFL, with as much excess
    # EE, does not, so SFL's is quoted too.
    top = int(np.flatnonzero(S.BIN_HI <= L_WALL)[-1])
    for p in ("ProjBICEP", "ProjSFL"):
        j = P[p]
        r = ((bb[MASK, j].mean(0) / ee[MASK, j].mean(0) - floor_bin)
             / (m_bb[j] / m_ee[j]))
        M.add(f"rise{short[p]}top", f(r[top]))
    M.add("risetopbin", f"{S.BIN_LO[top]} < \\ell < {S.BIN_HI[top]}")
    M.add("riselo", f"{L_RISE:.0f}")

    # The resampling term is not common to every projection near Nyquist: EE
    # over the model, relative to STG's, for SFL and BICEP, which distort the
    # grid the most and so carry the most sky power past the 2 arcmin Nyquist
    # (BICEP by compressing x, SFL by lengthening k_y through its shear).
    aband = (ell > L_ALIAS) & (ell < L_WALL)
    stg_t = ee[MASK, jstg].mean(0) / m_ee[jstg]
    for p in ("ProjSFL", "ProjBICEP"):
        j = P[p]
        rel = (ee[MASK, j].mean(0) / m_ee[j]) / stg_t
        M.add(f"EEalias{short[p]}", pct(float(rel[aband].mean()) - 1, 1))
    M.add("aliaslo", f"{L_ALIAS:.0f}")
    M.add("lmaxrun", f"{S.LMAX}")

    # Power conservation between the legs: the shear only rotates E into B,
    # so (EE + BB) on the QU leg equals EE on the E/B leg. Per (mask,
    # projection), seed-mean spectra, the per-bin ratio averaged over
    # 500 < l < L_WALL; the worst deviation over every pair but `CLIPPED`.
    # A band mean: single bins deviate by up to ~1e-3.
    ee_eb = S.load("EE", leg=S.EB_LEG).mean(2)
    tot = (ee.mean(2) + bb.mean(2)) / ee_eb
    dev = np.abs(tot[:, :, wide].mean(-1) - 1)
    for i, p in SL.CLIPPED:
        dev[i, P[p]] = np.nan
    M.add("powerconserve", f"${sci(np.nanmax(dev))}$")
    # The pixel Nyquist multipole, from the pixel scale rather than remembered.
    M.add("lnyquist", f"{np.pi / np.deg2rad(S.RES_ARCMIN / 60.0):.0f}")

    # ---- the fit ----------------------------------------------------------
    band0 = meas[0]
    keep = SL.law_mask(band0)
    k, rms, worst, n = SL.fit_constant(g2[keep], band0[keep])
    M.add("fitcoef", f(k, 4))
    M.add("fitrms", pct(rms))
    M.add("fitworst", pct(worst))
    M.add("numpairs", f"{n}")
    M.add("numpairstotal", f"{(len(S.PROJS) - 1) * len(S.SWEEP)}")
    M.add("fitoffset", pct(k / 2 - 1))
    # Spelled as a word, like \bbdecades: all three places the paper quotes it
    # are prose ("across four decades in <g^2>"), where a numeral reads oddly.
    M.add("decades", _WORDS[int(round(np.log10(g2[keep].max() / g2[keep].min())))])
    slope, inter = np.polyfit(np.log(g2[keep]), np.log(band0[keep]), 1)
    M.add("freepowerlaw", rf"{np.exp(inter):.3f}\,\gsq^{{{slope:.4f}}}")
    # Compare the three candidate shear reductions using their own best fits.
    for key, macro in (("g2", "rmsGsq"), ("q2", "rmsRatio"),
                       ("lnq2", "rmsLog")):
        _, reduction_rms, _, _ = SL.fit_constant(mom[key][keep], band0[keep])
        M.add(macro, pct(reduction_rms))

    # Same fit with the pipeline floor removed, to show it does not matter.
    sub = band0 - floor[0][:, None]
    ks, _, _, _ = SL.fit_constant(g2[keep], sub[keep])
    M.add("fitcoefsub", f(ks, 4))

    # ---- 4.3 band by band --------------------------------------------------
    pred = SL.predicted_ratios(mom, band_limit=BAND_LIMIT)
    # The narrow bands, up to the common leakage edge; BANDS[0] is their union.
    tested = np.array([b > 0 and hi <= L_CLEAN[1] for b, (lo, hi) in enumerate(SL.BANDS)])
    clean = (tested[:, None, None] & keep[None, :, :] & np.isfinite(meas)
             & np.isfinite(pred) & (meas > 20 * floor[:, :, None]))
    r = meas[clean] / pred[clean]
    M.add("bandbybandratio", rf"{r.mean():.4f} \pm {r.std():.4f}")
    M.add("numbands", f"{int(clean.sum())}")
    M.add("bandbybandworst", pct(float(np.max(np.abs(r - 1))), 1))
    M.add("numbandbands", f"{int(tested.sum())}")
    M.add("bandlo", f"{SL.BANDS[1][0]}")
    M.add("bandhi", f"{L_CLEAN[1]}")
    for p in ("ProjSFL", "ProjBICEP", "ProjZEA"):
        j = P[p]
        ratio = (meas[..., j] / pred[..., j])[clean[..., j]]
        M.add(f"band{short[p]}ratio", rf"{ratio.mean():.4f} \pm {ratio.std():.4f}")
        M.add(f"band{short[p]}worst", pct(float(np.max(np.abs(ratio - 1))), 1))
    # ZEA's excess over the model, in units of STG's floor, where the floor is
    # at least 5% of the model on the sweep.
    jz = P["ProjZEA"]
    measurable = keep[..., jz] & (floor > 0.05 * pred[..., jz])
    excess = (meas[..., jz] / pred[..., jz] - 1)[measurable] / (floor / pred[..., jz])[measurable]
    M.add("zeaFloorDisks", rng(excess.min(), excess.max(), 1))

    # The tilts on the reference footprint.
    tilt = meas[1:, MASK, :] / (2 * g2[MASK])
    for p in ("ProjSFL", "ProjBICEP"):
        j = P[p]
        M.add(f"tilt{short[p]}lo", f(tilt[0, j], 2))
        M.add(f"tilt{short[p]}hi", f(tilt[-1, j], 2))
    # ZEA raw, band by band -- the cleanest single test.
    zea = meas[1:, MASK, P["ProjZEA"]] / (2 * g2[MASK, P["ProjZEA"]])
    M.add("ZEAbands", ", ".join(f"{v:.3f}" for v in zea[:-1]))
    M.add("ZEAlastband", f(zea[-1], 3))
    zfloor = floor[1:, MASK] / meas[1:, MASK, P["ProjZEA"]]
    ok = np.flatnonzero(zfloor < 0.005)
    M.add("ZEAfloorclean", f"{SL.BANDS[1:][ok[-1]][1]}")

    # BICEP's magnification inside the reference footprint.
    M.add("magfactor", rf"{mom['s1_max'][MASK, P['ProjBICEP']]:.1f}\times")

    # ---- 4.4 the cylindrical near-degeneracy -------------------------------
    jS, jB = P["ProjSFL"], P["ProjBICEP"]
    last = len(S.SWEEP) - 1
    for name, i in (("small", 0), ("five", MASK), ("ten", last)):
        M.add(f"sflbicep{name}", pct(abs(g2[i, jB] / g2[i, jS] - 1), 1))
    M.add("clippedSFL", f(band0[last, jS] / (2 * g2[last, jS]), 2))
    M.add("clippedBICEP", f(band0[last, jB] / (2 * g2[last, jB]), 2))
    M.add("numclipped", f"{len(SL.CLIPPED)}")

    # ---- covariance: does the leaked B bin like a sky? ---------------------
    cov = _offdiag(ee, bb, ell)
    M.add("offdiagBB", pct(cov["bb_mean"], 1))
    M.add("offdiagEE", pct(cov["ee_mean"], 1))
    M.add("offdiagSTG", f(cov["stg_max"], 2))
    M.add("offdiagnoise", f(cov["noise"], 2))

    # The law's prediction on SPT's 1500d field, from its own moments. Only
    # the two the prose contrasts; the other two are printed by
    # shear_leakage.report.
    field = _field_1500d()
    for label in ("SFL", "BICEP"):
        M.add(f"tfd{label}", f"${sci(field[label])}$")
    M.add("tfdratio", f"{field['SFL'] / field['BICEP']:.1f}")

    # The same field simulated (`field_bandpowers`, 20 sims, QU leg; see
    # analysis/field_1500d.py): floor-subtracted band BB/EE over BAND, the
    # SFL/BICEP ratio of the per-sim values with its error, `tfdmeaslaw` the
    # SFL measurement over the leading-order law 2<g^2> on the pipeline mask,
    # and `tfdmeasmodel` the worst |meas/model - 1| over L_CLEAN among the
    # three leaking projections (STG is the floor).
    import field_1500d as FD
    fd = FD.load()
    lab = [S.PROJ_LABEL[p] for p in S.PROJS]
    for label in ("SFL", "BICEP"):
        M.add(f"tfd{label}meas", f"${sci(fd['meas_band'][lab.index(label)])}$")
    M.add("tfdratiomeas", f"{fd['sfl_bicep']:.2f}")
    M.add("tfdratiomeaserr", f"{fd['sfl_bicep_err']:.2f}")
    jsfl = lab.index("SFL")
    M.add("tfdmeaslaw", pct(fd["meas_band"][jsfl] / (2 * fd["pipeline_g2"][jsfl]) - 1, 0))
    leak = [j for j, l in enumerate(lab) if l != "STG"]
    M.add("tfdmeasmodel", pct(max(abs(fd["pipeline_clean"][j] - 1) for j in leak), 1))
    M.add("tfdnsims", f"{int(fd['n_sims'])}")



    # ---- the lensed reference -------------------------------------------
    # theory.load() stores raw C_l in uK^2 indexed directly by ell -- see
    # analysis/theory.py's TRAPS note -- so these ratios need no unit juggling.
    th = T.load()
    ell_ref = 1000
    cl_bb_lensed = float(th["lensed_BB"][ell_ref])
    lensed_bb_ee = cl_bb_lensed / float(th["lensed_EE"][ell_ref])
    M.add("lensedBBEE", f"${sci(lensed_bb_ee)}$")
    M.add("tfdSFLoverlens", f"{field['SFL'] / lensed_bb_ee:.0f}")

    # ---- the template applied to the simulations (`template_subtraction`) --
    # S_J[E] built per pixel from the grid metric and subtracted from the
    # polar-reflattened SFL and ZEA QU-leg maps, 20 sims, on the production
    # 1500d field mask (`reference`) and its separable taper (`field`). Noise-free: `...raw` / `...sub` BB/EE over L_CLEAN and
    # `...supp` their ratio. With white E noise of `templatenoise` on the
    # template's input (`...five`: the paper's nominal 5 uK-arcmin):
    # `...subnoise` (raw E) and `...subwiener` (Wiener-filtered E) at
    # l ~ 1000 (L_REF), `...cross` the l above which the
    # raw-E template adds more B than it removes; `...kernel` the measured BB
    # of S_J[unit E noise] over N^EE against the W^2-weighted <tan^2 2dpsi>
    # (`...tan`) and 2<g^2> (`...gg`), `...tanexcess` the fraction by which
    # the first exceeds the second; `...gain` the raw BB/EE at L_REF over the
    # noisy residual there, and `...lensfrac` the noise residual
    # kernel * N^EE as a fraction of the lensed BB in the same band.
    # `templatecorr`: the noise-free
    # map-level correlation of S_J[E] with the grid B on SFL's reference mask.
    import template_subtraction as TPL
    lv = TPL.NOISE_LEVELS[0]
    cl_bb_reference = float(th["lensed_BB"][TPL.L_REF[0]:TPL.L_REF[1]].mean())
    M.add("templatenoise", f"\\SI{{{lv:g}}}{{\\micro\\kelvin}}-arcmin")
    M.add("templatelref", f"${TPL.L_REF[0]}$--${TPL.L_REF[1]}$")
    for proj in ("ProjSFL", "ProjZEA"):
        ts = TPL.summary(TPL.run(proj))
        lab = S.PROJ_LABEL[proj]
        for m in ("reference", "field"):
            k = f"template{lab}{m}"
            M.add(f"{k}raw", f"${sci(ts[f'{m}_raw'])}$")
            M.add(f"{k}sub", f"${sci(ts[f'{m}_sub'])}$")
            M.add(f"{k}supp", f"${sci(ts[f'{m}_raw'] / ts[f'{m}_sub'])}$")
            if proj != "ProjSFL":
                continue
            M.add(f"{k}rawref", f"${sci(ts[f'{m}_raw_ref'])}$")
            for lvl, word in zip(TPL.NOISE_LEVELS, ("", "five")):
                M.add(f"{k}subnoise{word}", f"${sci(ts[f'{m}_sub_n{lvl:g}_ref'])}$")
                M.add(f"{k}subwiener{word}", f"${sci(ts[f'{m}_sub_w{lvl:g}_ref'])}$")
                M.add(f"{k}cross{word}", f"{ts[f'{m}_cross_n{lvl:g}']:.0f}")
                gain = ts[f"{m}_raw_ref"] / ts[f"{m}_sub_n{lvl:g}_ref"]
                M.add(f"{k}gain{word}", f"{float(f'{gain:.2g}'):g}")
                n_lvl = (lvl * np.deg2rad(1.0 / 60.0)) ** 2
                M.add(f"{k}lensfrac{word}",
                      pct(ts[f"{m}_kernel"] * n_lvl / cl_bb_reference, 0))
            M.add(f"{k}kernel", f"${f(ts[f'{m}_kernel'], 4)}"
                  f" \\pm {f(ts[f'{m}_kernel_err'], 4)}$")
            M.add(f"{k}tan", f(ts[f"{m}_tan2"], 4))
            M.add(f"{k}gg", f(2 * ts[f"{m}_g2"], 4))
            M.add(f"{k}tanexcess", pct(ts[f"{m}_tan2"] / (2 * ts[f"{m}_g2"]) - 1, 0))
        if proj == "ProjSFL":
            M.add("templatecorr", f(ts["corr"], 4))

    # ---- the coupling matrix against a diagonal transfer function ---------
    # F_b = <R C^CMB>_b / <C^CMB>_b per dl = 50 bandpower, calibrated on the
    # unlensed Planck TT and EE, applied to a white (Poisson-source) spectrum.
    # The error is that of the recovered bandpower, <R C>_b / (F_b <C>_b) - 1;
    # negative means the diagonal correction undershoots. F_b carries the
    # acoustic phase, so the error oscillates about a smooth offset with a
    # period of ~300 in l: each l0 is quoted as the mean over the six
    # bandpowers within l0 +- 150 (`...threeK`, `...fourK`) and the worst of
    # them (`...max`). SFL and ZEA are equal-area, so a D_l-flat (tSZ-like)
    # spectrum gives the same errors (`shear_leakage.diagonal_transfer_error`).
    # No band limit: the sky's foregrounds have none.
    white = np.where(ell_th >= 2, 1.0, 0.0)
    step, half = 5.0, 150.0
    for label, (mom_, i_, p, nd) in {
            "SFLtfd": (SL.field_moments(), 0, "ProjSFL", 0),
            "ZEAtfd": (SL.field_moments(), 0, "ProjZEA", 1)}.items():
        for l0, word in ((3000.0, "threeK"), (4000.0, "fourK")):
            ell_b = np.arange(l0 - half + step / 2, l0 + half, step)
            R = SL.coupling_matrix(mom_, i_, P[p], ell_b, ell_th,
                                   band_limit=None)
            for X in ("TT", "EE"):
                err = SL.diagonal_transfer_error(
                    R[f"{X}<-{X}"], ell_b, ell_th, th[f"unlensed_{X}"], white,
                    per_band=int(50 / step))
                worst = err[np.argmax(np.abs(err))]
                M.add(f"diag{X}{label}{word}", f"${100 * err.mean():+.{nd}f}\\%$")
                M.add(f"diag{X}{label}{word}max", f"${100 * worst:+.{nd}f}\\%$")

    # ---- TB from a TE-only sky: FlattenPol's residual rotation ------------
    # Delta is odd about the central meridian, so TB/TE vanishes on any
    # footprint symmetric about it; on the 1500d field's eastern half it does
    # not. `TBhalfSFL` is that half-field TB/TE; `DeltamaxSFL` the largest
    # |Delta| over the full 1500d field, in degrees.
    tetb = SL.check_te_tb()
    M.add("TBhalfSFL", pct(abs(tetb["1500d, RA 0 to 50", "ProjSFL"]["tb_te"]), 1))
    M.add("DeltamaxSFL", f"\\ang{{{np.degrees(tetb['1500d', 'ProjSFL']['delta_max']):.1f}}}")

    prose_numbers(M, mom)

    OUT.write_text(M.tex('numbers'))
    print(M.tex('numbers'))
    print(f"wrote {OUT}  ({len(M)} macros)")




# ==========================================================================
# Prose numbers: the 1500d field as reference footprint
#
# `prose_numbers` writes every macro Sects. 2-5 quote on SPT's 1500d field, on
# which the spectra figure and the scalar and leakage narrative rest, together
# with the literature footprints and the HEALPix pixel. The seven footprints
# form the sweep of Sect. 4.3 and keep their macros in `main`.
#
# A single dl = 50 band resolves the acoustic oscillation of a ratio to the
# input, so EE ratios are quoted as ratios of power averaged with a Hann window
# of half width L_HANN, which nulls a period of L_HANN, one acoustic spacing.
# ==========================================================================

L_HANN = 300.0


def _hann(y, ell, L):
    d = ell - L
    s = np.abs(d) < L_HANN
    w = np.cos(np.pi * d[s] / (2 * L_HANN))**2
    return (w * y[s]).sum() / w.sum()


def prose_numbers(M, mom):
    from scipy.signal import find_peaks
    import field_1500d as FD
    import healpix_sheared_window as HW
    import plot_model_residuals as PMR

    fd = FD.load()
    # The model of the residuals figure: every multipole, averaged per band.
    d = PMR.load_data()
    pm = d["mom"]
    lab = [S.PROJ_LABEL[p] for p in S.PROJS]
    J = {l: j for j, l in enumerate(lab)}
    sheared = ("SFL", "ZEA", "BICEP")
    ell = fd["ell"]
    cl, ee, bb = fd["cl_ee"], fd["ee_mean"], fd["bb_mean"]
    m_ee, m_bb, m_tt = d["model"]["EE"], d["model"]["BB"], d["model"]["TT"]
    tt_sims = d["measured"]["TT"]
    tt, cl_tt = tt_sims.mean(1), S.theory("unlensed_TT")
    g2 = fd["pipeline_g2"]
    wide = (ell > 500) & (ell < L_WALL)
    wide_bb = (ell > 500) & (ell < L_CLEAN[1])
    clean = (ell > L_CLEAN[0]) & (ell < L_CLEAN[1])

    idx = json.loads((S.RESULTS / "healpix_masks" / f"index_{S.FIELD}.json").read_text())
    M.add("fieldfsky", pct(idx["masks"][0]["fsky_binary"], 1))

    # ---- scalar: the acoustic spacing against the shear's band ------------
    th = T.load()
    ell_th, cl_th = th["ell"].astype(float), th["unlensed_EE"]
    tail = (ell_th > 3000) & (ell_th < 4500)
    n_tail = np.polyfit(np.log(ell_th[tail]), np.log(cl_th[tail]), 1)[0]
    M.add("gsqSFLfield", f(g2[J["SFL"]]))
    M.add("liftEEtailField", pct(n_tail * (n_tail + 2) / 4 * g2[J["SFL"]], 0))
    span = (ell_th > 500) & (ell_th < 4000)
    peaks = find_peaks(cl_th[span])[0]
    spacing = float(np.diff(peaks).mean())
    M.add("acousticspacing", f"{10 * round(spacing / 10):.0f}")
    M.add("hannwidth", f"{2 * L_HANN:.0f}")
    for l in ("SFL", "BICEP"):
        rms = np.sqrt(g2[J[l]])
        M.add(f"grms{l}field", f(rms, 2))
        M.add(f"lsmear{l}", f"{10 * round(spacing / (2 * rms) / 10):.0f}")

    ratio = ee / cl
    for l in lab:
        for L, word in ((1000., "oneK"), (3000., "threeK"), (4000., "fourK")):
            M.add(f"EEbox{l}{word}", f(_hann(ee[J[l]], ell, L) / _hann(cl, ell, L), 2))
        if l in ("SFL", "STG", "BICEP"):
            M.add(f"TTbox{l}fourK", f(_hann(tt[J[l]], ell, 4000.) / _hann(cl_tt, ell, 4000.), 2))
    s_one = (ell > 850) & (ell < 1450)
    s_three = (ell > 2850) & (ell < 3150)
    for l, s in (("SFL", s_one), ("STG", s_three)):
        M.add(f"EEswing{l}lo", f(ratio[J[l]][s].min(), 2))
        M.add(f"EEswing{l}hi", f(ratio[J[l]][s].max(), 2))
    centres = np.arange(800., 4500., 50.)
    z = J["ZEA"]
    zmeas = [_hann(ee[z], ell, L) / _hann(cl, ell, L) for L in centres]
    zmod = [_hann(m_ee[z], ell, L) / _hann(cl, ell, L) for L in centres]
    M.add("EEboxZEAdev", pct(np.max(np.abs(np.array(zmeas) - 1)), 0))
    M.add("EEboxZEAmodeldev", pct(np.max(np.abs(np.array(zmod) - 1)), 0))
    for L, word in ((1000., "OneK"), (3000., "ThreeK")):
        v = [_hann(ee[j], ell, L) / _hann(m_ee[j], ell, L) for j in J.values()]
        M.add(f"EEmodelRange{word}Field", rng(min(v), max(v), 3))
    v = [float((ee[j] / m_ee[j])[wide].mean()) for j in J.values()]
    M.add("wideEErangeField", rng(min(v), max(v), 3))

    # `lowlstruct`: the worst single band below l = 1000.
    pairs = [(tt[j], m_tt[j]) for j in J.values()] + [(ee[j], m_ee[j]) for j in J.values()]
    below = (ell > 500) & (ell < 1000)
    M.add("lowlstruct", pct(max(np.max(np.abs((a / b - 1)[below])) for a, b in pairs), 0))
    M.add("EEmodelworstclean", pct(max(np.max(np.abs((ee[j] / m_ee[j] - 1)[clean]))
                                       for j in J.values()), 0))
    ee_sims = S.load_field("EE")

    # Mean dilation of STG, weighted as the numerator of the forward model.
    s1, s2, w = SL._cells(pm, 0, J["STG"])
    wmean = np.sum(w * (s1 * s2)**2 * (s1 + s2) / 2) / np.sum(w * (s1 * s2)**2)
    M.add("STGshiftthreeK", f"{10 * round(3000 * (wmean - 1) / 10):.0f}")

    # ---- leakage on the field ---------------------------------------------
    sel = (ell > 500) & (ell < 3000)
    floor_bin = bb[J["STG"]] / ee[J["STG"]]
    def band(y, lo, hi):
        return y[(ell > lo) & (ell < hi)].mean()
    M.add("floorField", f"${sci(band(bb[J['STG']], *SL.BANDS[0]) / band(ee[J['STG']], *SL.BANDS[0]))}$")
    M.add("floorFieldTop", f"${sci(band(bb[J['STG']], *SL.BANDS[-1]) / band(ee[J['STG']], *SL.BANDS[-1]))}$")
    lvl = [bb[j][sel].mean() for j in J.values()]
    M.add("bbdecadesField", _WORDS[int(round(np.log10(max(lvl) / min(lvl))))])
    widev, cleanv, last = [], [], int(np.flatnonzero(S.BIN_HI <= L_WALL)[-1])
    for l in sheared:
        j = J[l]
        M.add(f"BBmodel{l}field", f(bb[j][sel].mean() / m_bb[j][sel].mean()))
        widev.append(float(((bb[j] / ee[j]) / (m_bb[j] / m_ee[j]))[wide_bb].mean()))
        r = (bb[j] / ee[j] - floor_bin) / (m_bb[j] / m_ee[j])
        cleanv.append(float(r[clean].mean()))
        M.add(f"rise{l}field", f(r[last], 2))
    M.add("wideBBrangeField", rng(min(widev), max(widev), 3))
    M.add("cleanBBrangeField", rng(min(cleanv), max(cleanv), 3))
    M.add("cleanBBworstField", pct(max(abs(v_ - 1) for v_ in cleanv), 1))
    bic_excess = (bb[J["BICEP"]] / ee[J["BICEP"]] - m_bb[J["BICEP"]] / m_ee[J["BICEP"]])[last]
    M.add("BICEPexcessfloor", f"{bic_excess / floor_bin[last]:.0f}")
    M.add("magfactorField", rf"{pm['s1_max'][0, J['BICEP']]:.1f}\times")

    # BICEP's BB/EE below l = 1000 against its plateau, in 100-wide bands.
    edges = np.arange(300., 1000., 100.)
    bic = J["BICEP"]
    low = [band(bb[bic], e, e + 100) / band(ee[bic], e, e + 100) for e in edges]
    lowm = [band(m_bb[bic], e, e + 100) / band(m_ee[bic], e, e + 100) for e in edges]
    M.add("BBEEBICEPlow", f(max(low), 3))
    M.add("BBEEBICEPlowmodel", f(max(lowm), 3))
    M.add("BBEEBICEPhigh", f(band(bb[bic], 3000, L_WALL) / band(ee[bic], 3000, L_WALL), 3))

    for l in ("SFL", "BICEP"):
        j = J[l]
        t = [band(bb[j], lo, hi) / band(ee[j], lo, hi) / (2 * g2[j])
             for lo, hi in SL.BANDS[1:]]
        M.add(f"tilt{l}fieldlo", f(t[0], 2))
        M.add(f"tilt{l}fieldhi", f(t[-1], 2))

    leak_zea = band(bb[z], *SL.BANDS[0]) / band(ee[z], *SL.BANDS[0])
    lensed = float(th["lensed_BB"][1000] / th["lensed_EE"][1000])
    M.add("ZEAleakField", f"${sci(leak_zea)}$")
    M.add("ZEAoverlens", pct(leak_zea / lensed, 0))

    # Power conservation, field and sweep; sweep pairs as in `main`.
    d_ee, d_bb = S.load("EE").mean(2), S.load("BB").mean(2)
    d_eb = S.load("EE", leg=S.EB_LEG).mean(2)
    dev = np.abs(((d_ee + d_bb) / d_eb)[:, :, wide].mean(-1) - 1)
    P = {p: j for j, p in enumerate(S.PROJS)}
    for i, p in SL.CLIPPED:
        dev[i, P[p]] = np.nan
    fdev = [abs(((ee[j] + bb[j]) / fd["ee_eb_mean"][j])[wide].mean() - 1) for j in J.values()]
    M.add("powerconserveAll", f"${sci(max(np.nanmax(dev), max(fdev)))}$")

    # Adjacent-band correlation of the 20 sims on the field: mean of the first
    # off-diagonal over 500 < l < L_WALL, with the error of that mean.
    def adjacent(x):
        return float(np.mean(np.diag(np.corrcoef(x[:, wide], rowvar=False), 1)))
    M.add("adjEESFL", f(adjacent(ee_sims[J["SFL"]]), 2))
    M.add("adjEEZEA", f(adjacent(ee_sims[J["ZEA"]]), 2))
    M.add("adjnoise", f(1 / np.sqrt((S.NSIMS - 1) * (wide.sum() - 1)), 2))

    # ---- a matched filter on a sheared grid ---------------------------------
    # An isotropic Gaussian filter on an equal-area grid recovers sqrt(1 - g^2)
    # of a compact source's amplitude in white noise.
    s1, s2, w = SL._cells(pm, 0, J["SFL"])
    g = (s1 - s2) / (s1 + s2)
    M.add("filterSFLmean", pct(float(np.sum(w * (1 - np.sqrt(1 - g**2))) / np.sum(w)), 1))
    M.add("filterSFLmax", pct(float(1 - np.sqrt(1 - g[w > 0].max()**2)), 0))

    # ---- Pires et al. (2020): their 10 x 10 deg gnomonic field --------------
    # TAN has principal scales sec^2 r and sec r, so g = tan^2(r / 2); uniform
    # weight over the square on the tangent plane.
    a = np.tan(np.deg2rad(5.0))
    x = (np.arange(800) + .5) / 800 * 2 * a - a
    r = np.arctan(np.hypot(*np.meshgrid(x, x)))
    M.add("TANtenBBEE", f"${sci(2 * np.mean(np.tan(r / 2)**4), 0)}$")

    # ---- HEALPix pixels -----------------------------------------------------
    px = HW.pixel_shear()
    hp_g = px["g"]
    M.add("hpgmaxeq", f(hp_g[np.abs(px["z"]) < 2 / 3].max(), 2))
    M.add("hpgsq", f(np.mean(hp_g**2), 3))
    ours, square = [], []
    for nside in (1024, 4096):
        for x_ in (0.5, 1, 1.5, 2, 2.5, 3 - 1 / nside):
            p, s_, q = HW.sheared_window(nside, x_)
            ours.append(abs(s_ / p - 1))
            square.append(abs(q / p - 1))
    M.add("hpwinmatch", f"${sci(max(ours), 0)}$")
    M.add("hpwinsquare", pct(max(square), 1))

    # ---- ZEA's last band on the reference footprint ------------------------
    meas = SL.measured_ratios()
    gz = mom["g2"][MASK, P["ProjZEA"]]
    M.add("ZEAlastexcess", pct(meas[-1, MASK, P["ProjZEA"]] / (2 * gz) - 1, 1))
    M.add("ZEAlastfloor", pct(meas[-1, MASK, P[SL.FLOOR_PROJ]] / (2 * gz), 1))


def _offdiag(ee, bb, ell, lo=500, hi=4800):
    """Mean off-diagonal bandpower correlation over a multipole range.

    Computed per (mask, projection) from the 20 seeds, then reported as the
    mean and worst case over the sheared configurations -- the ones whose BB
    is genuine leakage -- with the conformal projection's own worst case kept
    separate. STG's BB is a seed-independent numerical template, so its
    bandpowers correlate with each other almost perfectly; averaging it in
    with the rest would hide both facts.
    """
    sel = (ell > lo) & (ell < hi)
    jstg = S.PROJS.index(SL.FLOOR_PROJ)

    def per_config(arr, i, j):
        x = arr[i, j][:, sel]                      # (n_seed, n_bin)
        c = np.corrcoef(x, rowvar=False)
        off = ~np.eye(c.shape[0], dtype=bool)
        return float(np.nanmean(c[off]))

    out = {}
    for name, arr in (("ee", ee), ("bb", bb)):
        vals = [per_config(arr, i, j) for i in range(len(S.SWEEP))
                for j in range(len(S.PROJS)) if j != jstg]
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_max"] = float(np.max(vals))
    out["stg_max"] = max(per_config(bb, i, jstg) for i in range(len(S.SWEEP)))
    # Sampling floor: a sample correlation from n seeds has 1/sqrt(n-1) of
    # noise per entry even when the truth is zero.
    out["noise"] = 1.0 / np.sqrt(S.NSIMS - 1)
    return out


def _field_1500d():
    """Predicted BB/EE = 2<g^2> on SPT's real 1500d footprint, per projection.

    From the moments of the pipeline's own field mask.
    """
    g2 = SL.field_moments()["g2"][0]
    return {S.PROJ_LABEL[p]: 2 * float(g2[j]) for j, p in enumerate(S.PROJS)}


if __name__ == "__main__":
    main()
