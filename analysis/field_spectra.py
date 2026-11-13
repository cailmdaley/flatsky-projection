"""SPT's 1500d field, simulated: measured leakage against the forward model.

The seven disks are concentric about the projection centre, which makes
SFL's and BICEP's mask-averaged shear coincide. The 1500d field (RA +-50 deg,
dec -70 to -42 deg) is wide and short, so it splits them, and the paper's
prediction for it -- SFL and BICEP a factor `tfdratio` apart -- rests on the
forward model alone unless the field is simulated. `bandpowers` carries it
through the same pipeline and the same projected maps as the disks; this
script, the `field_spectra` output, compares the result with the model.

What is compared, per projection:

* BB/EE band-averaged over 1000 < l < 3000 (numerator and denominator
  averaged separately, as `shear_leakage.measured_ratios` does), minus the
  same quantity for STG, the pipeline's E-to-B floor on this footprint; and
  the leading-order law 2<g^2> beside it.
* the floor-subtracted ratio over the forward model, per bin, averaged over
  `L_CLEAN` -- the disk suite's clean statement (`cleanBBrange`).
* EE over the binned input spectrum, measured and modelled.

The model reads the shear moments of the pipeline's own field mask,
`spectra.Run.field_moments`.

Run:  bash env.sh python3 analysis/field_spectra.py
          --bandpowers results/fiducial/bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra data/theory_spectra.npz
          --shear-moments results/fiducial/shear_moments --out <dir>
Out:  <out>/field_spectra.npz (read it with `load`), and the diagnostic
      figure <out>/field_1500d.{pdf,png}.
"""

import argparse
from pathlib import Path

import numpy as np

import shear_leakage as SL
import spectra as S
import theory as T

BAND = SL.BANDS[0]            # (1000, 3000)
L_CLEAN = (1000, 4000)        # as the `numbers` output's L_CLEAN


def band_ratio(ee, bb, sel):
    """Per-sim and mean band BB/EE, numerator and denominator averaged separately."""
    per_sim = bb[:, sel].mean(1) / ee[:, sel].mean(1)
    return per_sim, bb[:, sel].mean() / ee[:, sel].mean()


def compute(run):
    ee = run.load_field("EE")                     # (proj, sim, bin)
    bb = run.load_field("BB")
    ee_eb = run.load_field("EE", leg=S.EB_LEG)
    ell = run.ell
    th = T.load(run.theory_spectra)
    ell_th, cl_th = th["ell"].astype(float), th["unlensed_EE"]
    cl_ee = run.theory("unlensed_EE")
    jstg = S.PROJS.index(SL.FLOOR_PROJ)

    moms = {"pipeline": run.field_moments()}
    sel = (ell > BAND[0]) & (ell < BAND[1])
    cband = (ell > L_CLEAN[0]) & (ell < L_CLEAN[1])
    n = ee.shape[1]

    out = dict(ell=ell, ee_mean=ee.mean(1), bb_mean=bb.mean(1),
               ee_std=ee.std(1, ddof=1), bb_std=bb.std(1, ddof=1),
               ee_eb_mean=ee_eb.mean(1), cl_ee=cl_ee, n_sims=n)

    floor_sims, floor = band_ratio(ee[jstg], bb[jstg], sel)
    floor_bin = bb[jstg].mean(0) / ee[jstg].mean(0)
    out["floor_band"] = floor
    meas, meas_err, persim = [], [], []
    for j in range(len(S.PROJS)):
        ps, m = band_ratio(ee[j], bb[j], sel)
        d = ps - floor_sims
        meas.append(m - floor)
        meas_err.append(d.std(ddof=1) / np.sqrt(n))
        persim.append(d)
    out["meas_band"] = np.array(meas)
    out["meas_band_err"] = np.array(meas_err)
    persim = np.array(persim)

    jsfl, jbic = S.PROJS.index("ProjSFL"), S.PROJS.index("ProjBICEP")
    r = persim[jsfl].mean() / persim[jbic].mean()
    # Delta-method error on the ratio of two correlated sim means.
    cov = np.cov(persim[[jsfl, jbic]]) / n
    a, b = persim[jsfl].mean(), persim[jbic].mean()
    out["sfl_bicep"] = r
    out["sfl_bicep_err"] = abs(r) * np.sqrt(cov[0, 0] / a**2 + cov[1, 1] / b**2
                                            - 2 * cov[0, 1] / (a * b))

    for name, mom in moms.items():
        g2 = mom["g2"][0]
        m_bb = np.empty((len(S.PROJS), len(ell)))
        m_ee = np.empty_like(m_bb)
        for j in range(len(S.PROJS)):
            m_bb[j], m_ee[j] = SL.predicted_spectra(mom, 0, j, ell, ell_th, cl_th,
                                                    band_limit=run.lmax)
        out[f"{name}_g2"] = g2
        out[f"{name}_m_bb"], out[f"{name}_m_ee"] = m_bb, m_ee
        out[f"{name}_model_band"] = np.array(
            [m_bb[j][sel].mean() / m_ee[j][sel].mean() for j in range(len(S.PROJS))])
        clean = []
        for j in range(len(S.PROJS)):
            rr = (bb[j].mean(0) / ee[j].mean(0) - floor_bin) / (m_bb[j] / m_ee[j])
            clean.append(float(rr[cband].mean()))
        out[f"{name}_clean"] = np.array(clean)
    out["projs"] = S.PROJS
    return out


def load(directory):
    """The `field_spectra` output as a dict of arrays."""
    with np.load(Path(directory) / "field_spectra.npz") as f:
        return {k: f[k] for k in f.files}


def report(res):
    lab = [S.PROJ_LABEL[p] for p in S.PROJS]
    jsfl, jbic = lab.index("SFL"), lab.index("BICEP")
    print(f"1500d field, {int(res['n_sims'])} sims, QU leg, BB/EE over {BAND[0]}<l<{BAND[1]}")
    print(f"  STG floor (band) {res['floor_band']:.3e}")
    print(f"  {'proj':<7}{'meas-floor':>12}{'+-':>10}{'model':>11}{'2<g2>':>11}"
          f"{'meas/model':>12}{'clean':>8}")
    for j, l in enumerate(lab):
        m, e = res["meas_band"][j], res["meas_band_err"][j]
        mo = res["pipeline_model_band"][j]
        print(f"  {l:<7}{m:12.3e}{e:10.1e}{mo:11.3e}{2 * res['pipeline_g2'][j]:11.3e}"
              f"{m / mo:12.3f}{res['pipeline_clean'][j]:8.3f}")
    print(f"  SFL/BICEP measured {res['sfl_bicep']:.2f} +- {res['sfl_bicep_err']:.2f}; "
          f"model {res['pipeline_model_band'][jsfl] / res['pipeline_model_band'][jbic]:.2f}; "
          f"2<g2> {res['pipeline_g2'][jsfl] / res['pipeline_g2'][jbic]:.2f}")
    ell = res["ell"]
    print("  EE/input (measured | model) at l ~ 1000, 3000, 4000:")
    for j, l in enumerate(lab):
        row = []
        for L in (1000, 3000, 4000):
            k = np.argmin(abs(ell - L))
            row.append(f"{res['ee_mean'][j][k] / res['cl_ee'][k]:.4f}|"
                       f"{res['pipeline_m_ee'][j][k] / res['cl_ee'][k]:.4f}")
        print(f"  {l:<7}" + "  ".join(row))


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as PS

    PS.rc()
    ell = res["ell"]
    n = float(res["n_sims"])
    jstg = S.PROJS.index(SL.FLOOR_PROJ)
    floor_bin = res["bb_mean"][jstg] / res["ee_mean"][jstg]
    keep = (ell > 500) & (ell < 4800)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(PS.COL_W, PS.COL_W * 1.25),
                                 sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    for j, p in enumerate(S.PROJS):
        c, mk = PS.PROJ_COLORS[p], PS.PROJ_MARKERS[p]
        y = res["bb_mean"][j] / res["ee_mean"][j]
        if p != SL.FLOOR_PROJ:
            y = y - floor_bin
        yerr = res["bb_std"][j] / np.sqrt(n) / res["ee_mean"][j]
        ax.errorbar(ell[keep], y[keep], yerr=yerr[keep], fmt=mk, color=c, ms=PS.MS * 0.8,
                    mew=PS.MEW, mfc="none", lw=0.5, label=PS.PROJ_LABEL[p])
        model = res["pipeline_m_bb"][j] / res["pipeline_m_ee"][j]
        if p != SL.FLOOR_PROJ:
            ax.plot(ell[keep], model[keep], color=c, lw=PS.LW)
        bx.plot(ell[keep], (res["ee_mean"][j] / res["cl_ee"])[keep], mk, color=c,
                ms=PS.MS * 0.8, mew=PS.MEW, mfc="none")
        bx.plot(ell[keep], (res["pipeline_m_ee"][j] / res["cl_ee"])[keep], color=c, lw=PS.LW)
    ax.set_yscale("log")
    ax.set_ylabel(r"$BB/EE$")
    ax.set_ylim(3e-8, 30)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax.legend(ncol=5, frameon=False, fontsize=PS.LEGEND_SIZE, loc="upper center",
              handletextpad=0.1, columnspacing=0.6)
    ax.set_title("1500d field: points simulated, lines forward model",
                 fontsize=PS.LEGEND_SIZE)
    bx.axhline(1, color=PS.RULE, lw=PS.LW_REF)
    bx.set_ylabel(r"$EE/C_\ell^{EE}$")
    bx.set_xlabel(r"$\ell$")
    fig.tight_layout(h_pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "bandpowers", "simulation_windows", "theory_spectra", "shear_moments")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = compute(S.Run.from_args(args))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "field_spectra.npz", **res)
    report(res)
    figure(res, args.out / "field_1500d.pdf")


if __name__ == "__main__":
    main()
