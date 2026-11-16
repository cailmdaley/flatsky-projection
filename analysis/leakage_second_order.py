"""The leakage law where the shear is not small: SFL on the two largest rectangles.

BB/EE = 2<g^2> is the leading order of <sin^2 2 dpsi>; the exact azimuthal
average is 2 g^2 (1 - 3 g^2 / 4) per point (paper Eq. law) before the
sky spectrum's response to the stretch is counted, and the forward model
(`shear_leakage.predicted_spectra`) keeps every order. On SFL's two largest
rectangles, rect_7p81 and rect_10p06, the local shear reaches g ~ 0.5-0.6, so
the leading-order law and the forward model part company, and the
measurement says which holds.

Per footprint and band (`shear_leakage.BANDS`), from the seed-mean QU-leg
bandpowers with the simulation windows removed:

  measured        BB/EE, numerator and denominator band-averaged separately,
                  with STG's ratio in the same band and footprint subtracted
                  (the pipeline's floor), and its standard error over seeds;
  model           the forward model's BB/EE in the same band;
  law             2<g^2> of the footprint's mask;
  exact_local     2<g^2 (1 - 3 g^2 / 4)>, the exact local kernel without the
                  stretch, from the shear moments' (s1, s2) cells.

The paper quotes none of this yet.

Run:  bash env.sh python3 analysis/leakage_second_order.py
          --bandpowers results/fiducial/bandpowers
          --simulation-windows results/fiducial/simulation_windows
          --theory-spectra results/fiducial/theory_spectra
          --shear-moments results/fiducial/shear_moments --out <dir>
Out:  <out>/leakage_second_order.json and a diagnostic plot,
      leakage_second_order.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import shear_leakage as SL
import spectra as S
import theory as T

PROJ = "ProjSFL"
FOOTPRINTS = ("rect_7p81", "rect_10p06")


def measured(run, i, j, jstg, bands):
    """Floor-subtracted band BB/EE and its standard error over seeds, per band."""
    ee, bb = run.load("EE")[i], run.load("BB")[i]
    out, err = [], []
    for lo, hi in bands:
        sel = (run.ell > lo) & (run.ell < hi)
        per_seed = (bb[j][:, sel].mean(1) / ee[j][:, sel].mean(1)
                    - bb[jstg][:, sel].mean(1) / ee[jstg][:, sel].mean(1))
        out.append(bb[j][:, sel].mean() / ee[j][:, sel].mean()
                   - bb[jstg][:, sel].mean() / ee[jstg][:, sel].mean())
        err.append(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
    return np.array(out), np.array(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "bandpowers", "simulation_windows", "theory_spectra", "shear_moments")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run = S.Run.from_args(args)
    th = T.load(run.theory_spectra)
    j, jstg = S.PROJS.index(PROJ), S.PROJS.index(SL.FLOOR_PROJ)
    grid = run.ell[(run.ell > SL.BANDS[0][0]) & (run.ell < max(hi for _, hi in SL.BANDS))]
    result = dict(projection=PROJ, bands=SL.BANDS, footprints={})
    for name in FOOTPRINTS:
        i = S.SWEEP.index(name)
        mom = run.moments([name])
        s1, s2, w = SL._cells(mom, 0, j)
        g2 = ((s1 - s2) / (s1 + s2))**2
        bb, ee = SL.predicted_spectra(mom, 0, j, grid, th["ell"].astype(float),
                                      th["unlensed_EE"], band_limit=run.lmax)
        model = [bb[(grid > lo) & (grid < hi)].mean() / ee[(grid > lo) & (grid < hi)].mean()
                 for lo, hi in SL.BANDS]
        meas, err = measured(run, i, j, jstg, SL.BANDS)
        row = dict(g2=float(mom["g2"][0, j]), s1_max=float(mom["s1_max"][0, j]),
                   s2_min=float(mom["s2_min"][0, j]),
                   law=float(2 * mom["g2"][0, j]),
                   exact_local=float(2 * np.sum(w * g2 * (1 - 0.75 * g2)) / np.sum(w)),
                   measured=meas.tolist(), measured_se=err.tolist(), model=model)
        result["footprints"][name] = row
        print(f"{name} {PROJ}: <g^2> {row['g2']:.4f}, 2<g^2> {row['law']:.4f}, "
              f"exact local {row['exact_local']:.4f}")
        print(f"  {'band':<12}{'measured':>12}{'+-':>9}{'model':>10}{'meas/model':>12}{'meas/2g2':>10}")
        for (lo, hi), m, e, mo in zip(SL.BANDS, meas, err, model):
            print(f"  {lo}-{hi:<7}{m:12.5f}{e:9.5f}{mo:10.5f}{m / mo:12.4f}{m / row['law']:10.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leakage_second_order.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.out / "leakage_second_order.png")


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as PS

    plt.rcParams.update(PS.rc())
    fig, ax = plt.subplots(figsize=(PS.COL_W, 2.4), layout="constrained")
    bands = result["bands"][1:]
    x = np.array([(lo + hi) / 2 for lo, hi in bands])
    for k, (name, row) in enumerate(result["footprints"].items()):
        color = PS.SECONDARY[k]
        m, e = np.array(row["measured"][1:]), np.array(row["measured_se"][1:])
        ax.errorbar(x, m / row["law"], e / row["law"], ls="none", marker="o", ms=3,
                    color=color, label=f"{name}, measured")
        ax.plot(x, np.array(row["model"][1:]) / row["law"], color=color, lw=1,
                label=f"{name}, forward model")
        ax.axhline(row["exact_local"] / row["law"], color=color, lw=0.6, ls=":")
    ax.axhline(1, color=PS.RULE, lw=0.6)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$(BB/EE) \,/\, 2\langle g^2\rangle$")
    ax.set_title("SFL: measured and modelled leakage over the leading-order law",
                 fontsize=PS.LEGEND_SIZE)
    ax.legend(frameon=False, fontsize=PS.LEGEND_SIZE)
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
