"""What a diagonal transfer function misses that the coupling matrix keeps, as measured.

Reads the `scalar_summary` output: CMB TT skies and independent unit-C_l white
skies through the scalar leg on the 1500d field, for all four projections.

  top     recovered white bandpower over the truth, minus one, after a diagonal
          correction F_b = measured TT / input TT. Dashed lines are the same
          quantity from the forward model alone. F_b carries the CMB's own
          response to the projection, including its acoustic phase.
  bottom  measured white over the coupling matrix applied to the white input,
          minus one, on the same skies.

Run:  bash env.sh python3 analysis/plot_transfer_error.py
          --scalar-summary <scalar_summary dir> --out <dir>
Out:  <out>/transfer_error.pdf, which the paper includes, and
      <out>/fig_transfer_error.png
"""

import argparse
from pathlib import Path

import numpy as np

import paper_style as PS
import spectra as S

ELL_LO, ELL_HI = 500.0, 4800.0
ZOOM = 5.0                   # bottom axis half-range, percent


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "scalar_summary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.scalar_summary / "scalar_summary.npz")
    ell = d["ell"]
    keep = (ell >= ELL_LO) & (ell <= ELL_HI)
    plt.rcParams.update(PS.rc())
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(PS.COL_W, 3.3), sharex=True, layout="constrained",
        height_ratios=(1.7, 1))

    for j, proj in enumerate(d["projections"]):
        color = PS.PROJ_COLORS[str(proj)]
        measured = 100 * (d["diagonal_white"][j] - 1)
        model = 100 * (d["predicted_diagonal_white"][j] - 1)
        closure = 100 * (d["full_white"][j] - 1)
        sem = 100 * d["full_white_sem"][j]
        top.plot(ell[keep], measured[keep], color=color, lw=1.1)
        top.plot(ell[keep], model[keep], color=PS.INK, lw=0.5, ls=(0, (3, 2)))
        bot.fill_between(ell[keep], (closure - sem)[keep], (closure + sem)[keep],
                         color=color, alpha=0.25, lw=0)
        bot.plot(ell[keep], closure[keep], color=color, lw=0.9)

    for ax in (top, bot):
        ax.axhline(0, color=PS.RULE, lw=0.6, zorder=0)
        ax.set_xlim(ELL_LO, ELL_HI)
    bot.set_ylim(-ZOOM, ZOOM)
    top.set_ylabel(r"diagonal error [\%]")
    bot.set_ylabel(r"$\mathcal{R}$ error [\%]")
    bot.set_xlabel(r"$\ell$")

    handles = [Line2D([], [], color=PS.PROJ_COLORS[str(p)], lw=1.3,
                      label=PS.PROJ_LABEL[str(p)]) for p in d["projections"]]
    handles.append(Line2D([], [], color=PS.INK, lw=0.6, ls=(0, (3, 2)), label="model"))
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False,
               handlelength=1.5, columnspacing=1.0)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "transfer_error.pdf")
    fig.savefig(args.out / "fig_transfer_error.png", dpi=200)
    print(f"wrote {args.out / 'transfer_error.pdf'}")


if __name__ == "__main__":
    main()
