"""What a diagonal transfer function misses that the coupling matrix keeps, as measured.

Reads `paper/scalar_summary.npz` from `paper_numbers.py --scalar-only`: 20 CMB
TT skies and 20 independent unit-C_l white skies through the scalar leg on the
1500d field, for all four projections.

  top     recovered white bandpower over the truth, minus one, after a diagonal
          correction F_b = measured TT / input TT. Dashed lines are the same
          quantity from the forward model alone. F_b carries the CMB's own
          response to the projection, including its acoustic phase.
  bottom  measured white over the coupling matrix applied to the white input,
          with the common measured/model residual of TT divided out, minus one.

Run:  bash env.sh python3 analysis/plot_transfer_error.py
Out:  figures/transfer_error.pdf
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as PS

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "figures" / "transfer_error.pdf"
FIELD = 1                    # index of the 1500d field in the summary
ELL_LO, ELL_HI = 500.0, 4800.0
ZOOM = 5.0                   # bottom axis half-range, percent


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = np.load(ROOT / "paper" / "scalar_summary.npz")
    ell = d["ell"]
    keep = (ell >= ELL_LO) & (ell <= ELL_HI)
    plt.rcParams.update(PS.rc())
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(PS.COL_W, 3.3), sharex=True, layout="constrained",
        height_ratios=(1.7, 1))

    for j, proj in enumerate(d["projections"]):
        color = PS.PROJ_COLORS[str(proj)]
        measured = 100 * (d["diagonal_white"][FIELD, j] - 1)
        model = 100 * (d["predicted_diagonal_white"][FIELD, j] - 1)
        closure = 100 * (d["full_white"][FIELD, j] - 1)
        sem = 100 * d["full_white_sem"][FIELD, j]
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
    bot.set_ylabel(r"$R$ error [\%]")
    bot.set_xlabel(r"$\ell$")

    handles = [Line2D([], [], color=PS.PROJ_COLORS[str(p)], lw=1.3,
                      label=PS.PROJ_LABEL[str(p)]) for p in d["projections"]]
    handles.append(Line2D([], [], color=PS.INK, lw=0.6, ls=(0, (3, 2)), label="model"))
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False,
               handlelength=1.5, columnspacing=1.0)
    fig.savefig(FIGURE)
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
