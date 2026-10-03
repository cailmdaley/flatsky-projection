"""One palette for the paper's figures, and one rule about it.

The rule
--------
A projection's colour means that projection, in every figure, and it never
means anything else. Any encoding that is *not* a projection -- a ladder of
mask sizes, a comparison of candidate shear reductions, a reference line --
draws from `SECONDARY` (neutrals) or `MASK_RAMP` (a single-hue sequential
ramp) so that the reader never has to ask whether a colour is carrying
identity or magnitude.

The palette
-----------
Five categorical hues in the paper's fixed projection order (SFL, SIN, STG,
ZEA, BICEP). Validated with the dataviz skill's checker in *all-pairs* mode
on a light surface:

    lightness band  PASS      chroma floor    PASS
    CVD separation  PASS (worst all-pairs dE 8.9, protan, STG-SIN)
    normal vision   PASS (worst all-pairs dE 18.0, BICEP-SIN)
    contrast        WARN on ZEA's gold (2.1:1), relieved as the skill
                    requires: every series carries a legend entry and a
                    distinct marker, so identity is never colour alone.

Marker shapes are part of the identity, not decoration: they are what keeps
the figures readable in greyscale print and for the residual CVD risk.
"""

# Projection identity. Order is the paper's: SFL, SIN, STG, ZEA, BICEP.
PROJ_COLORS = {
    "ProjSFL":   "#3F76B8",   # blue
    "ProjSIN":   "#EE6677",   # rose
    "ProjSTG":   "#228833",   # green
    "ProjZEA":   "#C7B03F",   # gold
    "ProjBICEP": "#AA3377",   # purple
}

PROJ_MARKERS = {
    "ProjSFL": "o", "ProjSIN": "s", "ProjSTG": "x", "ProjZEA": "^",
    "ProjBICEP": "D",
}

PROJS = list(PROJ_COLORS)

PROJ_LABEL = {
    "ProjSFL": "SFL", "ProjSIN": "SIN", "ProjSTG": "STG",
    "ProjZEA": "ZEA", "ProjBICEP": "BICEP",
}

# Everything that is not a projection. Neutral by construction: it reads as
# "different kind of thing" at a glance, and it cannot be mistaken for one of
# the five. Three steps, well separated in lightness, so they survive
# greyscale and every flavour of colour vision.
SECONDARY = ["#232323", "#616161", "#9E9E9E"]

# Sequential ramp for an ordered non-projection quantity (the ladder of mask
# sizes). One hue, light to dark, per the sequential rule -- a desaturated
# slate that no projection colour sits near.
# The light end stops at a mid slate rather than near-white: the smallest
# disks are drawn over the pale interior of the largest apodized mask, and a
# paler line there is a line the reader cannot see.
MASK_RAMP = ["#93A4B0", "#7C8F9D", "#657A8A", "#4F6678", "#3A5265", "#263E52",
             "#12293C"]

# Ink and rules. Text wears text tokens, never a series colour.
INK = "#0B0B0B"
INK_SOFT = "#52514E"
RULE = "#898781"


def mask_ramp(n):
    """`n` evenly spaced steps of the sequential slate ramp."""
    if n == len(MASK_RAMP):
        return list(MASK_RAMP)
    idx = [round(i * (len(MASK_RAMP) - 1) / max(n - 1, 1)) for i in range(n)]
    return [MASK_RAMP[i] for i in idx]


# Print geometry, read off aa.cls (\the\columnwidth, \the\textwidth in TeX
# points, 72.27 to the inch). Every figure is drawn at exactly the width it is
# placed at, so LaTeX includes it at scale 1 and 8 pt in matplotlib is 8 pt on
# the page.
COL_W = 256.0748 / 72.27     # 3.543 in, 90 mm
TEXT_W = 523.5307 / 72.27    # 7.244 in, 184 mm

FONT_SIZE = 8.0
LEGEND_SIZE = 7.5

# Marks. One weight for model curves, one for reference lines, one marker
# size for every per-bandpower point.
LW = 1.0
LW_REF = 0.8
MS = 3.2
MEW = 0.7


def rc():
    """The paper's matplotlib rcParams: 8 pt serif type set by LaTeX in the
    paper's own txfonts, recessive grid and axes, and a figure saved at its
    drawn size (no tight bbox), so the PDF width is the figsize width."""
    return {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{txfonts}",
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE, "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.3,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
        "grid.color": RULE,
        "axes.edgecolor": RULE, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.labelpad": 2.5,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "lines.linewidth": LW, "lines.markersize": MS,
        "lines.markeredgewidth": MEW,
        "figure.dpi": 150,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.bbox": None, "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
