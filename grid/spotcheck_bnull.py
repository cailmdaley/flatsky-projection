"""Spot-check the EB reference leg's B map is numerically null."""
import numpy as np
from spt3g import core, maps  # noqa: F401  (maps registers FlatSkyMap)

base = "/sptlocal/user/cailmdaley/pol_project/results/highell/projected_eb/"
for fname in ["eb_ProjSTG_seed0000.g3", "eb_ProjBICEP_seed0000.g3"]:
    frames = [fr for fr in core.G3File(base + fname) if "E" in fr]
    fr = frames[0]
    E = np.asarray(fr["E"])
    B = np.asarray(fr["B"])
    print(f"{fname}  E std {np.nanstd(E):.3g}  B std {np.nanstd(B):.3g}")
