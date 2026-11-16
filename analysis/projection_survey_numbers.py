"""Write paper/survey_numbers.tex from analysis/projection_survey.json.

The survey's geometric comparison on the unapodized 1500d rectangle: the
best conic of each class against the paper's zenithal pair, the conformal
optimum, and SFL re-centred on its own equator.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
d = json.loads((root / "analysis/projection_survey.json").read_text())
paper, field, opt = d["paper"], d["field"], d["opt"]

lcc = field["lcc"]["lnmu"]["rms_lnmu"]
aea = field["aea"]["g2"]["g2"]
sfl_rot = field["sinu"]["g2"]["g2"]
cheb = opt["conformal-LS"]["rms_lnmu"]

macros = {
    "surveyLCCoverSTG": f"{paper['STG']['rms_lnmu'] / lcc:.1f}",
    "surveyZEAoverAEA": f"{paper['ZEA']['g2'] / aea:.1f}",
    "surveyLCCoverOpt": f"{100 * (lcc / cheb - 1):.0f}\\%",
    "surveySFLrecentre": f"{paper['SFL']['g2'] / sfl_rot:.0f}",
}
out = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
(root / "paper/survey_numbers.tex").write_text(out)
print(out)
