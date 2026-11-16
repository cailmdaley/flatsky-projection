"""Projection-survey macros, survey_numbers.tex.

Formats `projection_survey`, the geometric comparison on the unapodized 1500d
rectangle:

  surveyLCCoverSTG   STG's rms of ln mu over the best Lambert conformal conic's
  surveyZEAoverAEA   ZEA's <g^2> over the best Albers equal-area conic's
  surveyLCCoverOpt   how far the conic's rms of ln mu sits above the
                     least-squares conformal optimum
  surveySFLrecentre  SFL's <g^2> as the paper uses it, over SFL with the field
                     moved onto its equator

Run:  bash env.sh python3 analysis/paper_numbers/survey_numbers.py
          --projection-survey <dir> --out <dir>
Out:  <out>/survey_numbers.tex and <out>/survey_numbers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402
from macros import Macros  # noqa: E402


def macros(d):
    paper, field, opt = d["paper"], d["field"], d["opt"]
    lcc = field["lcc"]["lnmu"]["rms_lnmu"]
    M = Macros()
    M.add("surveyLCCoverSTG", f"{paper['STG']['rms_lnmu'] / lcc:.1f}")
    M.add("surveyZEAoverAEA", f"{paper['ZEA']['g2'] / field['aea']['g2']['g2']:.1f}")
    M.add("surveyLCCoverOpt", f"{100 * (lcc / opt['conformal-LS']['rms_lnmu'] - 1):.0f}\\%")
    M.add("surveySFLrecentre", f"{paper['SFL']['g2'] / field['sinu']['g2']['g2']:.0f}")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "projection_survey")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = json.loads((args.projection_survey / "projection_survey.json").read_text())
    macros(d).write(args.out, "survey_numbers")


if __name__ == "__main__":
    main()
