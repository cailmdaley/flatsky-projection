"""The TT quadratic estimator on projected maps, summarized for Sect. 6.1.

Three QE outputs and one prediction, reduced to one summary:

  qe          lensing_qe_sims: per projection the mean-field power by L band,
              the signed grid-lens response over the flat control, the
              metric's rms and pseudo-B fraction, and the fit of the
              low-L mean field to the squeezed metric prediction
              (`lensing_report`).
  resolution  lensing_qe_sims against its 2-arcmin repeat
              lensing_qe_resolution: the paired fractional change of the
              response and of the template fit (`lensing_resolution_report`).
  sky         lensing_qe_sky_response: the spherical-lens overlap response
              over the flat control (`lensing_sky_report`).

The squeezed prediction (`lensing_mean_prediction`) uses the estimator of
lensing_qe_sims, read from one of its seeds and its index file, under the
1500d field mask with the apodization decision's taper.

Run:  bash env.sh python3 analysis/lensing_summary.py
          --lensing-qe-sims results/fiducial/lensing_qe_sims
          --lensing-qe-resolution results/fiducial/lensing_qe_resolution
          --lensing-qe-sky-response results/fiducial/lensing_qe_sky_response
          --theory-spectra data/theory_spectra.npz
          --apodization c2_2p5deg --out <dir>
Out:  <out>/lensing_summary.json ({"qe", "resolution", "sky"}),
      <out>/mean_prediction.npz, <out>/summary.npz (ensemble-mean mean fields),
      and the figures response.png, mean_fields.png and sky_response.png,
      which the paper does not include.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import spectra as S
from lensing_mean_prediction import predict
import lensing_report
import lensing_resolution_report
import lensing_sky_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stages"))
from _decisions import parse_apodization  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "lensing_qe_sims", "lensing_qe_resolution",
                 "lensing_qe_sky_response", "theory_spectra")
    ap.add_argument("--apodization", type=parse_apodization, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index = json.loads(next(args.lensing_qe_sims.glob("index_*.json")).read_text())
    with np.load(args.lensing_qe_sims / index["file"]) as seed:
        shape, pixel = tuple(seed["shape"]), float(seed["pixel_rad"])
    prediction = predict(args.theory_spectra, args.apodization, shape, pixel,
                         index["lmin"], index["lmax"])
    np.savez_compressed(args.out / "mean_prediction.npz", **prediction)

    summary = {
        "qe": lensing_report.report(args.lensing_qe_sims, prediction, args.out),
        "resolution": lensing_resolution_report.report(
            args.lensing_qe_sims, args.lensing_qe_resolution, prediction),
        "sky": lensing_sky_report.report(args.lensing_qe_sky_response, args.out),
    }
    (args.out / "lensing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
