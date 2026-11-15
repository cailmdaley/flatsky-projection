"""Section 5 geometry macros: shear of literature charts and footprint responses.

  gSq<chart>, bbEe<chart>                  <g^2> and 2<g^2> of the literature
                                           charts and two HEALPix faces
  anis<Sky|Grid><footprint><proj>...       response of an anisotropic sky (a
                                           local scan quadrupole, or a filter
                                           on the output grid) on the sweep's
                                           reference footprint, the 1500d
                                           field (the same rectangle) and its
                                           east half
  beam<footprint><proj>Pred{Mean,Min,Max}  sky-mode response of a source-grid
                                           1 arcmin Gaussian at ell = 3000
  scanQuadrupole{OneK,MidK,ThreeK}         quadrupole of the toy scan filter

The footprints are the analytic production masks, shaped by `_footprints` and
apodized as the `healpix_masks` index files record.

Run:  bash env.sh python3 analysis/paper_numbers/section5_numbers.py
          --healpix-masks <healpix_masks dir> --out <dir>
Out:  <out>/section5_numbers.tex and <out>/section5_numbers.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import section5_geometry as G5  # noqa: E402
import spectra as S  # noqa: E402
from macros import Macros, f, pct, sci  # noqa: E402


def macros(healpix_masks):
    M = Macros()
    footprints = G5.literature_footprints()
    footprints["HEALPixPolarFace"] = G5.healpix_face_g2()
    footprints["HEALPixEquatorialFace"] = G5.healpix_face_g2(face=4)
    for name, g2 in footprints.items():
        M.add(f"gSq{name}", sci(g2, 1))
        M.add(f"bbEe{name}", sci(2*g2, 1))
    names = {'reference': 'Disk', 'field1500d': 'Field', 'fieldEast': 'East'}
    for attachment in ('sky', 'grid'):
        responses = G5.footprint_response(healpix_masks, attachment=attachment)
        for (footprint, projection), (g2, zero, diagonal) in responses.items():
            prefix = f"anis{attachment.title()}{names[footprint]}{projection[4:]}"
            M.add(prefix + "GSq", sci(g2, 1))
            M.add(prefix + "BBEE", sci(2*g2, 1))
            M.add(prefix + "Aligned", pct(zero, 2))
            M.add(prefix + "Diagonal", pct(diagonal, 1))
    for (footprint, projection), (sphere, azimuth, low, high) in G5.beam_footprints(healpix_masks).items():
        prefix = f"beam{names[footprint]}{projection[4:]}"
        M.add(prefix + "PredMean", f(azimuth, 5))
        M.add(prefix + "PredMin", f(low, 5))
        M.add(prefix + "PredMax", f(high, 5))
    M.add("beamSpherePred", f(sphere, 5))
    for ell, name in ((1000, 'OneK'), (1500, 'MidK'), (3000, 'ThreeK')):
        M.add('scanQuadrupole'+name, f(G5.scan_filter_a2(ell), 3))
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "healpix_masks")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    macros(args.healpix_masks).write(args.out, "section5_numbers")
