"""Section 5 geometry macros, paper/section5_numbers.tex.

Not yet a recipe: reads its inputs from fixed paths and writes paper/section5_numbers.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectra as S  # noqa: E402,F401
from macros import WORDS as _WORDS, Macros, f, pct, rng, sci  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]


def section5_numbers():
    """Compute Section 5 geometry macros from specified footprint charts."""
    import section5_geometry as G5

    M = Macros()
    footprints = G5.literature_footprints()
    footprints["HEALPixPolarFace"] = G5.healpix_face_g2()
    footprints["HEALPixEquatorialFace"] = G5.healpix_face_g2(face=4)
    summary={'footprints':footprints,'anisotropic':{},'beam_sky_prediction':{},
             'parameters':{'a2':.3,'n':-3.,'fwhm_arcmin':1.,'beam_ell':3000.,
                           'mask_quadrature_nside':128,'face_nside':2048}}
    for name, g2 in footprints.items():
        M.add(f"gSq{name}", sci(g2, 1))
        M.add(f"bbEe{name}", sci(2*g2, 1))
    names = {'disk5': 'Disk', 'field1500d': 'Field', 'fieldEast': 'East'}
    for attachment in ('sky','grid'):
        responses = G5.footprint_response(verbose=False,attachment=attachment)
        for (footprint, projection), (g2, zero, diagonal) in responses.items():
            summary['anisotropic'][f'{attachment}_{footprint}_{projection}']={
                'g2':float(g2),'chi_zero':float(zero),'chi_diagonal':float(diagonal)}
            prefix = f"anis{attachment.title()}{names[footprint]}{projection[4:]}"
            M.add(prefix + "GSq", sci(g2, 1))
            M.add(prefix + "BBEE", sci(2*g2, 1))
            M.add(prefix + "Aligned", pct(zero, 2))
            M.add(prefix + "Diagonal", pct(diagonal, 1))
    for (footprint, projection), (sphere, azimuth, low, high) in G5.beam_footprints(verbose=False).items():
        summary['beam_sky_prediction'][f'{footprint}_{projection}']={
            'sphere':float(sphere),'mean':float(azimuth),'min':float(low),'max':float(high)}
        prefix = f"beam{names[footprint]}{projection[4:]}"
        M.add(prefix + "PredMean", f(azimuth, 5))
        M.add(prefix + "PredMin", f(low, 5))
        M.add(prefix + "PredMax", f(high, 5))
    M.add("beamSpherePred", f(sphere, 5))
    for ell,name in ((1000,'OneK'),(1500,'MidK'),(3000,'ThreeK')):
        M.add('scanQuadrupole'+name,f(G5.scan_filter_a2(ell),3))
    assert all(name.isalpha() for name in M), 'LaTeX control sequences must be alphabetic'
    path = ROOT / "paper" / "section5_numbers.tex"
    path.write_text(M.tex('section5_numbers'))
    (ROOT/'paper'/'section5_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f"Wrote {path}")


if __name__ == "__main__":
    section5_numbers()
