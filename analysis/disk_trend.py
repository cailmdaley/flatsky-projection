"""The W^2-weighted concentric-disk trend, over the actual seven paper radii.

The felt fiber `science/spt/pol-project-paper/tt-scalar-convexity/derivation.md`
carries a table of predicted TT ratio vs. disk radius that was built two
different ways in two different places, and they disagreed on the
radius <-> <g^2> mapping (e.g. the note's R=0.475 rad -> <g^2>=0.0337, versus
`plot_crawford_comparison.disk_geometry`'s R=0.454 rad -> <g^2>=0.0399 -- a
smaller radius giving a *larger* shear variance, which is not possible for a
monotonic quantity and is the tell that something in the construction, not
the physics, was off).

Both earlier computations shared the same two shortcuts, named in the note's
own "Files" footnote: the disk was a *hard-edged, uniform-weighted* cut, not
the actual apodized C2/2.5deg mask this project uses everywhere else, and the
disk was assembled from a flat (Delta-alpha, Delta-delta) Euclidean grid --
`plot_crawford_comparison.disk_geometry` explicitly builds `ang = sqrt(dalpha^2
+ ddelta^2)` and cuts at `ang <= radius_deg`, which is a Euclidean disk in an
oblique tangent-plane coordinate, not a true spherical cap, and at only
ngrid=260 (~53k interior pixels for the largest disk). A hard, unapodized edge
makes the disk average of g^2 dominated by the outermost ring of pixels kept
-- exactly where g is largest and growing fastest with radius -- so which
pixels survive the ngrid=260 quantization at a given nominal radius moves the
answer by tens of percent. That is the discrepancy: not a physics
disagreement, a discretization-noise problem intrinsic to averaging an
unapodized disk uniformly, made worse by comparing two different flat-grid
constructions at two different nominal radii that do not quite match.

This script removes both shortcuts. The disk is `make_masks.apodize_disk_analytic`
run on a real Healpix grid -- the same closed-form C2 taper the pipeline
applies to every mask in the actual analysis -- so the edge is soft and the
"which pixel is inside" question no longer dominates. <g^2> and the predicted
ratio are averaged with weight W^2, exactly as `predicted_spectrum` expects
(NaMaster's pseudo-Cl estimator weights the map by the mask, so power carries
mask^2). The disk is a true spherical cap (`hp.query_disc`), not a flat
Euclidean approximation.

Run inside the project container:
    systemd-run --user --scope -p MemoryMax=4G bash env.sh python3 analysis/disk_trend.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src" / "stages"))

import crawford_model as CMOD   # noqa: E402
import geometry as G            # noqa: E402
from make_masks import apodize_disk_analytic  # noqa: E402

import healpy as hp

OUT_DIR = HERE / "results_disk_trend"

# The actual seven-disk paper suite (src/stages/_project.py / the astra.yaml
# mask stage), not a synthetic sweep -- these are the radii whose spectra are
# measured and compared to the law in Section~\ref{sec:results}.
DISK_RADII_RAD = [0.175, 0.2, 0.225, 0.25, 0.35, 0.45, 0.65]

CENTER_RA_DEG = 0.0
CENTER_DEC_DEG = -56.0
APOTYPE = "C2"
APOSIZE_DEG = 2.5
NSIDE = 2048  # 1.7' pixels: >30x finer than the 2.5 deg taper, cheap at this radius range

ELL_TARGETS = [3000.0, 4000.0, 5000.0, 6000.0]


def center_vec():
    return hp.ang2vec(CENTER_RA_DEG, CENTER_DEC_DEG, lonlat=True)


def disk_result(radius_rad, ell_th, cl_th):
    w_full = apodize_disk_analytic(NSIDE, center_vec(), radius_rad, APOTYPE, APOSIZE_DEG)
    pix = np.nonzero(w_full > 0)[0]
    w = w_full[pix].astype(float)
    del w_full

    lon, lat = hp.pix2ang(NSIDE, pix, lonlat=True)  # degrees

    a = G.relative_jacobian(
        "ProjSFL", "ProjZEA", lon, lat,
        to_kw=dict(delta_center_deg=CENTER_DEC_DEG),
        from_kw=dict(delta_center_deg=CENTER_DEC_DEG))
    s1, s2 = G.singular_values(a)
    g = (s1 - s2) / (s1 + s2)

    w2 = w ** 2
    g2 = float(np.average(g ** 2, weights=w2))

    ell = np.asarray(ELL_TARGETS)
    proj0 = G.predicted_spectrum(ell, s1, s2, w2, ell_th, cl_th,
                                 band_limit=CMOD.BAND_LIMIT, n_bins=64, n_psi=96)
    proj5 = np.interp(ell, ell_th, cl_th)
    ratio = proj0 / proj5

    return {
        "radius_rad": radius_rad,
        "n_pix": int(pix.size),
        "rms_g": float(np.sqrt(g2)),
        "g2": g2,
        "ratio_3000": float(ratio[0]),
        "ratio_4000": float(ratio[1]),
        "ratio_5000": float(ratio[2]),
        "ratio_6000": float(ratio[3]),
    }


def main():
    ell_th, cl_th = CMOD.theory_input("tt")

    rows = []
    for r in DISK_RADII_RAD:
        row = disk_result(r, ell_th, cl_th)
        rows.append(row)
        print(f"R={r:.3f} rad  n_pix={row['n_pix']:>9d}  rms_g={row['rms_g']:.4f}  "
             f"<g^2>={row['g2']:.4f}  ratio(3000,4000,5000,6000) = "
             f"{row['ratio_3000']:.3f} {row['ratio_4000']:.3f} "
             f"{row['ratio_5000']:.3f} {row['ratio_6000']:.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    meta = {
        "nside": NSIDE, "apotype": APOTYPE, "aposize_deg": APOSIZE_DEG,
        "center_ra_deg": CENTER_RA_DEG, "center_dec_deg": CENTER_DEC_DEG,
        "weighting": "W^2 (apodized C2/2.5deg disk, analytic taper)",
        "projections": "SFL relative to ZEA (relative_jacobian), both at delta_center=-56",
        "spectrum": "crawford_model.theory_input('tt') -- lensed Planck 2018 base_plikHM_TTTEEE",
        "ell_band_limit": CMOD.BAND_LIMIT,
        "rows": rows,
    }
    (OUT_DIR / "disk_trend.json").write_text(json.dumps(meta, indent=2))

    with open(OUT_DIR / "disk_trend.csv", "w") as f:
        f.write("radius_rad,n_pix,rms_g,g2,ratio_3000,ratio_4000,ratio_5000,ratio_6000\n")
        for row in rows:
            f.write(f"{row['radius_rad']},{row['n_pix']},{row['rms_g']:.6f},"
                   f"{row['g2']:.6f},{row['ratio_3000']:.6f},{row['ratio_4000']:.6f},"
                   f"{row['ratio_5000']:.6f},{row['ratio_6000']:.6f}\n")
    print(f"\nwrote {OUT_DIR / 'disk_trend.json'} and disk_trend.csv")


if __name__ == "__main__":
    main()
