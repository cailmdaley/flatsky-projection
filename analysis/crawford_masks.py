"""Rebuild Crawford's three 1500d masks, exactly as his script makes them.

His generating script is
`/sptlocal/user/tcrawfor/code/spt3g_software/scratch/tcrawfor/proj0_distortion_check_cmbtt_24jul26.py`
(and a single-configuration ancestor dated 20 July). Everything here is a
transcription of it, so the masks this module returns are the masks his
bandpowers were measured through -- not a guess at their shape.

The three variants
------------------
All three start from the real SPT-3G 150 GHz winter apodization mask, which is
a 12000 x 18000 ZEA map at 0.25 arcmin centred on (0, -57.5 deg).

  full        the mask reprojected onto both target grids, untouched.
  cutcorner   the proj0 mask with every pixel outside |x| < 25 deg zeroed.
  iso_shrink  the proj0 mask shrunk about the grid centre by 1.4 in both
              axes, row by row.

Two things about the variants matter for interpretation and are easy to miss
by reading the plots alone.

First, "cut corner" is a misnomer for what the code does. It is a straight cut
in the SFL *x* coordinate, and SFL has x = (alpha - alpha_0) cos(delta), so the
cut removes large RA offsets -- which is precisely where SFL's shear lives,
since gamma = (alpha - alpha_0) sin|delta|. It is a shear cut wearing a corner
cut's name, and it should therefore reduce the projection excess far more than
its area cost suggests.

Second, both variants zero pixels with a hard edge, destroying the apodization
along the new boundary. Crawford's estimator (`quick_flatsky_routines.cl_flatsky`)
corrects only by <mask^2> and does no mode decoupling, so a hard edge couples
modes across the whole band. That is a property of the variants, not of the
projection, and it does not apply to the full-mask numbers.
"""

import numpy as np
from pathlib import Path

from spt3g import core, maps

DEG = core.G3Units.deg
ARCMIN = core.G3Units.arcmin

APOD_MASK = ("/sptlocal/analysis/pointsources19-23/"
             "150GHz_winter_2019-2023_tau_deconv_ptsrc_rms_outfiles/"
             "apodmask_150GHz_winter_2019-2023_tau_deconv_ptsrc_rms.g3")

CACHE = Path("/sptlocal/user/cailmdaley/pol_project/crawford_repro")

# proj5 is ZEA at the production field centre; proj0 is SFL at the 2021 one.
# The 1.5 deg difference is his, and it is inherited from two different
# conventions in spt3g_software rather than chosen.
DELTA_CENTER_PROJ5 = -57.5
DELTA_CENTER_PROJ0 = -56.0
ALPHA_CENTER = 0.0

FAC_SHRINK = 1.4
VARIANTS = ("full", "cutcorner", "iso_shrink")


def _grid(ngrid, res_arcmin, proj, delta_center_deg):
    return maps.FlatSkyMap(
        ngrid, ngrid, res=res_arcmin * ARCMIN, weighted=False,
        alpha_center=ALPHA_CENTER * DEG, delta_center=delta_center_deg * DEG,
        proj=getattr(maps.MapProjection, proj))


def _cut_corner(arr, ngrid, res_arcmin):
    """His cut: zero x-columns outside [1000, 4000] arcmin from the left edge.

    Written as `int(1000 * 1. / reso_arcmin)`, which is a pixel count that
    corresponds to the same *angle* at any resolution, so the cut is
    resolution-invariant: it keeps |x| <= 25 deg on a 4166 arcmin grid.
    """
    arr[:, 0:int(1000 * 1.0 / res_arcmin)] = 0.0
    arr[:, int(4000 * 1.0 / res_arcmin):] = 0.0
    return arr


def _iso_shrink(arr, ngrid, fac_shrink=FAC_SHRINK):
    """His shrink: halve the area by pulling every edge in by `fac_shrink`.

    Row by row, so it follows the mask's own outline rather than shrinking a
    bounding box. The >0.95 threshold means it measures the *unapodized*
    extent and then cuts hard, which is why the result has no taper.
    """
    orig = arr.copy()
    slice1 = orig[:, int(ngrid / 2)]
    ngx = len(slice1[slice1 > 0.95])
    hw = int(np.floor(ngx / 2.0 / fac_shrink))
    for q in np.arange(ngrid):
        if np.abs(q - ngrid / 2) > hw:
            arr[q, :] = 0.0
        else:
            row = arr[q, :].copy()
            ng = len(row[row > 0.95])
            hw2 = int(np.floor(ng / 2.0 / fac_shrink))
            row[0:int(ngrid / 2 - hw2)] = 0.0
            row[int(ngrid / 2 + hw2):] = 0.0
            arr[q, :] = row
    return arr


def build(variant, ngrid, res_arcmin, orig=None):
    """(mask_proj5, mask_proj0) as numpy arrays, per his construction order.

    For the two variants he edits the *proj0* mask and then reprojects it back
    onto the proj5 grid, so both spectra see the same sky region. That second
    reprojection is nearest-neighbour like everything else in his script.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if orig is None:
        orig = core.G3File(APOD_MASK).next()["T"]

    m5 = _grid(ngrid, res_arcmin, "ProjZEA", DELTA_CENTER_PROJ5)
    maps.reproj_map(orig, m5)
    m0 = _grid(ngrid, res_arcmin, "ProjSFL", DELTA_CENTER_PROJ0)
    maps.reproj_map(orig, m0)

    if variant != "full":
        arr = np.array(np.asarray(m0), dtype=float)
        arr = (_cut_corner(arr, ngrid, res_arcmin) if variant == "cutcorner"
               else _iso_shrink(arr, ngrid))
        m0 = maps.FlatSkyMap(
            np.ascontiguousarray(arr), res=res_arcmin * ARCMIN, weighted=False,
            alpha_center=ALPHA_CENTER * DEG,
            delta_center=DELTA_CENTER_PROJ0 * DEG,
            proj=maps.MapProjection.ProjSFL)
        m5 = _grid(ngrid, res_arcmin, "ProjZEA", DELTA_CENTER_PROJ5)
        maps.reproj_map(m0, m5)

    return np.array(np.asarray(m5)), np.array(np.asarray(m0)), m0


def cached_arrays(variant, ngrid, res_arcmin):
    """The two full 2D masks, memoized. What the coupling kernel needs.

    Stored float32: the masks are apodization weights in [0, 1] and nothing
    downstream needs more than 7 digits, while float64 would double a 100 MB
    file for nothing.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"maskarr_{variant}_n{ngrid}_r{res_arcmin:g}.npz"
    if path.exists():
        try:
            d = np.load(path)
            return d["m5"], d["m0"]
        except (EOFError, ValueError, OSError):
            # A run killed mid-write leaves a truncated npz that loads as an
            # empty archive. Rebuild rather than making every caller defensive.
            path.unlink(missing_ok=True)
    m5, m0, _ = build(variant, ngrid, res_arcmin)
    m5, m0 = m5.astype(np.float32), m0.astype(np.float32)
    # The suffix must stay .npz: np.savez silently appends it otherwise, and
    # the rename below would then chase a file that was never written.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, m5=m5, m0=m0)
    tmp.replace(path)   # atomic, so an interrupted write leaves no cache at all
    return m5, m0


def cached(variant, ngrid, res_arcmin):
    """As `build`, memoized to disk; also returns the proj0 sky coordinates.

    The (ra, dec) of every kept proj0 pixel is what the forward model needs,
    and `get_ra_dec_map` on a 10000^2 grid is not free, so it is cached too.
    Only pixels with mask > 0 are stored, which is a ~2x saving and all the
    model ever looks at.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"masks_{variant}_n{ngrid}_r{res_arcmin:g}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}

    m5, m0, m0_frame = build(variant, ngrid, res_arcmin)
    ra, dec = maps.get_ra_dec_map(m0_frame)
    ra = np.asarray(ra) / DEG
    dec = np.asarray(dec) / DEG
    ra = np.where(ra > 180.0, ra - 360.0, ra)

    keep = m0 > 0
    out = dict(
        w0=m0[keep].astype(np.float32),
        ra0=ra[keep].astype(np.float32),
        dec0=dec[keep].astype(np.float32),
        fsky5=np.array(np.mean(m5 ** 2)),
        fsky0=np.array(np.mean(m0 ** 2)),
        # Fraction of the square grid the mask covers at all, for sanity.
        cover0=np.array(np.mean(keep)),
    )
    np.savez_compressed(path, **out)
    return out
