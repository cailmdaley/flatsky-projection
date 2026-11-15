"""projcoupling against the research forward model on the 1500d field.

`shear_leakage.coupling_matrix` works on the HEALPix mask's pixels (sky
measure) with the Jacobian differenced through spt3g's `angle_to_xy`, and it
carries the residual Delta of spt3g's FlattenPol angle. `projcoupling` works on
the 2 arcmin analysis grid itself, from the sky coordinates of its pixels
(`xy_to_angle`), with the mask evaluated at those pixels, and assumes
flattening by the polar angle (Delta = 0). The research model is run with
Delta switched off (its cell means of sin 2Delta and sin^2 2Delta zeroed) and
on, so the comparison isolates everything else.

Both read the sky at the plane wavenumber itself (k_offset = 0 here; the
research model has no offset), band-limited at l = 6000, and both average
uniformly over k within each band of width 50: the research model on 50
samples at k = lo + 1/2 + i, projcoupling exactly.

Run:  bash env.sh python analysis/validate_projcoupling.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "projcoupling"))
sys.path.insert(0, str(ROOT / "src" / "stages"))
import field_1500d as FD
import make_masks as MM
import shear_leakage as SL
import spectra as S
import theory as T
import projcoupling as pc

from spt3g import core, maps

RES_ARCMIN = 2.0
SHAPE = (1500, 3600)            # (ny, nx): 50 x 120 deg, covers the tapered field
EDGES = np.arange(500, 4501, 50)
L_CMP = (1000, 4000)
BAND_LIMIT = S.LMAX


def grid_radec(proj):
    """RA, dec in degrees of every pixel of a 2' grid centred at (0, -56), NaN off the sphere."""
    ny, nx = SHAPE
    fsm = maps.FlatSkyMap(nx, ny, RES_ARCMIN * core.G3Units.arcmin,
                          proj=getattr(maps.MapProjection, proj),
                          alpha_center=0.0, delta_center=-56.0 * core.G3Units.deg)
    y, x = np.indices(SHAPE)
    ra, dec = fsm.xy_to_angle(core.DoubleVector(x.ravel().astype(float)),
                              core.DoubleVector(y.ravel().astype(float)))
    ra = np.asarray(ra).reshape(SHAPE) / core.G3Units.deg
    dec = np.asarray(dec).reshape(SHAPE) / core.G3Units.deg
    return ra, dec


def field_mask(ra, dec):
    """The pipeline's 1500d mask (`make_masks.rect_mask_analytic`) at given sky positions."""
    import json
    idx = json.loads(FD.MASK_INDEX.read_text())
    rec = idx["masks"][0]
    dra = np.radians((ra - idx["ra_deg"] + 180.0) % 360.0 - 180.0)
    inside = ((np.abs(dra) <= np.radians(rec["ra_half_deg"]))
              & (dec >= rec["dec_min_deg"]) & (dec <= rec["dec_max_deg"]))
    d = np.radians(dec)
    dist = np.minimum(d - np.radians(rec["dec_min_deg"]), np.radians(rec["dec_max_deg"]) - d)
    half = np.radians(rec["ra_half_deg"])
    for edge in (half, -half):
        dist = np.minimum(dist, np.arcsin(np.clip(np.cos(d) * np.abs(np.sin(dra - edge)), 0, 1)))
    w = np.zeros(SHAPE)
    w[inside] = MM.taper(dist[inside], idx["apotype"], idx["aposize_deg"])
    return w


def research(mom, j, ell_th):
    k = (EDGES[:-1, None] + 0.5 + np.arange(50)[None]).ravel()
    R = SL.coupling_matrix(mom, 0, j, k, ell_th, band_limit=BAND_LIMIT)
    return {b: R[b].reshape(len(EDGES) - 1, 50, -1).mean(1) for b in ("EE<-EE", "BB<-EE", "TT<-TT")}


def main():
    th = T.load()
    ell_th = th["ell"].astype(float)
    cl = {"EE": np.where(ell_th > BAND_LIMIT, 0.0, th["unlensed_EE"]),
          "TT": np.where(ell_th > BAND_LIMIT, 0.0, th["unlensed_TT"])}
    mid = 0.5 * (EDGES[:-1] + EDGES[1:])
    sel = (mid > L_CMP[0]) & (mid < L_CMP[1])

    t0 = time.time()
    mom = FD.pipeline_moments()
    print(f"research moments (HEALPix Nside {FD.MOMENT_NSIDE}): {time.time() - t0:.0f} s")
    mom_off = dict(mom, cell_sin2_2delta=np.zeros_like(mom["cell_sin2_2delta"]),
                   cell_sin_2delta=np.zeros_like(mom["cell_sin_2delta"]))

    for proj in ("ProjSFL", "ProjZEA"):
        j = S.PROJS.index(proj)
        ra, dec = grid_radec(proj)
        mask = field_mask(ra, dec)
        border = np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]].max()

        t0 = time.time()
        geo = pc.Geometry.from_radec(ra, dec, np.radians(RES_ARCMIN / 60))
        t_geo = time.time() - t0
        t0 = time.time()
        R = pc.coupling_matrix(geo, mask, EDGES, ell_th, k_offset=0.0)
        t_R = time.time() - t0
        mine = {b: R[b] for b in ("EE<-EE", "BB<-EE", "TT<-TT")}

        print(f"\n{proj}: grid {SHAPE[0]}x{SHAPE[1]}, mask max on grid border {border:.1e}; "
              f"geometry {t_geo:.1f} s, coupling matrix {t_R:.1f} s")
        print(f"  <g^2>: projcoupling {pc.gsq(geo, mask):.6e}   research {mom['g2'][0, j]:.6e}")
        for label, m in (("Delta off", mom_off), ("Delta on ", mom)):
            ref = research(m, j, ell_th)
            ee_m, ee_r = mine["EE<-EE"] @ cl["EE"], ref["EE<-EE"] @ cl["EE"]
            bb_m, bb_r = mine["BB<-EE"] @ cl["EE"], ref["BB<-EE"] @ cl["EE"]
            tt_m, tt_r = mine["TT<-TT"] @ cl["TT"], ref["TT<-TT"] @ cl["TT"]
            r_bbee = (bb_m / ee_m) / (bb_r / ee_r) - 1
            r_ee = ee_m / ee_r - 1
            r_tt = tt_m / tt_r - 1
            print(f"  research {label}, {L_CMP[0]} < l < {L_CMP[1]}, projcoupling/research - 1:")
            for name, r in (("BB/EE", r_bbee), ("EE   ", r_ee), ("TT   ", r_tt)):
                print(f"    {name}  mean {r[sel].mean():+.2e}  min {r[sel].min():+.2e}  max {r[sel].max():+.2e}")
        print(f"  BB/EE (projcoupling) at l = 1025, 2025, 3025, 3975: "
              + "  ".join(f"{(mine['BB<-EE'] @ cl['EE'] / (mine['EE<-EE'] @ cl['EE']))[np.argmin(abs(mid - L))]:.4e}"
                          for L in (1025, 2025, 3025, 3975)))

        for n_cells in (64, 256):
            Rn = pc.coupling_matrix(geo, mask, EDGES, ell_th, k_offset=0.0, n_cells=n_cells)
            d = (Rn["BB<-EE"] @ cl["EE"]) / (mine["BB<-EE"] @ cl["EE"]) - 1
            e = (Rn["EE<-EE"] @ cl["EE"]) / (mine["EE<-EE"] @ cl["EE"]) - 1
            print(f"  n_cells {n_cells} vs 128: BB max |d| {abs(d[sel]).max():.1e}, EE max |d| {abs(e[sel]).max():.1e}")
        sub = pc.Geometry(geo.s1[::4, ::4], geo.s2[::4, ::4])
        Rs = pc.coupling_matrix(sub, mask[::4, ::4], EDGES, ell_th, k_offset=0.0)
        d = (Rs["BB<-EE"] @ cl["EE"]) / (mine["BB<-EE"] @ cl["EE"]) - 1
        print(f"  every 4th pixel per axis vs all: BB max |d| {abs(d[sel]).max():.1e}")


if __name__ == "__main__":
    main()
