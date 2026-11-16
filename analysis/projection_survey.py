# /// script
# dependencies = ["pyproj", "numpy", "scipy", "matplotlib"]
# ///
"""Survey of map projections for the SPT 1500d field (RA -50..50, Dec -70..-42).

For each PROJ projection, the sky is first rotated so the field centre sits at a free
(lon, lat) of the projection's frame with a free heading (any oblique or
transverse aspect) and the projection's own shape parameters are
free (standard parallels, perspective height, ...).  Each projection is
optimised twice: once for the footprint mean of g^2 (E->B leakage 2<g^2>), once
for the variance of ln mu (scalar magnification spread).  The Jacobian is
taken by central differences in a local east/north frame on the unit sphere,
and the footprint is sampled uniformly in area.

Two conformal optima not in PROJ are computed directly in Mercator
coordinates z, where any conformal map has ln k = ln sec(phi) + h with h
harmonic: the least-squares one (h fitted with harmonic polynomials) and
Chebyshev's minimax one (h = -ln sec(phi) on the boundary, Laplace solve).

The field and its centre come from `src/stages/_footprints.py`. The script
needs PROJ (pyproj) and none of the pipeline's data, so it runs under uv
from its inline dependencies rather than in the SPT environment.

Run:  uv run analysis/projection_survey.py --out <dir>
Out:  <out>/projection_survey.json (`field`: each projection's two optima;
      `paper`: the paper's four at their fixed definitions; `opt`: the two
      conformal optima) and a diagnostic plot, projection_survey.{pdf,png},
      that the paper does not include.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.optimize import minimize
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stages"))
from _footprints import DEC_DEG, DEC_HALF_DEG, RA_DEG, RA_HALF_DEG  # noqa: E402

D2R = np.pi / 180
FIELD = dict(lon=(RA_DEG - RA_HALF_DEG, RA_DEG + RA_HALF_DEG),
             lat=(DEC_DEG - DEC_HALF_DEG, DEC_DEG + DEC_HALF_DEG))
CENTRE_LAT = DEC_DEG

PAPER = {  # the paper's four, at their fixed definitions
    "SFL": dict(proj="sinu"),
    "STG": dict(proj="stere", lat_0=CENTRE_LAT),
    "ZEA": dict(proj="laea", lat_0=CENTRE_LAT),
    "BICEP": dict(proj="eqc", lat_ts=CENTRE_LAT),
}

# shape parameters each family is optimised over (start values; R=1 always)
SHAPE = {
    **{p: dict(lat_1=-47.0, lat_2=-65.0) for p in
       ["aea", "lcc", "eqdc", "tissot", "murd1", "murd2", "murd3", "euler",
        "vitk1", "pconic", "imw_p"]},
    **{p: dict(lat_1=-56.0) for p in ["bonne", "ccon", "leac", "loxim", "wink1", "wink2"]},
    **{p: dict(lat_ts=-56.0) for p in ["eqc", "cea", "rpoly", "wag3"]},
    **{p: dict(lat_0=-56.0) for p in ["stere", "laea", "aeqd", "gnom", "ortho", "sterea", "lcca"]},
    "nsper": dict(lat_0=-56.0, h=3.0),
    "tpers": dict(lat_0=-56.0, h=3.0, tilt=0.0, azi=0.0),
    "lagrng": dict(W=2.0, lat_1=-56.0),
    "gn_sinu": dict(m=2.0, n=3.0),
    "urmfps": dict(n=0.5),
}
NOT_PROJECTIONS = set("""affine axisswap cart defmodel deformation geoc geocent geogoffset
gridshift helmert hgridshift horner latlon latlong lonlat longlat molodensky noop ob_tran
pipeline pop push set tinshift topocentric unitconvert vertoffset vgridshift xyzgridshift
s2 misrsom lsat som calcofi nzmg krovak mod_krovak alsk gs48 gs50 isea qsc geos ups utm
webmerc tobmerc airocean spilhaus""".split())
CONFORMAL = {"stere", "lcc", "merc", "tmerc", "etmerc", "lagrng", "sterea", "eqearth_no",
             "ups", "somerc", "gstmerc", "omerc", "rouss", "adams_hemi", "adams_ws1",
             "adams_ws2", "guyou", "peirce_q", "lee_os", "mil_os", "eisen", "august", "chamb_no"}


def sample(n_lon=48, n_lat=24):
    """Area-uniform grid (uniform in lon and sin lat) over the field."""
    lo = np.array(FIELD["lon"])
    la = np.array(FIELD["lat"])
    s = np.sin(la * D2R)
    u = (np.arange(n_lon) + 0.5) / n_lon
    v = (np.arange(n_lat) + 0.5) / n_lat
    lon = lo[0] + u * (lo[1] - lo[0])
    lat = np.arcsin(s[0] + v * (s[1] - s[0])) / D2R
    L, B = np.meshgrid(lon, lat)
    return L.ravel(), B.ravel()


def rotate(lon, lat, lam_c, phi_c, heading):
    """Move the field centre (0, CENTRE_LAT) to (lam_c, phi_c), turned by `heading` (degrees)."""
    lam, phi = lon * D2R, lat * D2R
    x, y, z = np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)

    def roty(x, z, b):  # lowers the latitude of points on lon=0 by b
        cb, sb = np.cos(b * D2R), np.sin(b * D2R)
        return cb * x + sb * z, -sb * x + cb * z
    x, z = roty(x, z, CENTRE_LAT)
    ct, st = np.cos(heading * D2R), np.sin(heading * D2R)
    y, z = ct * y - st * z, st * y + ct * z
    x, z = roty(x, z, -phi_c)
    return (np.arctan2(y, x) / D2R + lam_c + 180) % 360 - 180, np.arcsin(np.clip(z, -1, 1)) / D2R


def invariants(proj, lon, lat, rot=(0.0, CENTRE_LAT, 0.0), h=1e-4):
    """Singular-value invariants (g, ln mu) of the sky->plane Jacobian at each point."""
    def xy(lo, la):
        return np.array(proj(*rotate(lo, la, *rot)))
    dl = h / np.cos(lat * D2R)
    e = (xy(lon + dl, lat) - xy(lon - dl, lat)) / (2 * h * D2R)
    n = (xy(lon, lat + h) - xy(lon, lat - h)) / (2 * h * D2R)
    det = np.abs(e[0] * n[1] - e[1] * n[0])
    fro = (e ** 2).sum(0) + (n ** 2).sum(0)
    ssum, sdiff = np.sqrt(fro + 2 * det), np.sqrt(np.maximum(fro - 2 * det, 0))
    return sdiff / ssum, np.log(det)


def metrics(g, lnmu):
    return dict(g2=float(np.mean(g ** 2)), gmax=float(np.max(g)),
                rms_lnmu=float(np.std(lnmu)), range_lnmu=float(np.ptp(lnmu)),
                smear=float(np.sqrt(np.var(lnmu) / 4 + np.mean(g ** 2) / 2)))


def build(name, shape):
    return pyproj.Proj(proj=name, R=1, **shape)


def objective(params, name, keys, lon, lat, which):
    rot, shape = params[:3], dict(zip(keys, params[3:]))
    try:
        g, lnmu = invariants(build(name, shape), lon, lat, rot)
    except Exception:
        return 10.0
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(lnmu))):
        return 10.0
    val = np.mean(g ** 2) if which == "g2" else np.var(lnmu)
    return np.log10(val + 1e-14)


STARTS = [(0, CENTRE_LAT, 0), (0, 0, 0), (0, 0, 90), (0, -90, 0), (0, -30, 0), (0, CENTRE_LAT, 90)]


def optimise(name, which):
    shape0 = SHAPE.get(name, {})
    keys = list(shape0)
    lon, lat = sample()
    best = None
    for s in STARTS:
        x0 = np.array([*s, *shape0.values()], float)
        if objective(x0, name, keys, lon, lat, which) >= 10:
            continue
        r = minimize(objective, x0, args=(name, keys, lon, lat, which), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-4, maxiter=1500))
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        return None
    rot, shape = best.x[:3], dict(zip(keys, best.x[3:]))
    g, lnmu = invariants(build(name, shape), *sample(160, 80), rot)
    return dict(rot=[round(float(v), 2) for v in rot],
                shape={k: round(float(v), 3) for k, v in shape.items()},
                **metrics(g, lnmu))


def paper_four():
    out = {}
    for lab, kw in PAPER.items():
        kw = dict(kw)
        g, lnmu = invariants(pyproj.Proj(R=1, **kw), *sample(160, 80))
        out[lab] = metrics(g, lnmu)
    return out


def optimal_conformal(n=161):
    """Least-squares and Chebyshev (minimax) conformal maps of the field.

    Work in Mercator coordinates (x, y); a conformal map has
    ln k = ln sec(phi) + h(x, y) with h harmonic and otherwise free.
    """
    lo = np.array(FIELD["lon"]) * D2R
    la = np.array(FIELD["lat"]) * D2R
    yb = np.log(np.tan(np.pi / 4 + la / 2))
    x = np.linspace(*lo, n)
    y = np.linspace(*yb, n)
    X, Y = np.meshgrid(x, y)
    phi = 2 * np.arctan(np.exp(Y)) - np.pi / 2
    u = -np.log(np.cos(phi))           # ln sec(phi): Mercator's own scale
    w = np.cos(phi) ** 2               # sphere area element per dx dy
    w /= w.sum()

    # least squares: project u onto harmonic polynomials Re/Im (z - z0)^k
    z = (X - X.mean()) + 1j * (Y - Y.mean())
    z /= np.abs(z).max()
    basis = [np.ones_like(X)] + [f((z) ** k) for k in range(1, 15) for f in (np.real, np.imag)]
    A = np.stack([b.ravel() for b in basis], 1) * np.sqrt(w.ravel())[:, None]
    coef = np.linalg.lstsq(A, -u.ravel() * np.sqrt(w.ravel()), rcond=None)[0]
    lnk_ls = u + (np.stack([b.ravel() for b in basis], 1) @ coef).reshape(u.shape)

    # Chebyshev: h = -u on the boundary, harmonic inside (5-point Laplacian)
    hx, hy = x[1] - x[0], y[1] - y[0]
    m = n - 2
    Dx = diags([1, -2, 1], [-1, 0, 1], shape=(m, m)) / hx ** 2
    Dy = diags([1, -2, 1], [-1, 0, 1], shape=(m, m)) / hy ** 2
    Lap = kron(eye(m), Dx) + kron(Dy, eye(m))
    hb = -u.copy()
    rhs = np.zeros((m, m))
    rhs[:, 0] -= hb[1:-1, 0] / hx ** 2
    rhs[:, -1] -= hb[1:-1, -1] / hx ** 2
    rhs[0, :] -= hb[0, 1:-1] / hy ** 2
    rhs[-1, :] -= hb[-1, 1:-1] / hy ** 2
    hb[1:-1, 1:-1] = spsolve(Lap.tocsc(), rhs.ravel()).reshape(m, m)
    lnk_ch = u + hb

    def stats(lnk):
        lnmu = 2 * lnk
        mean = (w * lnmu).sum()
        return dict(g2=0.0, gmax=0.0, rms_lnmu=float(np.sqrt((w * (lnmu - mean) ** 2).sum())),
                    range_lnmu=float(np.ptp(lnmu)),
                    smear=float(np.sqrt((w * (lnmu - mean) ** 2).sum() / 4)))
    return {"conformal-LS": stats(lnk_ls), "conformal-Chebyshev": stats(lnk_ch)}


def candidates():
    names = sorted(set(pyproj.list.get_proj_operations_map()) - NOT_PROJECTIONS)
    lon, lat = sample()
    ok = []
    for nm in names:
        if objective(np.array([0, CENTRE_LAT, 0, *SHAPE.get(nm, {}).values()], float), nm,
                     list(SHAPE.get(nm, {})), lon, lat, "g2") < 10:
            ok.append(nm)
    return ok


def survey(names):
    res = {}
    for nm in names:
        res[nm] = {w: optimise(nm, w) for w in ("g2", "lnmu")}
        print(nm, {w: (r and (round(r["g2"], 6), round(r["rms_lnmu"], 4))) for w, r in res[nm].items()},
              flush=True)
    return res


def best_point(r):
    """A projection's representative optimum: the one with the smaller combined smear."""
    return min((v for v in r.values() if v is not None), key=lambda v: v["smear"])


def plot(res, paper, opt, path):
    fg, fm = 1e-7, 1e-4  # where g^2 = 0 and rms ln mu = 0 are drawn
    clip = lambda r: (max(r["g2"], fg), max(r["rms_lnmu"], fm))
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for nm, r in res.items():
        if any(r.values()):
            ax.scatter(*clip(best_point(r)), s=10, color="0.6", lw=0, zorder=2)
    named = {"lcc": ("Lambert conformal conic, Lagrange", (8, 3)),
             "lagrng": ("", (0, 0)),
             "aea": ("Albers equal-area conic", (-75, 12)),
             "gn_sinu": ("general sinusoidal", (-80, -14)),
             "eqdc": ("equidistant conic", (8, -10)),
             "mill": ("Miller (oblique)", (8, -3))}
    for nm, (lab, off) in named.items():
        xy = clip(best_point(res[nm]))
        ax.scatter(*xy, s=28, color="C2", zorder=4)
        ax.annotate(lab, xy, xytext=off, textcoords="offset points", fontsize=8, color="C2")
    offs = {"SFL": (-14, 9), "STG": (8, 2), "ZEA": (7, -16), "BICEP": (8, -3)}
    for i, (lab, rr) in enumerate(paper.items()):
        c = f"C{[3, 0, 1, 4][i]}"
        ax.scatter(*clip(rr), s=130, marker="*", color=c, edgecolor="k", lw=0.5, zorder=5)
        ax.annotate(lab, clip(rr), xytext=offs[lab], textcoords="offset points", fontsize=11,
                    weight="bold", color=c)
    for nm, rr in opt.items():
        ax.scatter(*clip(rr), s=30, marker="D", color="C5", zorder=5)
    ax.annotate("conformal optima\n(least-squares, Chebyshev)", clip(opt["conformal-LS"]),
                xytext=(8, -26), textcoords="offset points", fontsize=8, color="C5")
    ax.axvline(fg * 2, color="0.4", lw=0.5, ls=":")
    ax.axhline(fm * 2, color="0.4", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(fg / 2, 0.3)
    ax.set_ylim(fm / 2, 1)
    ax.text(fg * 1.15, 0.8, "$g=0$", fontsize=8, ha="center", va="top", color="0.3")
    ax.text(0.2, fm * 1.15, "$\\mu=1$", fontsize=8, ha="right", va="bottom", color="0.3")
    ax.set_xlabel(r"$\langle g^2\rangle$   ($BB/EE$ leakage $= 2\langle g^2\rangle$)")
    ax.set_ylabel(r"rms of $\ln\mu$ over the footprint")
    ax.set_title("Projections of the 1500d field, each in its best aspect", fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=160)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    names = candidates()
    print(len(names), "projections project the field:", " ".join(names), flush=True)
    res = survey(names)
    paper = paper_four()
    opt = optimal_conformal()
    print("paper four:", paper)
    print("optimal conformal:", opt)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "projection_survey.json").write_text(
        json.dumps(dict(field=res, paper=paper, opt=opt), indent=1) + "\n")
    plot(res, paper, opt, args.out / "projection_survey")


if __name__ == "__main__":
    main()
