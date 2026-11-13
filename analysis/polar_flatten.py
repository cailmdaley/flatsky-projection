"""Flattening by the polar angle: FlattenPol's residual is a choice.

FlattenPol rotates Q/U by Eq. (rot_angle), the equal-weight average of the
north-based and east-based angles. The polar decomposition J = R S of the
local Jacobian says the rotation to remove is R's, and FlattenPol misses it by
Delta = (g^2/2) sin 4beta + O(g^3) (`shear_leakage.flattening_residual`).
This module shows Delta is avoidable and measures what avoiding it buys.

The closed form
---------------
For a 2x2 Jacobian [[a, b], [c, d]] with det > 0, R(phi)^T J is symmetric iff
(a + d) sin phi = (c - b) cos phi, and the branch with positive trace makes
the symmetric factor positive definite, so phi = atan2(c - b, a + d) exactly.
FlattenPol works with the derivatives of the inverse map, (alpha, delta) by
pixel (x, y), whose polar rotation is the inverse of J's. Measuring the alpha
derivatives in arc length (times cos delta), the spin-2 angle FlattenPol
applies becomes

    rot_R = 2 atan2(-dd/dx - cos(delta) da/dy, dd/dy - cos(delta) da/dx),

against FlattenPol's rot = atan2(-da/dx, da/dy) + atan2(-dd/dy, -dd/dx).
In complex form, with n = d(theta)/dx + i d(theta)/dy (theta the colatitude)
and e = sin(theta) [d(phi)/dy - i d(phi)/dx], the polar angle is
arg[i (n - e)] and FlattenPol's is arg[i (n/|n| - e/|e|)]: the same
expression with both gradients normalised. They agree exactly when the two
gradients have equal length or are orthogonal, i.e. when the principal axes
lie along the sky diagonals or the sky axes -- the zeros of sin 4beta.

What runs here
--------------
`check_closed_form` checks phi against scipy.linalg.polar at random points of
every projection, the normalised-gradient identity for FlattenPol's angle, and
that rot_R built from spt3g's own `xy_to_angle` (the arithmetic of the
proposed C++ patch, analysis/notes/flattenpol-pr/) leaves no residual.

`map_test` takes the production QU-leg maps (flattened by the stock
FlattenPol), unflattens them with `flatten_pol(invert=True)`, reflattens with
rot_R in python, and measures both on (i) the 5% disk through the production
w22 workspace -- BB/EE against the forward model with Delta set to zero -- and
(ii) SPT's 1500d field and its two halves about the central meridian, from
purified coupled pseudo-spectra, where EB/EE from the E-only sky is the TB/TE
pattern's spin-2 twin: a frame rotation by Delta turns E into B by sin 2Delta,
so EB/EE carries (1/2) sin 4Delta where TB/TE carries sin 2Delta.

Run:  systemd-run --user --scope -p MemoryMax=12G bash env.sh python3 analysis/polar_flatten.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as G
import paper_style as PS
import shear_leakage as SL
import spectra as S
import theory as T
from template_geometry import (  # noqa: F401  (re-exported for callers)
    APOD_DEG, FIELD, _c2_taper, _wrap, field_masks, flatten_angles)

from spt3g import core, maps

DEG = core.G3Units.deg
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "highell"
CACHE = SL.CACHE_DIR / "polar_flatten_maps.npz"
FIGURE = ROOT / "analysis" / "notes" / "flattenpol-pr" / "flattenpol_polar.png"
MASK = 5                       # 5% disk, the paper's reference footprint
PAD_DEG = 4.0
L_CLEAN = (1000, 4000)


def check_closed_form(n=4000, seed=1):
    """Closed form vs scipy.linalg.polar, and the spt3g-side residuals, per projection.

    Points uniform in RA -60..60, dec -80..-35, which covers every footprint
    in the paper. Returns {proj: dict} with, in degrees: max |phi - scipy|,
    max asymmetry of R(phi)^T J (relative), max |FlattenPol - normalised
    form|, max |Delta| for FlattenPol and for rot_R, and max g.
    """
    from scipy.linalg import polar
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-60, 60, n)
    delta = rng.uniform(-80, -35, n)
    out = {}
    for proj in S.PROJS:
        jac = G.jacobian(proj, alpha, delta)
        a, b, c, d = jac[:, 0, 0], jac[:, 0, 1], jac[:, 1, 0], jac[:, 1, 1]
        phi = np.arctan2(c - b, a + d)
        err, asym = np.zeros(n), np.zeros(n)
        for k in range(n):
            R, P = polar(jac[k])
            err[k] = abs(_wrap(phi[k] - np.arctan2(R[1, 0], R[0, 0])))
            Rc = np.array([[np.cos(phi[k]), -np.sin(phi[k])],
                           [np.sin(phi[k]), np.cos(phi[k])]])
            M = Rc.T @ jac[k]
            asym[k] = abs(M[0, 1] - M[1, 0]) / np.abs(M).max()

        fsm = maps.FlatSkyMap(3300, 3300, 2.0 * core.G3Units.arcmin,
                              proj=getattr(maps.MapProjection, proj),
                              alpha_center=0.0, delta_center=-56 * DEG)
        x, y = fsm.angle_to_xy(core.DoubleVector(alpha * DEG),
                               core.DoubleVector(delta * DEG))
        x, y = np.asarray(x), np.asarray(y)
        rot, rot_r, _, dc = flatten_angles(fsm, x, y)

        # FlattenPol's angle as the normalised-gradient form arg[i(n/|n| - e/|e|)].
        h = 1e-3
        def ang(xx, yy):
            aa, dd = fsm.xy_to_angle(core.DoubleVector(xx), core.DoubleVector(yy))
            return np.asarray(aa), np.asarray(dd)
        a1, d1 = ang(x - h, y); a2, d2 = ang(x + h, y)
        a3, d3 = ang(x, y - h); a4, d4 = ang(x, y + h)
        th_x, th_y = -(d2 - d1) / (2 * h), -(d4 - d3) / (2 * h)
        ph_x, ph_y = _wrap(a2 - a1) / (2 * h), _wrap(a4 - a3) / (2 * h)
        nvec = th_x + 1j * th_y
        evec = np.cos(dc) * (ph_y - 1j * ph_x)
        norm_form = 2 * np.angle(1j * (nvec / abs(nvec) - evec / abs(evec)))
        polar_form = 2 * np.angle(1j * (nvec - evec))

        half = lambda v: (v + np.pi / 2) % np.pi - np.pi / 2
        s1, s2 = G.singular_values(jac)
        out[proj] = dict(
            scipy=np.degrees(err.max()),
            asym=asym.max(),
            normalised=np.degrees(np.abs(_wrap(norm_form - rot)).max()),
            complex_polar=np.degrees(np.abs(_wrap(polar_form - rot_r)).max()),
            delta_stock=np.degrees(np.abs(half(rot / 2 + phi)).max()),
            delta_polar=np.degrees(np.abs(half(rot_r / 2 + phi)).max()),
            g_max=float(((s1 - s2) / (s1 + s2)).max()),
        )
    return out


def field_geometry(fsm, proj="ProjSFL"):
    """Per-pixel flattening angles on `fsm`'s grid and the 1500d geometry.

    Returns the pol_conv sign, the stock-to-polar factor exp(i s (rot_R - rot))
    and Delta = (rot - rot_R)/2 on the full grid, and on the field's bounding
    box (plus a pad) the field masks, the box slices, its side lengths in
    radians, and the singular values (s1, s2) of the sky Jacobian per pixel.
    """
    ny, nx = fsm.shape
    sign = -1.0 if fsm.pol_conv == maps.MapPolConv.COSMO else 1.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    rot, rot_r, alpha, delta = (v.reshape(ny, nx) for v in
                                flatten_angles(fsm, xx.ravel(), yy.ravel()))
    del xx, yy
    fm = field_masks(alpha, delta)
    rows, cols = np.nonzero(fm["1500d"])
    pad = int(round(PAD_DEG * DEG / fsm.res))
    r0, r1 = rows.min() - pad, rows.max() + pad + 1
    c0, c1 = cols.min() - pad, cols.max() + pad + 1
    box = (slice(r0, r1), slice(c0, c1))
    jac = G.jacobian(proj, np.degrees(alpha[box]).ravel(),
                     np.degrees(delta[box]).ravel())
    s1, s2 = (v.reshape(r1 - r0, c1 - c0) for v in G.singular_values(jac))
    return dict(sign=sign, extra=np.exp(1j * sign * _wrap(rot_r - rot)),
                Delta=_wrap(rot - rot_r) / 2,
                masks={k: v[box] for k, v in fm.items()}, box=box,
                lx=(c1 - c0) * fsm.res, ly=(r1 - r0) * fsm.res, s1=s1, s2=s2)


def eb_model(geo, ell, n_chi=64, stride=4):
    """EB and EE per l from an E-only sky on each field mask, from Delta alone.

    Returns {mask: (EB, EE)} in arbitrary common units, one entry per `ell`.

    A plane mode at azimuth chi to the principal axis reads the sky at
    l rho(chi) and is rotated by 2 dpsi(chi) + 2 Delta, so per pixel
    EB = (1/2) sin(4 dpsi + 4 Delta) C(l rho) and EE = cos^2(2 dpsi + 2 Delta)
    C(l rho), averaged over chi and over the footprint with weight W^2. The
    sin 4dpsi cos 4Delta part is odd in chi and drops out. Reading C at the
    sheared wavenumber is what makes the ratio grow with l: on the damping
    tail the stretched, high-shear corners -- where |Delta| is largest --
    gain weight. Pixels subsampled by `stride` in each direction.
    """
    th = T.load()
    lt, ct = th["ell"].astype(float), th["unlensed_EE"]
    sl = (slice(None, None, stride), slice(None, None, stride))
    s1, s2 = geo["s1"][sl].ravel(), geo["s2"][sl].ravel()
    dl = (geo["Delta"][geo["box"]][sl]).ravel()
    chi = (np.arange(n_chi) + 0.5) * (np.pi / n_chi)
    dpsi = np.arctan2(s2[:, None] * np.sin(chi), s1[:, None] * np.cos(chi)) - chi
    dpsi = (dpsi + np.pi / 2) % np.pi - np.pi / 2
    rho = np.sqrt(s1[:, None] ** 2 * np.cos(chi) ** 2
                  + s2[:, None] ** 2 * np.sin(chi) ** 2)
    a = 2 * dpsi + 2 * dl[:, None]
    k_eb = geo["sign"] * 0.5 * np.sin(2 * a)
    k_ee = np.cos(a) ** 2
    out = {}
    for name, m in geo["masks"].items():
        w = (m[sl].ravel() ** 2)
        ok = (w > 0) & np.isfinite(dl) & np.isfinite(s1)
        num, den = [], []
        for L in np.atleast_1d(ell):
            c = np.interp(L * rho[ok], lt, ct)
            num.append(np.sum(w[ok, None] * k_eb[ok] * c))
            den.append(np.sum(w[ok, None] * k_ee[ok] * c))
        out[name] = (np.array(num), np.array(den))
    return out


def map_test(proj="ProjSFL", seeds=range(S.NSIMS), refresh=False):
    """Stock vs polar flattening on the production QU maps. Cached.

    Returns a dict of arrays: ell (bin centres), for the 5% disk decoupled
    `disk_{EE,EB,BB}_{stock,polar}` (seed, bin), and for the 1500d field and
    halves coupled pseudo-spectra `{field}_{EE,EB,BB}_{stock,polar}`, plus
    the per-pixel predictions `{field}_eb_pred` (W^2-weighted
    s (1/2) sin 4Delta <cos 4dpsi> over W^2-weighted <cos^2 2dpsi>, s the
    pol_conv sign) and `{field}_tb_pred` (the same for TB/TE), and the
    Delta map over the field's bounding box for the figure.
    """
    if CACHE.exists() and not refresh:
        with np.load(CACHE) as f:
            out = {k: f[k] for k in f.files}
        if "east_eb_model_num" not in out:
            first = next(core.G3File(str(
                RUN / "projected_qu" / f"qu_{proj}_seed0000.g3")))
            _add_eb_model(out, field_geometry(first["Q"]))
            _save(out)
        return out
    import pymaster as nmt

    l0 = np.arange(S.LMAX // S.DELTA_L) * float(S.DELTA_L)
    b = nmt.NmtBinFlat(l0, l0 + S.DELTA_L)
    out = {"ell": b.get_effective_ells()}

    d = np.load(RUN / "flatsky_masks" / f"mask_{proj}_{MASK}.npz")
    dmask = d["mask"]
    lo, hi = int(d["lo"]), int(d["hi"])
    lx, ly = float(d["lx_rad"]), float(d["ly_rad"])
    ws = nmt.NmtWorkspaceFlat()
    ws.read_from(str(RUN / "coupling_matrices" / f"w22_{proj}_{MASK}.fits"))

    first = next(core.G3File(str(RUN / "projected_qu" / f"qu_{proj}_seed0000.g3")))
    geo = field_geometry(first["Q"])
    sign, extra, Delta, fm, fbox = (geo[k] for k in
                                    ("sign", "extra", "Delta", "masks", "box"))
    flx, fly = geo["lx"], geo["ly"]
    s1, s2 = geo["s1"], geo["s2"]
    g2 = ((s1 - s2) / (s1 + s2)) ** 2
    dl = Delta[fbox]
    out["field_delta_deg"] = np.degrees(dl)
    out["field_mask"] = fm["1500d"]
    for name, m in fm.items():
        w = m ** 2
        ok = np.isfinite(dl) & np.isfinite(g2)
        # <cos 4dpsi>_chi = (1 - g^2)(1 - 3g^2) over <cos^2 2dpsi>_chi =
        # 1 - 2g^2 + (3/2)g^4: the spin-4 and spin-2 rungs of the ladder.
        kern = sign * 0.5 * np.sin(4 * dl) * (1 - g2) * (1 - 3 * g2)
        ee = 1 - 2 * g2 + 1.5 * g2 ** 2
        out[f"{name}_eb_pred"] = np.sum((w * kern)[ok]) / np.sum((w * ee)[ok])
        out[f"{name}_tb_pred"] = sign * (np.sum((w * np.sin(2 * dl) * (1 - g2))[ok])
                                  / np.sum((w * np.cos(2 * dl) * (1 - g2))[ok]))
    out["disk_delta_max_deg"] = np.degrees(np.nanmax(np.abs(Delta[lo:hi, lo:hi][dmask > 0])))

    seeds = list(seeds)
    keys = [f"{f}_{x}_{v}" for f in ("disk", *fm) for x in ("EE", "EB", "BB")
            for v in ("stock", "polar")]
    acc = {k: [] for k in keys}

    def qu(z, box, mask):
        z = np.where(np.isfinite(z[box]), z[box], 0.0)
        return [z.real.copy(), z.imag.copy()]

    for si, seed in enumerate(seeds):
        fr = first if si == 0 else next(core.G3File(
            str(RUN / "projected_qu" / f"qu_{proj}_seed{seed:04d}.g3")))
        z_stock = np.asarray(fr["Q"], dtype=float) + 1j * np.asarray(fr["U"], dtype=float)
        z_polar = z_stock * extra
        for v, z in (("stock", z_stock), ("polar", z_polar)):
            box = (slice(lo, hi), slice(lo, hi))
            f2 = nmt.NmtFieldFlat(lx, ly, dmask, qu(z, box, dmask), purify_b=True)
            cl = ws.decouple_cell(nmt.compute_coupled_cell_flat(f2, f2, b))
            for k, row in (("EE", 0), ("EB", 1), ("BB", 3)):
                acc[f"disk_{k}_{v}"].append(cl[row])
            for name, m in fm.items():
                mm = np.ascontiguousarray(m)
                f2 = nmt.NmtFieldFlat(flx, fly, mm, qu(z, fbox, mm),
                                      purify_b=(name == "1500d"))
                cl = nmt.compute_coupled_cell_flat(f2, f2, b)
                for k, row in (("EE", 0), ("EB", 1), ("BB", 3)):
                    acc[f"{name}_{k}_{v}"].append(cl[row])
        print(f"  map_test {proj}: seed {seed} done", flush=True)
    out.update({k: np.array(v) for k, v in acc.items()})
    out["seeds"] = np.array(seeds)
    _add_eb_model(out, geo)
    _save(out)
    return out


def _add_eb_model(out, geo):
    """`eb_model` on the 50-wide bins with 500 < l < 4500."""
    ell = out["ell"]
    sel = (ell > 500) & (ell < 4500)
    for name, (num, den) in eb_model(geo, ell[sel]).items():
        for k, v in (("num", num), ("den", den)):
            full = np.full(ell.size, np.nan)
            full[sel] = v
            out[f"{name}_eb_model_{k}"] = full


def _save(out):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(CACHE)


def model_ratio(mom, i, j, ell, with_delta):
    """Forward-model BB/EE per bin, with FlattenPol's Delta or with the polar angle."""
    th = T.load()
    if not with_delta:
        mom = dict(mom, cell_sin2_2delta=np.zeros_like(mom["cell_sin2_2delta"]))
    bb, ee = SL.predicted_spectra(mom, i, j, ell, th["ell"].astype(float),
                                  th["unlensed_EE"], band_limit=S.LMAX)
    return bb / ee


def summary(res, mom_disk, mom_field, proj="ProjSFL"):
    """The numbers: measured and modelled BB/EE shifts, EB/EE on the halves."""
    j = S.PROJS.index(proj)
    ell = res["ell"]
    band = (ell > L_CLEAN[0]) & (ell < L_CLEAN[1])
    out = {}
    r_st = res["disk_BB_stock"] / res["disk_EE_stock"]
    r_po = res["disk_BB_polar"] / res["disk_EE_polar"]
    per_seed = (r_st / r_po)[:, band].mean(1) - 1
    out["disk_shift_maps"] = per_seed.mean()
    out["disk_shift_maps_err"] = per_seed.std(ddof=1) / np.sqrt(per_seed.size)
    m_d = model_ratio(mom_disk, MASK, j, ell, True)
    m_0 = model_ratio(mom_disk, MASK, j, ell, False)
    out["disk_shift_model"] = (m_d / m_0)[band].mean() - 1
    mf_d = model_ratio(mom_field, 0, j, ell, True)
    mf_0 = model_ratio(mom_field, 0, j, ell, False)
    out["field_shift_model"] = (mf_d / mf_0)[band].mean() - 1
    f_st = res["1500d_BB_stock"] / res["1500d_EE_stock"]
    f_po = res["1500d_BB_polar"] / res["1500d_EE_polar"]
    ps = (f_st / f_po)[:, band].mean(1) - 1
    out["field_shift_maps"] = ps.mean()
    out["field_shift_maps_err"] = ps.std(ddof=1) / np.sqrt(ps.size)
    out["field_leak_model"] = mf_0[band].mean()
    for name in ("1500d", "east", "west"):
        for v in ("stock", "polar"):
            eb = (res[f"{name}_EB_{v}"][:, band].mean(1)
                  / res[f"{name}_EE_{v}"][:, band].mean(1))
            out[f"{name}_eb_{v}"] = eb.mean()
            out[f"{name}_eb_{v}_err"] = eb.std(ddof=1) / np.sqrt(eb.size)
        out[f"{name}_eb_pred"] = float(res[f"{name}_eb_pred"])
        out[f"{name}_eb_model"] = (np.nanmean(res[f"{name}_eb_model_num"][band])
                                   / np.nanmean(res[f"{name}_eb_model_den"][band]))
        out[f"{name}_tb_pred"] = float(res[f"{name}_tb_pred"])
    return out


def _wide(x, ell, edges):
    """Band means of the 50-wide bins over `edges`, along the last axis."""
    return np.stack([x[..., (ell > lo) & (ell < hi)].mean(-1)
                     for lo, hi in zip(edges[:-1], edges[1:])], -1)


def figure(res, mom_disk, proj="ProjSFL", path=FIGURE):
    """Three panels: Delta over the 1500d field; the BB/EE FlattenPol adds
    over the polar angle on the 5% disk, against the forward model's
    sin^2 2Delta term; EB/EE on the 1500d field's halves, both flattenings,
    against `eb_model` (dashed). Spectra in
    500-wide bands, numerator and denominator band-averaged separately; error
    bars are the error of the mean over the simulations."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PS.rc())
    j = S.PROJS.index(proj)
    ell = res["ell"]
    edges = np.arange(500, 4501, 500)
    lc = 0.5 * (edges[:-1] + edges[1:])
    nsim = res["disk_EE_stock"].shape[0]
    stock_c, polar_c = PS.SECONDARY[0], PS.PROJ_COLORS[proj]
    styles = (("stock", stock_c, "o", "FlattenPol", -40),
              ("polar", polar_c, "s", "polar angle", 40))

    fig, axes = plt.subplots(1, 3, figsize=(PS.TEXT_W, 2.5),
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1]),
                             layout="constrained")
    ax = axes[0]
    dm = np.where(res["field_mask"] > 0, res["field_delta_deg"], np.nan)
    lim = np.nanmax(np.abs(dm))
    im = ax.imshow(dm, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(r"FlattenPol $-$ polar angle, 1500d")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.08, pad=0.03)
    cb.set_label(r"$\Delta$ [deg]")
    for xf, txt in ((0.03, "E"), (0.97, "W")):
        ax.text(xf, 0.05, txt, transform=ax.transAxes, ha="center",
                color=PS.INK_SOFT)

    ax = axes[1]
    m_0 = _wide(model_ratio(mom_disk, MASK, j, ell, False), ell, edges)
    m_d = _wide(model_ratio(mom_disk, MASK, j, ell, True), ell, edges)
    ratio = {v: _wide(res[f"disk_BB_{v}"], ell, edges)
             / _wide(res[f"disk_EE_{v}"], ell, edges) for v in ("stock", "polar")}
    r = (ratio["stock"] / ratio["polar"] - 1) * 100
    ax.errorbar(lc, r.mean(0), r.std(0, ddof=1) / np.sqrt(nsim), color=stock_c,
                marker="o", ls="none", ms=PS.MS, capsize=0,
                label=f"simulations ({nsim})")
    ax.plot(lc, (m_d / m_0 - 1) * 100, color=polar_c, lw=PS.LW,
            label=r"forward model, $\sin^2 2\Delta$ term")
    ax.axhline(0, color=PS.RULE, lw=PS.LW_REF)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$(BB/EE)_{\rm FlattenPol} / (BB/EE)_{\rm polar} - 1$ [\%]")
    ax.set_title(r"Leakage excess, SFL, 5\% disk")
    ax.legend(loc="upper left", fontsize=PS.LEGEND_SIZE - 1)

    ax = axes[2]
    for name, fill in (("east", True), ("west", False)):
        for v, c, mk, lab, dx in styles:
            r = (_wide(res[f"{name}_EB_{v}"], ell, edges)
                 / _wide(res[f"{name}_EE_{v}"], ell, edges)) * 100
            ax.errorbar(lc + dx, r.mean(0), r.std(0, ddof=1) / np.sqrt(nsim),
                        color=c, marker=mk, ls="none", ms=PS.MS, capsize=0,
                        mfc=c if fill else "white",
                        label=f"{lab}, {name}")
        model = (_wide(res[f"{name}_eb_model_num"], ell, edges)
                 / _wide(res[f"{name}_eb_model_den"], ell, edges))
        ax.plot(lc, 100 * model, color=stock_c, lw=PS.LW_REF, ls="--",
                label=r"model from $\Delta$" if name == "east" else None)
    ax.axhline(0, color=PS.RULE, lw=PS.LW_REF)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$EB/EE$ from an $E$-only sky [\%]")
    ax.set_title("TB-type pattern, SFL, 1500d halves")
    ax.set_ylim(-2.8, 4.2)
    ax.legend(loc="upper left", fontsize=PS.LEGEND_SIZE - 1.5, ncol=2,
              columnspacing=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def main():
    print("closed form vs scipy.linalg.polar, and residuals from xy_to_angle (deg):")
    for proj, r in check_closed_form().items():
        print(f"  {S.PROJ_LABEL[proj]:<6} g_max {r['g_max']:.3f}  |phi-scipy| {r['scipy']:.1e}"
              f"  asym {r['asym']:.1e}  |FlattenPol-normalised| {r['normalised']:.1e}"
              f"  |arg form - rot_R| {r['complex_polar']:.1e}"
              f"  |Delta| stock {r['delta_stock']:.3f}  polar {r['delta_polar']:.1e}")
    refresh = "--refresh" in sys.argv
    res = map_test(refresh=refresh)
    mom_disk = SL.load_moments()
    mom_field = SL.field_moments()
    for k, v in summary(res, mom_disk, mom_field).items():
        print(f"  {k:<24} {v: .4e}")
    figure(res, mom_disk)


if __name__ == "__main__":
    main()
