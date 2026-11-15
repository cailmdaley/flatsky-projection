"""The E-mode template of Sect. subsec:template, applied to the simulations.

The claim
---------
A sky E mode lands on the flat grid as E cos 2dpsi and B sin 2dpsi, so the B
the projection makes is tan 2dpsi times the grid E. dpsi is fixed by the local
Jacobian and by the direction of the grid wavevector, so the template
B_temp = S_J[E_hat] can be built from the measured E and subtracted. With
E_hat = E + N_E, the signal cancels and the noise leaves
<tan^2 2dpsi> N^EE ~ 2 <g^2> N^EE (Eq. template_resid).

The operator on a map
---------------------
S_J depends on position (through J) and on mode direction (through dpsi), so
it is a pseudo-differential operator, not a Fourier multiplier. Its symbol
f(x, phi) = tan 2dpsi(x, phi) is pi-periodic in the grid mode angle phi, so it
expands exactly in even angular harmonics,

    f(x, phi) = sum_n c_2n(x) exp(2 i n phi),   c_-2n = conj(c_2n),

and the operator becomes a short sum of products of smooth position maps with
Fourier-filtered E maps,

    B_temp(x) = c_0(x) F_0(x) + 2 Re sum_{n>=1} c_2n(x) F_2n(x),
    F_2n = IFFT[exp(2 i n phi_l) E_l].

This is the local approximation of the paper made literal: each pixel applies
the multiplier of its own Jacobian. The harmonics fall off as g^n; |n| <= 6
truncates f at the 1e-5 level for the shears here (`check_truncation`).

dpsi itself, per pixel. With A the Jacobian from grid pixels (x, y) to sky
tangent coordinates (east arc, north), a sky wavevector k appears on the grid
at l = A^T k. Writing A = R Sigma' (polar decomposition, Sigma' = sqrt(A^T A),
the metric's square root), the sky mode's direction in the flattened frame is
Sigma'^{-1} l, so dpsi = arg(Sigma'^{-1} l) - arg l. Only the grid metric
A^T A enters; the handedness of (x, y) against (east, north) and the pol
convention set one overall sign, fixed once from the noise-free correlation
of B_temp with the grid B (`TEMPLATE_SIGN`).

The maps are the production QU-leg maps reflattened by the polar angle
(`polar_flatten`), so the rotation is removed exactly and what is left is the
pure shear the template models. E and B on the grid come from the FFT of the
whole 3300^2 grid (E_l = cos 2phi Q + sin 2phi U), B_temp is evaluated on a
box covering every mask, and is subtracted at the Q/U level as the pure-B
field (Q, U) = (-sin 2phi, cos 2phi) B_temp. The spectra of the result go
through NaMaster exactly as in `polar_flatten`: purified, decoupled with the
production w22 workspace on the 5% disk; purified coupled pseudo-spectra on
the 1500d field.

The template here takes E from the full, unmasked grid: it tests the operator,
not the separate question of estimating E near a mask edge.

Noise
-----
White noise is added to E only (the template's input), as a scalar white map
of power Delta^2 in uK-arcmin, so the residual's noise part isolates
S_J[N_E]. A run at unit level gives the template's noise kernel directly,
BB_noise / Delta^2, to compare with the W^2-weighted <tan^2 2dpsi>.

Built from the raw E_hat, the template's noise term is white while the
leakage follows EE down the damping tail, so above the l where N^EE reaches
C^EE the subtraction adds more B than it removes. The template from the
Wiener-filtered E, C/(C + N) E_hat, leaves <tan^2 2dpsi> C N/(C + N) instead,
which never exceeds the raw leakage; both are run.

Run:  systemd-run --user --scope -p MemoryMax=12G bash env.sh python3 analysis/template_subtraction.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as PS
import shear_leakage as SL
import spectra as S
import theory as T
from template_geometry import (  # noqa: F401  (re-exported for callers)
    ARCMIN_RAD, N_HARM, N_PHI, PAD_DEG, TAPER_DEG, TEMPLATE_SIGN,
    Grid, _grid, _metric, harmonics, symbol)

from spt3g import core, maps

DEG = core.G3Units.deg
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "highell"
CACHE_DIR = SL.CACHE_DIR
FIGURE = ROOT / "figures" / "template_subtraction"
MASK = 5
NOISE_LEVELS = (2.5, 5.0)   # uK-arcmin on E: SPT-3G-like, and the paper's nominal
L_CLEAN = (1000, 4000)
L_REF = (900, 1100)         # the paper's l = 1000 reference, four 50-wide bins


def white_e(shape, level, seed, res):
    """E_l of a white scalar map of power level^2 (uK-arcmin)^2."""
    sigma = level * ARCMIN_RAD / res        # res in radians per pixel
    n = np.random.default_rng(seed).standard_normal(shape) * sigma
    return np.fft.fft2(n)


def run(proj="ProjSFL", seeds=range(S.NSIMS), refresh=False):
    """Raw and template-subtracted spectra on the 5% disk and the 1500d field.

    Cached. Returns a dict: ell; geometry scalars (W^2-weighted <g^2>,
    <tan^2 2dpsi>, truncation error, the noise-free correlation coefficient
    of B_temp with B_grid in the disk); and per mask m in (disk, field),
    arrays (seed, bin) `{m}_EE`, `{m}_BB_raw`, `{m}_BB_sub`,
    `{m}_BB_sub_n{level}` and `{m}_BB_sub_w{level}` (template from the raw
    and from the Wiener-filtered noisy E) for each noise level, and
    `{m}_BB_tn1`, the BB of
    S_J of unit (1 uK-arcmin) E noise alone. Disk spectra are decoupled with
    the production workspace; field spectra are coupled pseudo-spectra, with
    the mean W^2 over the field box stored as `field_w2`.
    """
    cache = CACHE_DIR / f"template_subtraction_{proj}.npz"
    if cache.exists() and not refresh:
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}
    import pymaster as nmt

    gr = Grid(proj, RUN / "projected_qu" / f"qu_{proj}_seed0000.g3",
              RUN / "flatsky_masks" / f"mask_{proj}_{MASK}.npz", T.OUT)
    sign = TEMPLATE_SIGN
    l0 = np.arange(S.LMAX // S.DELTA_L) * float(S.DELTA_L)
    b = nmt.NmtBinFlat(l0, l0 + S.DELTA_L)
    out = {"ell": b.get_effective_ells(), "trunc": gr.trunc}
    for m in ("disk", "field"):
        out[f"{m}_g2"] = gr.weighted(gr.g ** 2, m)
        out[f"{m}_tan2"] = gr.weighted(gr.tan2, m)
    out["field_w2"] = float(np.mean(gr.field["mask"] ** 2))
    ws = nmt.NmtWorkspaceFlat()
    ws.read_from(str(RUN / "coupling_matrices" / f"w22_{proj}_{MASK}.fits"))

    def spectra(q, u, m):
        M = getattr(gr, m)
        box = M["box"]
        f2 = nmt.NmtFieldFlat(M["lx"], M["ly"], M["mask"],
                              [np.ascontiguousarray(q[box]),
                               np.ascontiguousarray(u[box])], purify_b=True)
        cl = nmt.compute_coupled_cell_flat(f2, f2, b)
        if m == "disk":
            cl = ws.decouple_cell(cl)
        return cl[0], cl[3]

    variants = (["raw", "sub", "tn1"] + [f"sub_n{lv:g}" for lv in NOISE_LEVELS]
                + [f"sub_w{lv:g}" for lv in NOISE_LEVELS])
    acc = {f"{m}_{k}": [] for m in ("disk", "field")
           for k in ["EE"] + [f"BB_{v}" for v in variants]}
    corr = []
    for seed in seeds:
        fr = next(core.G3File(str(RUN / "projected_qu" / f"qu_{proj}_seed{seed:04d}.g3")))
        z = (np.asarray(fr["Q"], float) + 1j * np.asarray(fr["U"], float)) * gr.extra
        del fr
        q, u = z.real.copy(), z.imag.copy()
        del z
        e_l, b_l = gr.eb(q, u)
        bt = sign * gr.template(e_l)
        bg = np.fft.ifft2(b_l).real
        w = np.zeros(gr.shape); w[gr.disk["box"]] = gr.disk["mask"]
        corr.append(np.sum(w ** 2 * bg * bt)
                    / np.sqrt(np.sum(w ** 2 * bg ** 2) * np.sum(w ** 2 * bt ** 2)))
        del bg, b_l, w
        tq, tu = gr.qu_of_b(bt)
        n_l = white_e(gr.shape, 1.0, 10_000 + seed, gr.res)
        tn = sign * gr.template(n_l)
        nq, nu = gr.qu_of_b(tn)
        del tn, bt
        fields = {"raw": (q, u), "sub": (q - tq, u - tu), "tn1": (nq, nu)}
        for lv in NOISE_LEVELS:
            fields[f"sub_n{lv:g}"] = (q - tq - lv * nq, u - tu - lv * nu)
            # The template from the Wiener-filtered E, C / (C + N).
            n_ee = (lv * ARCMIN_RAD) ** 2
            wf = gr.c_ee / (gr.c_ee + n_ee)
            wq, wu = gr.qu_of_b(sign * gr.template(wf * (e_l + lv * n_l)))
            fields[f"sub_w{lv:g}"] = (q - wq, u - wu)
            del wf
        del e_l, n_l
        for m in ("disk", "field"):
            for v, (qq, uu) in fields.items():
                ee, bb = spectra(qq, uu, m)
                if v == "raw":
                    acc[f"{m}_EE"].append(ee)
                acc[f"{m}_BB_{v}"].append(bb)
        del fields, q, u, tq, tu, nq, nu
        print(f"  {proj} seed {seed}: corr(B_temp, B_grid) = {corr[-1]:.5f}", flush=True)
    out.update({k: np.array(v) for k, v in acc.items()})
    out["corr"] = np.array(corr)
    out["seeds"] = np.array(list(seeds))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(cache)
    return out


def check(proj="ProjSFL"):
    """Seed 0: E/B convention, template sign, and correlation, from FFTs alone."""
    gr = Grid(proj, RUN / "projected_qu" / f"qu_{proj}_seed0000.g3",
              RUN / "flatsky_masks" / f"mask_{proj}_{MASK}.npz", T.OUT)
    fr = next(core.G3File(str(RUN / "projected_qu" / f"qu_{proj}_seed0000.g3")))
    z = (np.asarray(fr["Q"], float) + 1j * np.asarray(fr["U"], float)) * gr.extra
    q, u = z.real.copy(), z.imag.copy()
    e_l, b_l = gr.eb(q, u)
    w = np.zeros(gr.shape); w[gr.disk["box"]] = gr.disk["mask"]
    em, bm = np.fft.ifft2(e_l).real, np.fft.ifft2(b_l).real
    bt = gr.template(e_l)
    w2 = w ** 2
    print(f"{proj}: truncation {gr.trunc:.2e}; disk <g^2> {gr.weighted(gr.g**2, 'disk'):.4e}"
          f" <tan^2> {gr.weighted(gr.tan2, 'disk'):.4e};"
          f" field <g^2> {gr.weighted(gr.g**2, 'field'):.4e} <tan^2> {gr.weighted(gr.tan2, 'field'):.4e}")
    print(f"  map-space var ratio B/E in disk: {np.sum(w2*bm**2)/np.sum(w2*em**2):.4e}")
    for s in (1, -1):
        r = bm - s * bt
        print(f"  sign {s:+d}: corr {s*np.sum(w2*bm*bt)/np.sqrt(np.sum(w2*bm**2)*np.sum(w2*bt**2)):+.5f}"
              f"  var(resid)/var(B) {np.sum(w2*r**2)/np.sum(w2*bm**2):.4e}")


def summary(res):
    """BB/EE per variant, mean and error over seeds.

    Over L_CLEAN for every variant (`{m}_{v}`), and over L_REF (`{m}_{v}_ref`,
    the paper's l = 1000 reference), numerator and denominator band-averaged
    separately. With noise the residual is white while EE falls, so the
    L_CLEAN average of a noisy variant is dominated by its low end; `_ref`
    and `{m}_cross_n{level}`, the first l at which the unfiltered template's
    residual exceeds the raw leakage, describe it better.
    """
    ell = res["ell"]
    out = {"trunc": float(res["trunc"]), "corr": float(res["corr"].mean()),
           "corr_min": float(res["corr"].min())}
    n = res["disk_EE"].shape[0]
    for m in ("disk", "field"):
        for suffix, (lo, hi) in (("", L_CLEAN), ("_ref", L_REF)):
            band = (ell > lo) & (ell < hi)
            ee = res[f"{m}_EE"][:, band].mean(1)
            for v in [k[len(m) + 4:] for k in res if k.startswith(f"{m}_BB_")]:
                r = res[f"{m}_BB_{v}"][:, band].mean(1) / ee
                out[f"{m}_{v}{suffix}"] = r.mean()
                out[f"{m}_{v}{suffix}_err"] = r.std(ddof=1) / np.sqrt(n)
        for lv in NOISE_LEVELS:
            excess = (res[f"{m}_BB_sub_n{lv:g}"].mean(0)
                      > res[f"{m}_BB_raw"].mean(0)) & (ell > 500)
            out[f"{m}_cross_n{lv:g}"] = float(ell[excess][0]) if excess.any() else np.nan
        # The noise kernel: BB of S_J[unit E noise] over Delta^2 (per-seed mean
        # over the band), against the W^2-weighted <tan^2 2dpsi>.
        norm = res["field_w2"] if m == "field" else 1.0
        kern = res[f"{m}_BB_tn1"][:, band].mean(1) / ARCMIN_RAD ** 2 / norm
        out[f"{m}_kernel"] = kern.mean()
        out[f"{m}_kernel_err"] = kern.std(ddof=1) / np.sqrt(n)
        out[f"{m}_tan2"] = float(res[f"{m}_tan2"])
        out[f"{m}_g2"] = float(res[f"{m}_g2"])
    return out


def _wide(x, ell, edges):
    """Band means of the 50-wide bins over `edges`, along the last axis."""
    return np.stack([x[..., (ell > lo) & (ell < hi)].mean(-1)
                     for lo, hi in zip(edges[:-1], edges[1:])], -1)


def figure(res_by_proj, path=FIGURE, level=NOISE_LEVELS[0]):
    """BB/EE vs l on the 5% disk and the 1500d field.

    SFL raw, template-subtracted noise-free, and subtracted with `level`
    uK-arcmin of E noise, from the raw and from the Wiener-filtered E, each
    noisy case against its prediction: <tan^2 2dpsi> N / EE for the raw E,
    and <tan^2 2dpsi> N / (EE + N) for the Wiener-filtered E. ZEA raw and
    noise-free subtracted for contrast; the lensed BB/EE for scale. Means over
    the simulations in 250-wide bands; EE in the predictions is the measured
    one (on the field, the coupled pseudo-EE over the mean W^2).
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(PS.rc())
    edges = np.arange(500, 4801, 250)
    lc = 0.5 * (edges[:-1] + edges[1:])
    th = T.load()
    lens = np.interp(lc, th["ell"], th["lensed_BB"]) / np.interp(lc, th["ell"], th["lensed_EE"])
    n_ee = (level * ARCMIN_RAD) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(PS.TEXT_W, 3.0), sharey=True,
                             layout="constrained")
    for ax, m in zip(axes, ("disk", "field")):
        for proj, res in res_by_proj.items():
            col = PS.PROJ_COLORS[proj]
            lab = S.PROJ_LABEL[proj]
            ell = res["ell"]
            norm = res["field_w2"] if m == "field" else 1.0
            ee = _wide(res[f"{m}_EE"], ell, edges)

            def ratio(v):
                return (_wide(res[f"{m}_BB_{v}"], ell, edges) / ee).mean(0)
            ax.plot(lc, ratio("raw"), color=col, lw=PS.LW, marker="o", ms=PS.MS,
                    label=f"{lab}, raw")
            ax.plot(lc, ratio("sub"), color=col, lw=PS.LW, ls=":", marker="^",
                    ms=PS.MS, mfc="white", label=f"{lab}, subtracted, noise-free")
            if proj != "ProjSFL":
                continue
            ax.plot(lc, ratio(f"sub_n{level:g}"), color=PS.SECONDARY[0], lw=PS.LW,
                    ls="none", marker="s", ms=PS.MS,
                    label=f"{lab}, subtracted, {level:g}" r" $\mu$K-arcmin $E$ noise")
            ax.plot(lc, ratio(f"sub_w{level:g}"), color=PS.SECONDARY[1], lw=PS.LW,
                    ls="none", marker="D", ms=PS.MS * 0.9,
                    label=f"{lab}, same, Wiener-filtered $E$")
            eev = (ee / norm).mean(0)
            ax.plot(lc, res[f"{m}_tan2"] * n_ee / eev, color=PS.SECONDARY[0],
                    lw=PS.LW_REF, ls="--",
                    label=r"predicted, $\langle\tan^2 2\delta\psi\rangle N^{EE}/C^{EE}$")
            ax.plot(lc, res[f"{m}_tan2"] * n_ee / (eev + n_ee), color=PS.SECONDARY[1],
                    lw=PS.LW_REF, ls="-.",
                    label=r"predicted, $\langle\tan^2 2\delta\psi\rangle N^{EE}/(C^{EE}+N^{EE})$")
        ax.plot(lc, lens, color=PS.RULE, lw=PS.LW_REF, label=r"lensed $BB/EE$")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 3)
        ax.set_xlabel(r"$\ell$")
        ax.text(0.03, 0.97, {"disk": r"5\% disk", "field": "1500d field"}[m],
                transform=ax.transAxes, ha="left", va="top", color=PS.INK_SOFT)
    axes[0].set_ylabel(r"$BB/EE$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3,
               fontsize=PS.LEGEND_SIZE - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext), dpi=200)
    print(f"wrote {path}.png/.pdf")


def main():
    if "--check" in sys.argv:
        for p in ("ProjSFL", "ProjZEA"):
            check(p)
        return
    refresh = "--refresh" in sys.argv
    res = {p: run(p, refresh=refresh) for p in ("ProjSFL", "ProjZEA")}
    for p, r in res.items():
        print(p)
        for k, v in summary(r).items():
            print(f"  {k:<22} {v: .4e}")
    figure(res)


if __name__ == "__main__":
    main()
