#!/usr/bin/env python
"""NaMaster flat-sky EE/BB bandpowers for both the Q/U and E/B legs.

Two legs, two different NaMaster objects
-----------------------------------------
The Q/U leg (``qu_{proj}_seed{N:04d}.g3``, keys ``Q``/``U``) is a genuine
spin-2 field: Q and U were built from the sky's actual polarization tensor,
flattened with ``FlattenPol`` in ``project_qu.py`` (already applied -- this
stage must *not* call it again). It goes into
``NmtFieldFlat(lx, ly, mask, [q, u], purify_b=...)`` and is decoupled with the
spin-2 workspace (``w22_{proj}_{i}.fits``, ``ncls=4``, order EE/EB/BE/BB);
this leg carries the projection's E-to-B leakage, which is the thing the
whole analysis measures.

The E/B leg (``eb_{proj}_seed{N:04d}.g3``, keys ``E``/``B``) is *not* a
second spin-2 field with a confusingly-named ``[E, B]`` pair. Its maps were
synthesized directly from the E- and B-mode alms with ``hp.alm2map`` used as
a spin-0 transform and projected with no ``FlattenPol`` (see
``project_eb.py``): there is no Q/U tensor in this leg's history for a spin-2
decomposition to act on. Handing ``[E, B]`` to a spin-2 field would ask
NaMaster to E/B-decompose two maps that were never a polarization tensor,
manufacturing spurious mixing between fields that share no spin-2
relationship -- precisely the leakage this leg exists to be free of. pymaster
enforces the physical point mechanically: ``NmtFieldFlat`` raises
``ValueError`` if purification is requested for anything but a spin-2 field,
because B-mode purification is specifically about the spin-2 E/B ambiguity at
the mask boundary, which a scalar field does not have (checked against
pymaster 1.5.1's ``field.py`` before writing this).

So the E/B leg is two independent spin-0 auto-spectra:
``NmtFieldFlat(lx, ly, mask, [E])`` and ``NmtFieldFlat(lx, ly, mask, [B])``,
each with ``purify_e=purify_b=False`` (undefined for spin-0), each decoupled
through the *scalar* workspace ``w00_{proj}_{i}.fits`` that
``coupling_matrices.py`` now also writes for exactly this reason. "ee" for
this leg is the E-field's own auto-power; "bb" is the B-field's. Both
``compute_coupled_cell_flat`` calls return a single row (``ncls=1``), so no
index-0-vs-3 split is needed the way it is for the Q/U leg.

Pixel window
------------
Two independent windows are divided out of the *power* spectrum, hence
squared (a window multiplies the map, so it enters C_l as window**2):

* ``flat_only`` (production choice for high-ell work): divides by
  ``spt3g.mapspectra.pixel_window.pixwin_flatsky``, the azimuthally-averaged
  2-arcmin flat-sky top-hat window from the interpolate-then-rebin step.
  A smooth analytic function of ell rather than a tabulated array, so it is
  evaluated directly at each bandpower's effective ell
  (``NmtBinFlat.get_effective_ells()``) rather than binned from a fine grid.
* ``healpix_only``: divides by the Nside-8192 Healpix window, tabulated on
  integer ell by ``hp.pixwin`` and top-hat-binned onto the bandpowers with
  ``NmtBinFlat.bin_cell``. Empirically the unlensed_TQU inputs carry NO such
  window (Cl of the preprocessed alms matches theory flat to ell = 6000,
  incompatible with pixwin**2 at > 50 sigma) -- the sims were synthesized
  in harmonic space without pixelization smoothing -- so this mode
  over-corrects and exists only for the explicit comparison.
* ``healpix_and_flat``: both of the above; over-corrects for the same
  reason.

Resolution and pixel size are fixed for this whole analysis (see the
README's "Grid and geometry" section and ``astra.yaml``'s resolution-chain
note), not per-run decisions, so ``INPUT_HEALPIX_NSIDE`` and ``RES_ARCMIN``
below are constants rather than flags.

Resumable: bandpowers are checkpointed per (leg, projection, mask) into
``<out>/partial/{leg}_{proj}_mask{i}.npy``, an array of shape
``(2, nseed, nbins)`` (axis 0 = ee, bb) written after every seed, atomically
(temp file + rename) so a kill mid-seed leaves the previous seed's result
intact rather than a truncated file. ``bandpowers.npz`` is assembled from the
partials at the end of every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _decisions import (
    add_common_args,
    parse_delta_ell,
    parse_lmax,
    parse_pixel_window,
    parse_purify_b,
    read_index,
    set_threads,
    write_index,
)

LEGS = ("qu", "eb")

# Fixed for the whole analysis (README "Grid and geometry", astra.yaml
# resolution-chain note) -- not a `_decisions` knob, so not a flag here.
INPUT_HEALPIX_NSIDE = 8192
RES_ARCMIN = 2.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qu-maps", required=True, help="projected_qu output directory")
    p.add_argument("--eb-maps", required=True, help="projected_eb output directory")
    p.add_argument("--masks", required=True, help="flatsky_masks output directory")
    p.add_argument("--workspaces", required=True, help="coupling_matrices output directory")
    p.add_argument("--projections", nargs="+", required=True)
    p.add_argument("--purify-b", required=True, help="purify | standard | true | false")
    p.add_argument("--pixel-window", required=True, help="none | healpix_only | healpix_and_flat")
    p.add_argument("--lmax", required=True, help="l4000 | l6000 | bare integer")
    p.add_argument("--delta-ell", type=parse_delta_ell, default=10)
    add_common_args(p, seeds=True)
    return p


def atomic_save(path: Path, **arrays) -> None:
    """Write an npz/npy through a temp file, then rename into place.

    Both ``np.save`` and ``np.savez`` append their extension only if the
    filename doesn't already end with it, so the ``.tmp.npy`` / ``.tmp.npz``
    naming below is safe -- matching the crash-safety pattern already used
    for this project's mask cache (``analysis/crawford_masks.cached_arrays``)
    and for this stage's sibling, ``coupling_matrices.atomic_write_workspace``.
    """
    if path.suffix == ".npy":
        (value,) = arrays.values()
        tmp = path.with_name(path.stem + ".tmp.npy")
        import numpy as np

        np.save(tmp, value)
    else:
        import numpy as np

        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
    tmp.replace(path)


def load_workspace_manifest(workspaces_dir: Path, projections: list[str]):
    """Group the coupling_matrices manifest by projection; verify uniform masks."""
    manifest = read_index(workspaces_dir)
    by_proj: dict[str, dict[int, dict]] = {}
    for entry in manifest["workspaces"]:
        if entry["projection"] in projections:
            by_proj.setdefault(entry["projection"], {})[entry["mask_index"]] = entry

    missing = [p for p in projections if p not in by_proj]
    if missing:
        raise FileNotFoundError(
            f"no workspaces found for projection(s) {missing} under {workspaces_dir}"
        )

    index_sets = {p: frozenset(by_proj[p]) for p in projections}
    first = index_sets[projections[0]]
    mismatched = {p: s for p, s in index_sets.items() if s != first}
    if mismatched:
        raise ValueError(
            "projections do not share the same set of mask indices: "
            f"{ {p: sorted(s) for p, s in index_sets.items()} }"
        )
    mask_indices = sorted(first)
    return manifest, by_proj, mask_indices


def load_mask(masks_dir: Path, proj: str, i: int):
    import numpy as np

    d = np.load(masks_dir / f"mask_{proj}_{i}.npz")
    mask = np.asarray(d["mask"], dtype=np.float64)
    lo, hi = int(d["lo"]), int(d["hi"])
    lx_rad, ly_rad = float(d["lx_rad"]), float(d["ly_rad"])
    return mask, lo, hi, lx_rad, ly_rad


def finite_in_mask(m, mask):
    """Zero non-finite pixels, requiring that none carry mask weight.

    Orthographic (SIN) grids are undefined beyond the unit disk -- 1 rad =
    57.3 deg from centre -- so the 110-deg box's corners are NaN. Every mask
    stays inside that disk, but NaN * 0 is NaN and NaMaster then returns an
    all-NaN spectrum; zeroing them is exact because the mask weight there is
    zero. A NaN *inside* the mask would be a real problem, so it raises.
    """
    import numpy as np

    bad = ~np.isfinite(m)
    if not bad.any():
        return m
    if (mask[bad] != 0).any():
        raise ValueError(
            f"{int((mask[bad] != 0).sum())} non-finite map pixels carry mask weight"
        )
    out = m.copy()
    out[bad] = 0.0
    return out


def compute_pixel_window(mode: str, b, ell_eff, lmax: int):
    import numpy as np

    if mode == "none":
        return np.ones_like(ell_eff)

    correction = np.ones_like(ell_eff)

    if mode in {"healpix_only", "healpix_and_flat"}:
        # Empirically the unlensed_TQU inputs carry NO Nside-8192 window
        # (Cl(alms)/Cl(theory) flat to ell=6000), so these modes over-correct;
        # they are kept for the explicit comparison, not for production.
        import healpy as hp

        ells = np.arange(lmax + 1, dtype=float)
        pw_hp_sq = hp.pixwin(INPUT_HEALPIX_NSIDE)[: lmax + 1] ** 2
        correction = correction * b.bin_cell(ells, pw_hp_sq)

    if mode in {"flat_only", "healpix_and_flat"}:
        from spt3g.mapspectra.pixel_window import pixwin_flatsky

        pw_flat = pixwin_flatsky(ell_eff, RES_ARCMIN)
        correction = correction * pw_flat**2

    return correction


def compute_qu_bandpowers(q, u, mask, lx_rad, ly_rad, purify_b, b, w):
    """EE, BB from a genuine spin-2 (Q, U) field, decoupled through a w22 workspace.

    Factored out of the main loop so the exact same call sequence can be
    exercised directly -- e.g. by ``tmp_verify`` scripts feeding it 2021
    reference maps -- without going through the directory/manifest plumbing.
    """
    import pymaster as nmt

    f2 = nmt.NmtFieldFlat(lx_rad, ly_rad, mask, [q, u], purify_b=purify_b)
    dcl = w.decouple_cell(nmt.compute_coupled_cell_flat(f2, f2, b))
    return dcl[0], dcl[3]


def compute_eb_bandpowers(e, bmap, mask, lx_rad, ly_rad, b, w):
    """EE, BB from two independent spin-0 (E, B) fields, decoupled through a w00 workspace.

    See the module docstring for why this is two spin-0 auto-spectra rather
    than one spin-2 cross-spectrum.
    """
    import pymaster as nmt

    fE = nmt.NmtFieldFlat(lx_rad, ly_rad, mask, [e])
    fB = nmt.NmtFieldFlat(lx_rad, ly_rad, mask, [bmap])
    ee = w.decouple_cell(nmt.compute_coupled_cell_flat(fE, fE, b))[0]
    bb = w.decouple_cell(nmt.compute_coupled_cell_flat(fB, fB, b))[0]
    return ee, bb


def load_partial(path: Path, shape):
    import numpy as np

    if path.exists():
        try:
            arr = np.load(path)
        except (EOFError, ValueError, OSError):
            # A kill mid-write leaves a truncated .npy; rebuild from scratch
            # rather than making every caller defensive.
            path.unlink(missing_ok=True)
            return np.full(shape, np.nan)
        if arr.shape != shape:
            raise ValueError(
                f"{path} has shape {arr.shape}, expected {shape} -- "
                "this --out directory was likely used with different "
                "--seed-start/--seed-stop/--lmax/--delta-ell before"
            )
        return arr
    return np.full(shape, np.nan)


def main() -> None:
    args = build_parser().parse_args()
    set_threads(args.max_procs)

    import numpy as np
    import pymaster as nmt
    from spt3g import core, maps  # noqa: F401  (registers FlatSkyMap for G3File)

    purify_b = parse_purify_b(args.purify_b)
    pixel_window_mode = parse_pixel_window(args.pixel_window)
    lmax = parse_lmax(args.lmax)
    delta_ell = args.delta_ell
    nbins = lmax // delta_ell

    l0 = (np.arange(nbins) * delta_ell).astype(float)
    lf = ((np.arange(nbins) + 1) * delta_ell).astype(float)
    b = nmt.NmtBinFlat(l0, lf)
    ell_eff = b.get_effective_ells()

    workspaces_dir = Path(args.workspaces)
    ws_manifest, ws_by_proj, mask_indices = load_workspace_manifest(
        workspaces_dir, list(args.projections)
    )
    if ws_manifest["lmax"] != lmax or ws_manifest["delta_ell"] != delta_ell:
        raise ValueError(
            f"--lmax/--delta-ell ({lmax}, {delta_ell}) do not match the "
            f"coupling_matrices manifest ({ws_manifest['lmax']}, "
            f"{ws_manifest['delta_ell']}) -- rebuild coupling_matrices first"
        )
    if ws_manifest["purify_b"] != purify_b:
        raise ValueError(
            f"--purify-b={purify_b} does not match the coupling_matrices "
            f"manifest ({ws_manifest['purify_b']}) -- the w22 workspace was "
            "built assuming a specific purify_b and the two must agree"
        )

    pixwin_correction = compute_pixel_window(pixel_window_mode, b, ell_eff, lmax)

    qu_maps_dir = Path(args.qu_maps)
    eb_maps_dir = Path(args.eb_maps)
    masks_dir = Path(args.masks)
    out_dir = Path(args.out)
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_stop))
    nseed = len(seeds)
    projections = list(args.projections)

    for leg in LEGS:
        for proj in projections:
            for i in mask_indices:
                mask, lo, hi, lx_rad, ly_rad = load_mask(masks_dir, proj, i)

                ws_path = workspaces_dir / ws_by_proj[proj][i]["w22" if leg == "qu" else "w00"]
                w = nmt.NmtWorkspaceFlat()
                w.read_from(str(ws_path))
                if w.wsp.bin.n_bands != nbins:
                    raise ValueError(
                        f"{ws_path} has {w.wsp.bin.n_bands} bands, expected "
                        f"{nbins} -- built with a different --lmax/--delta-ell"
                    )

                partial_path = partial_dir / f"{leg}_{proj}_mask{i}.npy"
                arr = load_partial(partial_path, (2, nseed, nbins))

                for si, seed in enumerate(seeds):
                    if not np.isnan(arr[:, si, :]).any():
                        continue  # already computed by an earlier, killed run

                    if leg == "qu":
                        frame_path = qu_maps_dir / f"qu_{proj}_seed{seed:04d}.g3"
                        frame = next(core.G3File(str(frame_path)))
                        q = finite_in_mask(
                            np.asarray(frame["Q"], dtype=np.float64)[lo:hi, lo:hi], mask
                        )
                        u = finite_in_mask(
                            np.asarray(frame["U"], dtype=np.float64)[lo:hi, lo:hi], mask
                        )
                        ee, bb = compute_qu_bandpowers(
                            q, u, mask, lx_rad, ly_rad, purify_b, b, w
                        )
                    else:
                        frame_path = eb_maps_dir / f"eb_{proj}_seed{seed:04d}.g3"
                        frame = next(core.G3File(str(frame_path)))
                        e = finite_in_mask(
                            np.asarray(frame["E"], dtype=np.float64)[lo:hi, lo:hi], mask
                        )
                        bmap = finite_in_mask(
                            np.asarray(frame["B"], dtype=np.float64)[lo:hi, lo:hi], mask
                        )
                        ee, bb = compute_eb_bandpowers(
                            e, bmap, mask, lx_rad, ly_rad, b, w
                        )

                    arr[0, si] = ee / pixwin_correction
                    arr[1, si] = bb / pixwin_correction
                    atomic_save(partial_path, value=arr)
                    print(
                        f"{leg} {proj} mask{i} seed{seed:04d} done", flush=True
                    )

                assert not np.isnan(arr).any(), (
                    f"{partial_path} incomplete after processing all requested "
                    "seeds -- this should not happen"
                )

    # Assemble the one canonical output from the (now-complete) partials.
    ee = np.empty((len(LEGS), len(projections), len(mask_indices), nseed, nbins))
    bb = np.empty_like(ee)
    for li, leg in enumerate(LEGS):
        for pi, proj in enumerate(projections):
            for mi, i in enumerate(mask_indices):
                arr = np.load(partial_dir / f"{leg}_{proj}_mask{i}.npy")
                ee[li, pi, mi] = arr[0]
                bb[li, pi, mi] = arr[1]

    atomic_save(
        out_dir / "bandpowers.npz",
        ell_eff=ell_eff,
        ee=ee,
        bb=bb,
        legs=np.array(LEGS),
        projections=np.array(projections),
        masks=np.array(mask_indices),
        seeds=np.array(seeds),
    )

    write_index(
        out_dir,
        {
            "bandpowers": "bandpowers.npz",
            "legs": list(LEGS),
            "projections": projections,
            "masks": mask_indices,
            "seeds": seeds,
            "lmax": lmax,
            "delta_ell": delta_ell,
            "nbins": nbins,
            "purify_b": purify_b,
            "pixel_window": pixel_window_mode,
        },
    )


if __name__ == "__main__":
    main()
