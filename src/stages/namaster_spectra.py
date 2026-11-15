#!/usr/bin/env python
"""NaMaster flat-sky EE/BB bandpowers for both the Q/U and E/B legs.

Two legs, two different NaMaster objects
-----------------------------------------
The Q/U leg (``qu_{proj}_seed{N:04d}.g3``, keys ``Q``/``U``) is a genuine
spin-2 field: Q and U were built from the sky's actual polarization tensor,
flattened with ``FlattenPol`` in ``project_qu.py`` (already applied -- this
stage must *not* call it again). It goes into
``NmtFieldFlat(lx, ly, mask, [q, u], purify_b=...)`` and is decoupled with the
spin-2 workspace (``w22_{proj}_{footprint}.fits``, ``ncls=4``, order EE/EB/BE/BB);
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
pymaster's ``field.py`` before writing this).

So the E/B leg is two independent spin-0 auto-spectra:
``NmtFieldFlat(lx, ly, mask, [E])`` and ``NmtFieldFlat(lx, ly, mask, [B])``,
each with ``purify_e=purify_b=False`` (undefined for spin-0), each decoupled
through the *scalar* workspace ``w00_{proj}_{footprint}.fits`` that
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
  ``NmtBinFlat.bin_cell``. The skies are drawn in harmonic space
  (``sky_alms.py``) and synthesized as point values, so they carry no such
  window; this mode over-corrects and exists only for the explicit
  comparison.
* ``healpix_and_flat``: both of the above; over-corrects for the same
  reason.

The pixel size is fixed for the whole analysis (``_footprints.RES_ARCMIN``),
not a per-run decision, so it is a constant rather than a flag.

Footprints
----------
Every footprint the workspaces index lists is measured, in registry order
(``_footprints.FOOTPRINTS``), and the output arrays carry a footprint axis
whose names are stored as ``footprints``.

Resumable: bandpowers are checkpointed per (leg, projection, footprint) into
``<out>/partial/{leg}_{proj}_{footprint}.npy``, an array of shape
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
    index_name,
    parse_delta_ell,
    parse_lmax,
    parse_pixel_window,
    parse_purify_b,
    read_index_entries,
    resolve_seeds,
    set_threads,
    write_index,
)
from _footprints import RES_ARCMIN, ordered

LEGS = ("qu", "eb")

# The Nside of the healpix_* pixel-window modes, which over-correct and exist
# for the comparison only.
INPUT_HEALPIX_NSIDE = 8192


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
    """(manifest scalars, {proj: {footprint: entry}}, footprints in registry order).

    The workspaces arrive as one ``index_<proj>_<footprint>.json`` per grid
    unit; ``read_index_entries`` merges them and checks they agree on
    binning and purification. Every projection must carry the same set of
    footprints.
    """
    manifest, workspaces = read_index_entries(workspaces_dir, "workspaces")
    by_proj: dict[str, dict[str, dict]] = {}
    for entry in workspaces:
        if entry["projection"] in projections:
            by_proj.setdefault(entry["projection"], {})[entry["footprint"]] = entry

    missing = [p for p in projections if p not in by_proj]
    if missing:
        raise FileNotFoundError(
            f"no workspaces found for projection(s) {missing} under {workspaces_dir}"
        )

    sets = {p: frozenset(by_proj[p]) for p in projections}
    if len(set(sets.values())) > 1:
        raise ValueError(
            "projections do not share the same set of footprints: "
            f"{ {p: sorted(s) for p, s in sets.items()} }"
        )
    return manifest, by_proj, ordered(sets[projections[0]])


def load_mask(masks_dir: Path, proj: str, footprint: str):
    """(mask, window, lx_rad, ly_rad); `window` slices the cut out of a full map."""
    import numpy as np

    d = np.load(Path(masks_dir) / f"mask_{proj}_{footprint}.npz")
    mask = np.asarray(d["mask"], dtype=np.float64)
    rows = slice(int(d["ylo"]), int(d["yhi"]))
    cols = slice(int(d["xlo"]), int(d["xhi"]))
    lx_rad, ly_rad = float(d["lx_rad"]), float(d["ly_rad"])
    return mask, (rows, cols), lx_rad, ly_rad


def finite_in_mask(m, mask):
    """Zero non-finite pixels, requiring that none carry mask weight.

    A projection can leave grid pixels with no sky behind them (NaN), as an
    orthographic grid does beyond 1 rad from its centre. NaN * 0 is NaN and
    NaMaster then returns an all-NaN spectrum; zeroing them is exact because
    the mask weight there is zero. A NaN *inside* the mask would be a real problem, so it raises.
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
        # The skies carry no HEALPix window, so these modes over-correct;
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
    ws_manifest, ws_by_proj, footprints = load_workspace_manifest(
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
    seeds = resolve_seeds(args)
    nseed = len(seeds)
    projections = list(args.projections)
    # A unit job (explicit --seeds) writes its bandpowers under a name keyed
    # by its coordinates, so the unit gather can hardlink every shard into
    # one canonical directory without collision, the per-(projection, seed)
    # layout analysis/ reads. A whole-output run keeps the single bandpowers.npz.
    unit = args.seeds is not None
    stem = (
        "bandpowers_" + "_".join(projections) + "_seed" + "_".join(f"{s:04d}" for s in seeds)
        if unit
        else "bandpowers"
    )
    partial_dir = out_dir / ("partial" if not unit else f"partial_{stem}")
    partial_dir.mkdir(parents=True, exist_ok=True)

    for leg in LEGS:
        for proj in projections:
            for fp in footprints:
                mask, window, lx_rad, ly_rad = load_mask(masks_dir, proj, fp)

                ws_path = workspaces_dir / ws_by_proj[proj][fp]["w22" if leg == "qu" else "w00"]
                w = nmt.NmtWorkspaceFlat()
                w.read_from(str(ws_path))
                if w.wsp.bin.n_bands != nbins:
                    raise ValueError(
                        f"{ws_path} has {w.wsp.bin.n_bands} bands, expected "
                        f"{nbins} -- built with a different --lmax/--delta-ell"
                    )

                partial_path = partial_dir / f"{leg}_{proj}_{fp}.npy"
                arr = load_partial(partial_path, (2, nseed, nbins))

                for si, seed in enumerate(seeds):
                    if not np.isnan(arr[:, si, :]).any():
                        continue  # already computed by an earlier, killed run

                    if leg == "qu":
                        frame_path = qu_maps_dir / f"qu_{proj}_seed{seed:04d}.g3"
                        frame = next(core.G3File(str(frame_path)))
                        q = finite_in_mask(
                            np.asarray(frame["Q"], dtype=np.float64)[window], mask
                        )
                        u = finite_in_mask(
                            np.asarray(frame["U"], dtype=np.float64)[window], mask
                        )
                        ee, bb = compute_qu_bandpowers(
                            q, u, mask, lx_rad, ly_rad, purify_b, b, w
                        )
                    else:
                        frame_path = eb_maps_dir / f"eb_{proj}_seed{seed:04d}.g3"
                        frame = next(core.G3File(str(frame_path)))
                        e = finite_in_mask(
                            np.asarray(frame["E"], dtype=np.float64)[window], mask
                        )
                        bmap = finite_in_mask(
                            np.asarray(frame["B"], dtype=np.float64)[window], mask
                        )
                        ee, bb = compute_eb_bandpowers(
                            e, bmap, mask, lx_rad, ly_rad, b, w
                        )

                    arr[0, si] = ee / pixwin_correction
                    arr[1, si] = bb / pixwin_correction
                    atomic_save(partial_path, value=arr)
                    print(
                        f"{leg} {proj} {fp} seed{seed:04d} done", flush=True
                    )

                assert not np.isnan(arr).any(), (
                    f"{partial_path} incomplete after processing all requested "
                    "seeds -- this should not happen"
                )

    # Assemble the one canonical output from the (now-complete) partials.
    ee = np.empty((len(LEGS), len(projections), len(footprints), nseed, nbins))
    bb = np.empty_like(ee)
    for li, leg in enumerate(LEGS):
        for pi, proj in enumerate(projections):
            for fi, fp in enumerate(footprints):
                arr = np.load(partial_dir / f"{leg}_{proj}_{fp}.npy")
                ee[li, pi, fi] = arr[0]
                bb[li, pi, fi] = arr[1]

    atomic_save(
        out_dir / f"{stem}.npz",
        ell_eff=ell_eff,
        ee=ee,
        bb=bb,
        legs=np.array(LEGS),
        projections=np.array(projections),
        footprints=np.array(footprints),
        seeds=np.array(seeds),
    )

    if unit:
        import shutil

        shutil.rmtree(partial_dir)  # shards must not collide at gather

    write_index(
        out_dir,
        {
            "bandpowers": f"{stem}.npz",
            "legs": list(LEGS),
            "projections": projections,
            "footprints": footprints,
            "seeds": seeds,
            "lmax": lmax,
            "delta_ell": delta_ell,
            "nbins": nbins,
            "purify_b": purify_b,
            "pixel_window": pixel_window_mode,
        },
        name=index_name(seeds if unit else None, projections),
    )


if __name__ == "__main__":
    main()
