"""Translate ASTRA decision option-ids into the numbers the stages need.

Lightcone substitutes ``{decisions.<id>}`` with the *option id* — the literal
string ``l4000``, not ``4000`` — so every stage flag that carries a decision has
to accept that token. Each parser here also accepts the bare value, so the same
scripts run by hand without knowing anything about ASTRA.

Keep this the only place the mapping lives.
"""

from __future__ import annotations

import argparse

__all__ = [
    "parse_lmax",
    "parse_nside",
    "parse_interpolation",
    "parse_apodization",
    "parse_purify_b",
    "parse_pixel_window",
    "parse_cosmology",
    "parse_flattening",
    "add_common_args",
    "set_threads",
]


def _bad(flag: str, value: str, allowed) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(
        f"{flag}: cannot interpret {value!r}; expected one of {sorted(allowed)} "
        "or a bare value"
    )


def parse_lmax(value: str) -> int:
    """``l4000`` | ``l6000`` | ``4000`` -> int."""
    text = str(value).strip()
    if text.startswith("l") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    raise _bad("--lmax", value, {"l4000", "l6000"})


def parse_delta_ell(value: str) -> int:
    """``d10`` | ``d50`` | ``10`` -> int."""
    text = str(value).strip()
    if text.startswith("d") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    raise _bad("--delta-ell", value, {"d10", "d50"})


def parse_nside(value: str) -> int:
    """``nside4096`` | ``nside8192`` | ``4096`` -> int."""
    text = str(value).strip()
    if text.startswith("nside") and text[5:].isdigit():
        return int(text[5:])
    if text.isdigit():
        return int(text)
    raise _bad("--nside", value, {"nside4096", "nside8192"})


def parse_interpolation(value: str) -> dict:
    """-> ``{"interp": bool, "rebin": int}``.

    ``interp_rebin8``: bilinear interpolation onto a grid oversampled 8x,
    then averaged down. ``nearest`` turns interpolation off and
    measures the bare pixelization floor.
    """
    table = {
        "interp_rebin8": {"interp": True, "rebin": 8},
        "interp_rebin1": {"interp": True, "rebin": 1},
        "nearest": {"interp": False, "rebin": 1},
    }
    text = str(value).strip()
    if text in table:
        return dict(table[text])
    raise _bad("--interpolation", value, table)


def parse_apodization(value: str) -> dict:
    """-> ``{"apotype": "C1"|"C2", "aposize": float}`` in degrees."""
    table = {
        "c2_2p5deg": {"apotype": "C2", "aposize": 2.5},
        "c1_2p5deg": {"apotype": "C1", "aposize": 2.5},
        "c2_5deg": {"apotype": "C2", "aposize": 5.0},
    }
    text = str(value).strip()
    if text in table:
        return dict(table[text])
    raise _bad("--apodization", value, table)


def parse_purify_b(value: str) -> bool:
    """``purify`` | ``standard`` | ``true`` | ``false`` -> bool."""
    text = str(value).strip().lower()
    if text in {"purify", "true", "1", "yes"}:
        return True
    if text in {"standard", "false", "0", "no"}:
        return False
    raise _bad("--purify-b", value, {"purify", "standard"})


def parse_pixel_window(value: str) -> str:
    """``none`` | ``flat_only`` | ``healpix_only`` | ``healpix_and_flat`` -> itself."""
    text = str(value).strip()
    allowed = {"none", "flat_only", "healpix_only", "healpix_and_flat"}
    if text in allowed:
        return text
    raise _bad("--pixel-window", value, allowed)


# Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing best fit: the chi^2
# minimum of that chain, not its posterior mean (mean ombh2 = 0.02237 against
# the best fit's 0.0223828). The numbers are those of
# planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini, the CAMB
# parameter file in spt3g_software's simulations/ tree; CAMB 1.5.4 at
# default accuracy with them reproduces that file's unlensed EE to 0.03%
# over 100 < ell < 3000. One massive neutrino of 0.06 eV is omnuh2 =
# 6.451439e-4.
COSMOLOGIES = {
    "planck2018": dict(
        H0=67.32117,
        ombh2=0.0223828,
        omch2=0.1201075,
        omk=0.0,
        tau=0.05430842,
        mnu=0.06,
        num_massive_neutrinos=1,
        nnu=3.046,
        As=2.100549e-9,
        ns=0.9660499,
        pivot_scalar=0.05,
    ),
}


def parse_cosmology(value: str) -> dict:
    """``planck2018`` -> its CAMB parameters, a fresh dict."""
    text = str(value).strip()
    if text in COSMOLOGIES:
        return dict(COSMOLOGIES[text])
    raise _bad("--cosmology", value, COSMOLOGIES)


def parse_flattening(value: str) -> str:
    """``polar`` | ``spt3g`` -> itself."""
    text = str(value).strip()
    if text in {"polar", "spt3g"}:
        return text
    raise _bad("--flattening", value, {"polar", "spt3g"})


def set_threads(max_procs: int) -> None:
    """Bound every threading runtime this stack can reach.

    Call this *before* importing numpy, healpy or spt3g. amundsen is a shared
    24-core login node and the project constraint is a couple of cores.
    """
    import os

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(max_procs)


def add_common_args(parser: argparse.ArgumentParser, *, seeds: bool = False) -> None:
    """Flags every stage shares."""
    parser.add_argument("--out", required=True, help="output DIRECTORY (created)")
    parser.add_argument(
        "--max-procs",
        type=int,
        default=4,
        help="thread cap; also the process-pool size where a stage forks (default 4)",
    )
    if seeds:
        parser.add_argument("--seed-start", type=int, default=0)
        parser.add_argument("--seed-stop", type=int, default=20)
        parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=None,
            help="explicit seed list; overrides --seed-start/--seed-stop. "
            "This is how a per-unit grid job names its single seed "
            "(recipes say --seeds {units.seed}).",
        )


def resolve_seeds(args) -> list:
    """The seed list a stage should process: ``--seeds`` wins outright."""
    if getattr(args, "seeds", None) is not None:
        return list(dict.fromkeys(args.seeds))
    return list(range(args.seed_start, args.seed_stop))


def index_name(seeds=None, projections=None, footprints=None, tag=None) -> str:
    """Name of the stage's index file.

    ``index.json`` for a whole-output run; a per-shard run (a unit job,
    which names its coordinates explicitly -- ``--seeds`` and/or
    ``--footprint``) gets a name keyed by those coordinates, prefixed by
    ``tag`` when one output carries several products (the scalar
    spectra), so shard indexes never collide when the unit gather
    assembles the canonical directory. Consumers that need the whole
    picture read every shard with :func:`read_index_entries`.
    """
    if seeds is None and footprints is None:
        return "index.json"
    parts = ["index"]
    if tag is not None:
        parts.append(tag)
    if seeds is not None:
        parts += [f"{s:04d}" for s in seeds]
    if projections:
        parts += list(projections)
    if footprints is not None:
        parts += list(footprints)
    return "_".join(parts) + ".json"


def write_index(out_dir, payload: dict, *, name: str = "index.json") -> None:
    """Write ``<out>/index.json``, the manifest downstream stages read."""
    import json
    from pathlib import Path

    path = Path(out_dir) / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_index(in_dir) -> dict:
    import json
    from pathlib import Path

    return json.loads((Path(in_dir) / "index.json").read_text())


def read_index_entries(in_dir, key: str) -> tuple[dict, list]:
    """Merge the ``key`` lists of every index file in ``in_dir``.

    A whole-output stage writes one ``index.json``; a unit-fanned stage
    leaves one ``index_<coords>.json`` per shard after the gather. Both
    layouts read the same way here: the scalar fields (binning, purify_b,
    ...) come from the first file and are checked equal across all of
    them, and the ``key`` lists are concatenated in filename order.
    """
    import json
    from pathlib import Path

    paths = sorted(Path(in_dir).glob("index*.json"))
    if not paths:
        raise FileNotFoundError(f"no index*.json under {in_dir}")
    head: dict | None = None
    entries: list = []
    for path in paths:
        manifest = json.loads(path.read_text())
        entries.extend(manifest.get(key, []))
        scalars = {k: v for k, v in manifest.items() if not isinstance(v, (list, dict))}
        if head is None:
            head = scalars
        elif scalars != head:
            raise ValueError(
                f"index files under {in_dir} disagree on scalar fields: "
                f"{path.name} has {scalars}, {paths[0].name} has {head}"
            )
    return dict(head or {}), entries
