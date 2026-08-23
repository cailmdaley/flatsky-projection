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

    ``interp_rebin8`` is the 2021 setting: bilinear interpolation onto a grid
    oversampled 8x, then averaged down. ``nearest`` turns interpolation off and
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
    """``none`` | ``healpix_only`` | ``healpix_and_flat`` -> itself."""
    text = str(value).strip()
    if text in {"none", "healpix_only", "healpix_and_flat"}:
        return text
    raise _bad("--pixel-window", value, {"none", "healpix_only", "healpix_and_flat"})


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


def write_index(out_dir, payload: dict) -> None:
    """Write ``<out>/index.json``, the manifest downstream stages read."""
    import json
    from pathlib import Path

    path = Path(out_dir) / "index.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_index(in_dir) -> dict:
    import json
    from pathlib import Path

    return json.loads((Path(in_dir) / "index.json").read_text())
