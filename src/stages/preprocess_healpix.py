#!/usr/bin/env python
"""Turn Healpix TQU realizations into band-limited T/E/B spherical harmonics.

Two downstream stages -- the Q/U leg and the E/B leg of the flat-sky
projection test -- both need to start from the *same* sky. If each stage
ran its own map2alm, the two legs would carry independent transform noise
on top of the projection effect we actually want to isolate. This stage
runs the spherical harmonic transform once per realization and both legs
read its output, so any difference measured downstream is projection, not
two different draws of the same random process.

Each input file is a Healpix Nside 8192, RING-ordered TQU map (a FITS binary
table with TEMPERATURE, Q_POLARISATION, U_POLARISATION columns). hp.map2alm
turns [T, Q, U] into [alm_T, alm_E, alm_B]. The monopole (l=0, an arbitrary
zero-point with no cosmological content) and dipole (l=1, dominated by the
Solar System's own motion through the CMB rest frame) are zeroed by
convention -- this only does real work on T, since E and B are spin-2
quantities and have no l<2 modes to begin with.

This stage has no Healpix resolution of its own: whatever grid the input
maps were made at is what gets transformed, truncated at --lmax. Nside is
read from the FITS data itself, never assumed.

Resumability matters here: one input file is 9.7 GB and map2alm at Nside
8192 takes several minutes, so a run killed partway through must not have
to redo the seeds it already finished. Each seed's alms are written through
a temporary file and renamed into place, so a kill can never leave a
half-written file that a resumed run would mistake for a finished one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _decisions import (  # noqa: E402
    add_common_args,
    index_name,
    parse_lmax,
    resolve_seeds,
    set_threads,
    write_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute T/E/B alms from Healpix TQU realizations.",
    )
    parser.add_argument(
        "--sims-dir", required=True,
        help="directory of TQU*_..._seed{N}.fits realizations "
             "(unlensed_TQU1 or unlensed_TQU2)",
    )
    parser.add_argument(
        "--lmax", required=True, type=parse_lmax,
        help="maximum multipole to keep (accepts l4000/l6000 or a bare integer)",
    )
    parser.add_argument(
        "--iter", dest="n_iter", type=int, default=3,
        help="map2alm Jacobi iterations (default 3, healpy's default and what "
             "the 2021 run used implicitly; 0 is four times cheaper)",
    )
    add_common_args(parser, seeds=True)
    return parser


CGROUP_MAX = Path("/sys/fs/cgroup/user.slice")
PEAK_RSS_GIB = {0: 45.0, 3: 49.0}  # measured at Nside 8192; see check_memory_headroom


def check_memory_headroom(n_iter: int) -> None:
    """Refuse to start if the user's memory cgroup cannot hold one transform.

    amundsen caps memory per UNIX *user*, not per process — 60 GiB shared across
    every process the account runs, agents and notebooks included. One map2alm at
    Nside 8192 peaks at 45 GB with iter=0 and 49 GB with iter=3, so a single seed
    nearly fills the cap by itself. Going over is a SIGKILL: no traceback, no log,
    an empty output directory, and the only evidence in `dmesg -T | grep oom`.
    That failure is expensive precisely because it looks like nothing happened,
    so it is worth ten lines to fail loudly at the start instead.
    """
    import os

    uid = os.getuid()
    slice_dir = CGROUP_MAX / f"user-{uid}.slice"
    try:
        limit = int((slice_dir / "memory.max").read_text().strip())
        current = int((slice_dir / "memory.current").read_text().strip())
    except (OSError, ValueError):
        return  # no cgroup here; nothing to check

    gib = 1024 ** 3
    headroom = (limit - current) / gib
    needed = PEAK_RSS_GIB.get(n_iter, PEAK_RSS_GIB[3])
    print(f"memory cgroup: {limit / gib:.0f} GiB cap, {current / gib:.1f} GiB in "
          f"use, {headroom:.1f} GiB free; this stage peaks near {needed:.0f} GiB")
    if headroom < needed:
        raise SystemExit(
            f"refusing to start: {headroom:.1f} GiB free under the {limit / gib:.0f} "
            f"GiB per-user cgroup, but map2alm at Nside 8192 with iter={n_iter} "
            f"peaks near {needed:.0f} GiB. Stop other work of yours on this host "
            "(agents, notebooks, background jobs) and retry, or run this on the grid."
        )


def find_input_fits(sims_dir: str, seed: int) -> Path:
    """Locate the FITS file for one seed without hardcoding the filename prefix.

    ``--sims-dir`` points at either unlensed_TQU1 (prefix ``TQU1_``) or
    unlensed_TQU2 (prefix ``TQU2_``). Rather than assume which, glob for the
    seed suffix and demand exactly one hit -- a wrong or ambiguous
    --sims-dir should fail immediately, not silently pick a file.
    """
    matches = sorted(Path(sims_dir).glob(f"*seed{seed}.fits"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one file matching '*seed{seed}.fits' in "
            f"{sims_dir!r}, found {len(matches)}: {matches}"
        )
    return matches[0]


def monopole_dipole_filter(lmax: int):
    """almxfl filter of length lmax + 1: zero l = 0, 1; pass l = 2 .. lmax.

    The 2021 script built this filter as
    ``np.concatenate([[0, 0], np.ones(lmax - 2)])``, which has length
    lmax, one short of the lmax + 1 entries almxfl expects (one per l
    from 0 to lmax). healpy's almxfl silently zero-pads a short filter out
    to lmax, so the 2021 filter didn't just zero the monopole and dipole --
    it also zeroed l = lmax. Verified in the seed-0 regression (2026-08-15):
    the 2021 alms agree with this stage's to round-off at every l up to and
    including lmax - 1, and are exactly zero at l = lmax. Anyone reproducing
    the 2021 bandpowers exactly needs to know the top bandpower there is
    missing its top multipole; this filter reproduces the *intent* (monopole
    and dipole only) rather than the bug.
    """
    import numpy as np

    filt = np.ones(lmax + 1)
    filt[:2] = 0.0
    return filt


def alms_for_seed(input_path: Path, lmax: int, filt, n_iter: int = 3):
    """Load one TQU realization and return its filtered [alm_T, alm_E, alm_B].

    ``n_iter`` is healpy's Jacobi iteration count, and it is the dominant cost
    knob: the transform runs ``n_iter + 1`` times. Measured on amundsen at Nside
    8192 with four threads, one ``iter=0`` transform is 131 s at lmax 4000 and
    243 s at lmax 6000, so the default ``iter=3`` costs roughly four times that.
    The default is kept because it is what the 2021 run used (it never passed
    ``iter``, so it inherited healpy's 3), and quadrature error at Nside 8192
    without iteration is not negligible at the top of the band.
    """
    import numpy as np
    import healpy as hp
    from spt3g.maps.fitsio import load_skymap_fits

    frame = load_skymap_fits(str(input_path), keys="TQU")
    tqu = [np.asarray(frame[key]) for key in "TQU"]
    alms = hp.map2alm(tqu, lmax=lmax, pol=True, iter=n_iter)
    for alm in alms:
        hp.almxfl(alm, filt, inplace=True)
    return alms


def write_alms_atomically(alms, out_path: Path) -> None:
    """Write alms through a temporary file, then rename into place.

    A process killed mid-write must never leave a file at `out_path` --
    the resumability check that skips finished seeds only looks at whether
    that path exists, so a truncated file there would be mistaken for a
    finished one forever. Renaming a completed temporary file is atomic on
    a POSIX filesystem, so `out_path` only ever appears once its contents
    are complete.
    """
    import healpy as hp

    tmp_path = out_path.with_name(out_path.stem + ".tmp.fits")
    tmp_path.unlink(missing_ok=True)
    hp.write_alm(str(tmp_path), alms)
    tmp_path.replace(out_path)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # Must precede importing numpy/healpy/spt3g: this is what bounds the
    # threaded runtimes those libraries spin up.
    set_threads(args.max_procs)

    check_memory_headroom(args.n_iter)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    filt = monopole_dipole_filter(args.lmax)

    seeds = resolve_seeds(args)
    seed_records = []
    for seed in seeds:
        out_path = out_dir / f"alms_seed{seed:04d}.fits"
        input_path = find_input_fits(args.sims_dir, seed)

        already_done = (
            out_path.exists()
            and out_path.stat().st_mtime >= input_path.stat().st_mtime
        )
        if already_done:
            print(f"seed {seed:4d}: skip, {out_path.name} already exists")
        else:
            t_start = time.time()
            alms = alms_for_seed(input_path, args.lmax, filt, args.n_iter)
            write_alms_atomically(alms, out_path)
            wall_time = time.time() - t_start
            print(f"seed {seed:4d}: wrote {out_path.name} in {wall_time:7.1f}s")
        sys.stdout.flush()

        seed_records.append({"seed": seed, "alms": out_path.name})

    write_index(out_dir, {
        "stage": "preprocess_healpix",
        "sims_dir": str(args.sims_dir),
        "lmax": args.lmax,
        "n_iter": args.n_iter,
        "seeds_processed": seeds,
        "max_procs": args.max_procs,
        "seeds": seed_records,
    }, name=index_name(args.seeds))


if __name__ == "__main__":
    main()
