"""Gaussian skies, one per seed, as spherical-harmonic coefficients.

Every projection leg of a seed starts from the same alms, so the legs differ
by projection alone. Two spectra:

``--spectrum cmb``: T, E and B drawn with ``healpy.synalm`` from the unlensed
TT, EE and TE of ``theory_spectra``, with BB = 0, so B is identically zero
and every B the pipeline recovers is leakage. The theory spectra are zero at
ell = 0 and 1, so the draw has no monopole or dipole.

``--spectrum white``: one scalar sky with C_ell = 1 for 2 <= ell <= ell_max,
a spectrum with neither the CMB's slope nor its acoustic phases.

Both are band-limited at ``--lmax``. The draw is fixed by the seed: numpy's
global generator, which synalm uses, is seeded from ``SeedSequence((NAMESPACE,
seed))``, with a namespace per spectrum so the CMB and white skies of one
seed are independent. Written with ``hp.write_alm`` to
``<out>/alms_seed{N:04d}.fits``, one HDU per component (T, E, B for cmb; T
for white), through a temporary file renamed into place.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _decisions import add_common_args, index_name, parse_lmax, resolve_seeds, set_threads, write_index

NAMESPACE = {"cmb": 20261115, "white": 20261105}


def spectra(spectrum: str, lmax: int, theory: Path | None):
    """The C_ell synalm draws from: one array (white), or TT, EE, BB, TE (cmb)."""
    import numpy as np

    if spectrum == "white":
        cl = np.ones(lmax + 1)
        cl[:2] = 0.0
        return cl
    with np.load(theory / "theory_spectra.npz") as z:
        tt, ee, te = (z[f"unlensed_{k}"][:lmax + 1] for k in ("TT", "EE", "TE"))
    return [tt, ee, np.zeros(lmax + 1), te]   # TT, EE, BB, TE


def draw(cls, lmax: int, spectrum: str, seed: int):
    import healpy as hp
    import numpy as np

    np.random.seed(np.random.SeedSequence((NAMESPACE[spectrum], seed)).generate_state(1)[0])
    return hp.synalm(cls, lmax=lmax, new=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spectrum", choices=sorted(NAMESPACE), required=True)
    p.add_argument("--theory", type=Path, help="theory_spectra output directory (cmb only)")
    p.add_argument("--lmax", required=True, type=parse_lmax)
    add_common_args(p, seeds=True)
    args = p.parse_args(argv)
    set_threads(args.max_procs)

    import healpy as hp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cls = spectra(args.spectrum, args.lmax, args.theory)
    seeds = resolve_seeds(args)
    for seed in seeds:
        path = out / f"alms_seed{seed:04d}.fits"
        if path.exists():
            continue
        t0 = time.time()
        alms = draw(cls, args.lmax, args.spectrum, seed)
        tmp = out / f"alms_seed{seed:04d}.tmp.fits"
        hp.write_alm(str(tmp), alms, overwrite=True)
        tmp.replace(path)
        print(f"seed {seed}: {path.name} in {time.time() - t0:.1f}s", flush=True)
    write_index(out, {"stage": "sky_alms", "spectrum": args.spectrum, "lmax": args.lmax,
                      "components": ["T", "E", "B"] if args.spectrum == "cmb" else ["T"],
                      "seed_namespace": NAMESPACE[args.spectrum], "seeds": seeds},
                name=index_name(args.seeds))


if __name__ == "__main__":
    main()
