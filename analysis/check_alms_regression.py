#!/usr/bin/env python
"""Regress the ported preprocessing stage against the 2021 alms, seed by seed.

The 2021 intermediates all survive, so the port is validated stage by stage
rather than end to end. This check compares one seed of the new stage's output
(`alms_seed{N:04d}.fits`) against the 2021 `almsTQU_seed{N}.fits` produced from
the same input FITS with the same healpy defaults (iter=3, lmax=4000).

Expected disagreement, and only this one: the 2021 monopole/dipole filter was
built two entries short and healpy zero-padded it, so the 2021 alms are missing
l = lmax-1 and l = lmax. The new stage filters only l = 0, 1. Everything below
l = lmax-1 must agree to float32 round-off; at the top two multipoles the 2021
alms must be exactly zero and the new ones must not be.

Usage:
    env-shell.sh python analysis/check_alms_regression.py \
        --new /sptlocal/user/cailmdaley/pol_project/regression/alms_l4000 \
        --old /sptlocal/analysis/pol_project2020/results/preprocessed_alms \
        --seed 0 --lmax 4000
"""

from __future__ import annotations

import argparse

import healpy as hp
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lmax", type=int, default=4000)
    args = ap.parse_args()

    lmax = args.lmax
    ok = True
    for i, name in enumerate("TEB"):
        new = hp.read_alm(f"{args.new}/alms_seed{args.seed:04d}.fits", hdu=i + 1)
        old = hp.read_alm(f"{args.old}/almsTQU_seed{args.seed}.fits", hdu=i + 1)
        assert new.shape == old.shape, (name, new.shape, old.shape)

        ls, _ = hp.Alm.getlm(lmax)
        body = ls < lmax - 1          # below the 2021 filter defect
        top = ls >= lmax - 1          # l = lmax-1, lmax

        scale = np.sqrt(np.mean(np.abs(old[body]) ** 2))
        diff = np.max(np.abs(new[body] - old[body])) / scale
        old_top = np.max(np.abs(old[top]))
        new_top = np.sqrt(np.mean(np.abs(new[top]) ** 2)) / scale

        body_ok = diff < 1e-6
        top_ok = old_top == 0.0 and new_top > 1e-3
        ok &= body_ok and top_ok
        print(
            f"{name}: body max|Δ|/rms = {diff:.3e} [{'OK' if body_ok else 'FAIL'}]"
            f"   top-two-l: 2021 max = {old_top:.1e}, new rms/rms = {new_top:.3f}"
            f" [{'OK' if top_ok else 'FAIL'}]"
        )

    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
