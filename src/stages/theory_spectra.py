"""CMB power spectra of the cosmology decision, from CAMB.

Writes ``<out>/theory_spectra.npz``: lensed and unlensed TT, EE, BB and TE as
raw C_l in uK^2, indexed by multipole from ell = 0 to LMAX, and the run's
parameters in a pickled ``header`` (``np.load(..., allow_pickle=True)
["header"][()]``). Every stage and analysis script that needs a theory curve
reads this one file and slices what it uses; the simulated skies are drawn
from its unlensed TT, EE and TE.

Conventions:
  - raw C_l, not D_l = l(l+1)C_l/2pi; convert at plot time.
  - indices 0 and 1 are zero, as CAMB's raw_cl returns them, so every array
    can be indexed directly by ell.
  - LMAX = 10000 reaches past any band limit the pipeline uses, so one file
    serves every consumer.
  - lens_potential_accuracy = 2 and CAMB's default NonLinear_lens (halofit on
    the lensing potential only): the lensed BB amplitude depends on both at
    the several-percent level. No tensors, so unlensed BB is zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _decisions import parse_cosmology, write_index

LMAX = 10000
SPECTRA = ["TT", "EE", "BB", "TE"]  # CAMB's raw_cl column order


def generate(cosmology: dict, lmax: int = LMAX) -> dict:
    """Raw C_l [uK^2] from ell = 0, lensed and unlensed, and a header."""
    import camb
    import numpy as np

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=cosmology["H0"], ombh2=cosmology["ombh2"], omch2=cosmology["omch2"],
        omk=cosmology["omk"], tau=cosmology["tau"], mnu=cosmology["mnu"],
        num_massive_neutrinos=cosmology["num_massive_neutrinos"], nnu=cosmology["nnu"],
    )
    pars.InitPower.set_params(
        As=cosmology["As"], ns=cosmology["ns"], pivot_scalar=cosmology["pivot_scalar"],
    )
    pars.WantTensors = False
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(
        pars, CMB_unit="muK", raw_cl=True,
        spectra=["unlensed_scalar", "lensed_scalar"],
    )
    unlensed, lensed = powers["unlensed_scalar"], powers["lensed_scalar"]

    out = {"ell": np.arange(unlensed.shape[0])}
    for i, spec in enumerate(SPECTRA):
        out[f"unlensed_{spec}"] = unlensed[:, i]
        out[f"lensed_{spec}"] = lensed[:, i]
    out["header"] = dict(
        cosmology=cosmology,
        lmax=lmax,
        camb_version=camb.__version__,
        units="raw C_l in uK^2, indexed from ell=0 (index i = multipole i)",
        column_order=SPECTRA,
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cosmology", required=True, help="cosmology option id (planck2018)")
    p.add_argument("--out", type=Path, required=True, help="output DIRECTORY (created)")
    args = p.parse_args(argv)

    import numpy as np

    spectra = generate(parse_cosmology(args.cosmology))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "theory_spectra.npz",
             **{k: v for k, v in spectra.items() if k != "header"},
             header=np.array(spectra["header"], dtype=object))
    write_index(args.out, {"stage": "theory_spectra", "cosmology": args.cosmology,
                           "lmax": LMAX, "camb_version": spectra["header"]["camb_version"],
                           "file": "theory_spectra.npz"})


if __name__ == "__main__":
    main()
