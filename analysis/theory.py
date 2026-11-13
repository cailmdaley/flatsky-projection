"""CMB power spectra in the input simulations' cosmology, from CAMB.

Generates the analysis input `theory_spectra` (data/theory_spectra.npz): lensed
and unlensed TT, EE, BB and TE as raw C_l in uK^2, indexed by multipole from
ell = 0 to 10050. Every stage and analysis script that needs a theory curve
reads this one file and slices what it uses.

COSMOLOGY PROVENANCE. The unlensed input sims
(/sptgrid/simulation/lensing_maps/unlensed_TQU1/TQU1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed*.fits)
carry their cosmology in the filename: the Planck 2018
base_plikHM_TTTEEE_lowl_lowE_lensing chain, best-fit (not posterior-mean)
parameters. The 2021 pipeline's own config (/sptlocal/analysis/pol_project2020)
has no camb/cosmology block of its own -- it consumes those sims as a given.
The exact numbers below are copied from
planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini, a CAMB Fortran
parameter file that ships inside spt3g_software's simulations/ tree (missing
from this checkout, but present byte-identically in three independent
colleagues' checkouts: rgualtie, wlwu, tcrawfor -- md5sums of the .dat outputs
agree, so this is the shared reference, not a personal regeneration). Running
CAMB 1.5.4 at *default* accuracy with these numbers reproduces that Fortran
file's unlensed EE to 0.03% over 100 < ell < 3000, even though the Fortran run
used accuracy_boost=4 / l_accuracy_boost=4 -- default accuracy is already
enough at the precision this project needs.

TRAPS.
  - raw_cl vs D_l: CAMB's natural output is D_l = l(l+1)C_l/2pi in uK^2. This
    module stores raw C_l (divide that out) because that is what synfast-style
    consumers and NaMaster comparisons want; do not re-introduce D_l here,
    convert it at plot time instead.
  - ell=0,1 are physically undefined for polarization (and l=0,1 for TT are
    the monopole/dipole, conventionally excluded). CAMB's raw_cl output
    already zeros indices 0 and 1; we keep that convention rather than NaN,
    so every array here is safe to index directly by ell with no NaN traps.
  - lmax=10000 reaches well past the simulations' band limit of 6000, so one
    file serves every consumer, including the lensed BB curve around ell~1000.
  - NonLinear_lens (halofit applied to the lensing potential only, not the
    matter power spectrum) is CAMB's default once lens_potential_accuracy > 0
    and is what the reference Fortran run effectively used too; do not turn
    it off, the BB floor amplitude depends on it at the several-percent level.
"""

import argparse
from pathlib import Path

import numpy as np

# Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing best-fit parameters, as
# used to draw /sptgrid/simulation/lensing_maps/unlensed_TQU1 (see module
# docstring for the exact provenance chain). These are the *best-fit* chi^2
# point of that chain, not the posterior mean -- close but not identical
# (e.g. mean ombh2 = 0.02237 vs best-fit 0.0223828).
COSMOLOGY = dict(
    H0=67.32117,
    ombh2=0.0223828,
    omch2=0.1201075,
    omk=0.0,
    tau=0.05430842,
    mnu=0.06,                    # single massive neutrino, matches omnuh2=6.451439e-4
    num_massive_neutrinos=1,
    nnu=3.046,
    As=2.100549e-9,
    ns=0.9660499,
    pivot_scalar=0.05,
)

LMAX = 10000

SPECTRA = ["TT", "EE", "BB", "TE"]  # CAMB's raw_cl column order


def generate(lmax=LMAX, cosmology=None):
    """Run CAMB and return a dict of raw C_l [uK^2] arrays indexed from ell=0.

    Returns both lensed and unlensed TT/EE/BB/TE, plus the ell array and a
    metadata dict recording exactly what produced them.
    """
    import camb

    cosmology = dict(COSMOLOGY if cosmology is None else cosmology)

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
    # lens_potential_accuracy=2: the lensed BB floor is the entire point of
    # this run, so pay for a converged lensing potential rather than the
    # lens_potential_accuracy=1 default.
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(
        pars, CMB_unit="muK", raw_cl=True,
        spectra=["unlensed_scalar", "lensed_scalar"],
    )
    # 'lensed_scalar' is CAMB's key for lensed C_l even with WantTensors=False
    # (it's identical to 'total' in that case); 'unlensed_scalar' is unlensed.
    unlensed = powers["unlensed_scalar"]  # (lmax+1, 4): TT EE BB TE
    lensed = powers["lensed_scalar"]

    ell = np.arange(unlensed.shape[0])

    out = {"ell": ell}
    for i, spec in enumerate(SPECTRA):
        out[f"unlensed_{spec}"] = unlensed[:, i]
        out[f"lensed_{spec}"] = lensed[:, i]

    out["header"] = dict(
        cosmology=cosmology,
        lmax=lmax,
        camb_version=camb.__version__,
        units="raw C_l in uK^2, indexed from ell=0 (index i = multipole i)",
        column_order=SPECTRA,
        provenance=(
            "Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing best fit, "
            "recovered from the filename of "
            "/sptgrid/simulation/lensing_maps/unlensed_TQU1 and cross-checked "
            "against planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini "
            "in spt3g_software/simulations (agreement 0.03% in unlensed EE, "
            "100<ell<3000). See analysis/theory.py module docstring."
        ),
    )
    return out


def save(path, spectra=None):
    """Write the npz. Overwrites cleanly -- safe to rerun after any change."""
    spectra = generate() if spectra is None else spectra
    # np.savez can't store a dict directly; wrap it as a 0-d object array.
    # np.load(..., allow_pickle=True)['header'][()] unpacks it back to a dict.
    np.savez(path, **{k: v for k, v in spectra.items() if k != "header"},
              header=np.array(spectra["header"], dtype=object))
    print(f"wrote {path}")
    return path


def load(path):
    """Load the saved spectra back into a dict, header included as a real dict."""
    with np.load(path, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files if k != "header"}
        out["header"] = f["header"][()]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="the .npz to write")
    save(ap.parse_args().out)
