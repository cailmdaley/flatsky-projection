"""Is Crawford's variant deficit his mask surgery rather than the projection?

The geometry model reproduces his full-mask ratio to a few percent but
over-predicts both surgical variants by 15-25% above l = 4000. Both variants
zero pixels with a hard edge, and his estimator does no mode decoupling. This
script tests that explanation with no simulations at all: it evaluates the
exact coupling each of his six masks applies, and asks whether folding it in
moves the prediction onto the measurement.

The prediction to compare against his numbers is

    ratio = binned[ K_proj0 * P_model ] / binned[ K_proj5 * P_input ]

-- the geometry model pushed through proj0's coupling, over the input pushed
through proj5's. If the deficit is the surgery, this closes the variants and
leaves the full mask where it already is.

Writes a JSON summary next to the cached masks; prints a table.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import crawford as C          # noqa: E402
import crawford_masks as CM   # noqa: E402
import crawford_model as M    # noqa: E402
import geometry as G          # noqa: E402
import mask_coupling as MC    # noqa: E402

NGRID, RES = 5000, 1.0
SHOW = [10, 20, 30, 40, 50]
OUT = CM.CACHE / "mask_coupling.json"


def model_proj0(variant, spec, ell_fine):
    """The geometry model's proj0 spectrum, unbinned, on `ell_fine`."""
    s1, s2, w = M.geometry_of(variant, NGRID, RES)
    ell_th, cl_th = M.theory_input(spec)
    return G.predicted_spectrum(ell_fine, s1, s2, w, ell_th, cl_th,
                                band_limit=M.BAND_LIMIT, n_bins=40, n_psi=64)


def main():
    ell_fine = np.arange(1.0, 6001.0)
    results = {}

    for spec in ("tt", "ee"):
        ell_th, cl_th = M.theory_input(spec)
        f_in = MC.spectrum_interpolator(ell_th, cl_th, band_limit=M.BAND_LIMIT)

        for variant in CM.VARIANTS:
            m5, m0 = CM.cached_arrays(variant, NGRID, RES)
            k5 = MC.kernel(m5.astype(float))
            k0 = MC.kernel(m0.astype(float))

            ell2d = MC.ell_grid(NGRID, RES)
            in2d = f_in(ell2d)

            p0 = model_proj0(variant, spec, ell_fine)
            f_model = MC.spectrum_interpolator(ell_fine, p0,
                                               band_limit=None)
            mod2d = f_model(ell2d)

            num = MC.bin_annuli(MC.couple(mod2d, k0), ell2d, C.N_BINS)
            den = MC.bin_annuli(MC.couple(in2d, k5), ell2d, C.N_BINS)
            raw_num = MC.bin_annuli(mod2d, ell2d, C.N_BINS)
            raw_den = MC.bin_annuli(in2d, ell2d, C.N_BINS)

            coupled = num / den
            uncoupled = raw_num / raw_den
            _, meas = M.measured(M.VARIANT_FILE[variant], spec)

            results[f"{spec}/{variant}"] = dict(
                ksum5=float(k5.sum()), ksum0=float(k0.sum()),
                bias5=list(np.nan_to_num(den / raw_den)),
                bias0=list(np.nan_to_num(num / raw_num)),
                coupled=list(np.nan_to_num(coupled)),
                uncoupled=list(np.nan_to_num(uncoupled)),
                measured=list(meas),
            )

            print(f"\n{spec.upper()} {variant}  (sum K: proj5 {k5.sum():.6f}, "
                  f"proj0 {k0.sum():.6f})")
            print("  l           " + "".join(f"{C.ELL[i]:8.0f}" for i in SHOW))
            print("  measured    " + "".join(f"{meas[i]:8.3f}" for i in SHOW))
            print("  geometry    " + "".join(f"{uncoupled[i]:8.3f}" for i in SHOW))
            print("  +coupling   " + "".join(f"{coupled[i]:8.3f}" for i in SHOW))
            print("  meas/geom   " + "".join(f"{meas[i]/uncoupled[i]:8.3f}" for i in SHOW))
            print("  meas/+coup  " + "".join(f"{meas[i]/coupled[i]:8.3f}" for i in SHOW))
            print("  mask bias0  " + "".join(f"{(num/raw_num)[i]:8.3f}" for i in SHOW))
            print("  mask bias5  " + "".join(f"{(den/raw_den)[i]:8.3f}" for i in SHOW))
            sys.stdout.flush()

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
