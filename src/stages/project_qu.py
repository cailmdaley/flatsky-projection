"""Project Q/U onto the flat sky -- the leaky leg.

Synthesizes Q and U on an intermediate Healpix grid from the per-seed T/E/B
alms the EB leg also uses, resamples them into each flat-sky projection, and
flattens the polarization basis (`_project.flatten`, by the `--flattening`
decision). This is what a real flat-sky polarization analysis does, and
because Q/U is a spin-2 field, it is the leg that carries the
projection-induced E-to-B leakage this project measures. Compare
project_eb.py, which projects the same sky as two independent spin-0 scalars
and cannot leak by construction.
"""

from __future__ import annotations

import argparse
import sys

from _decisions import parse_flattening
from _project import Leg, flatten, run_leg

LEG = Leg(name="qu", keys=("Q", "U"))


def _synthesize(alm_file, nside):
    """Q, U on the intermediate Healpix grid.

    hp.alm2map_spin([alm_E, alm_B], nside, 2, lmax) is the spin-2 synthesis
    of hp.alm2map(alms_TEB, pol=True) without the T map. Each numpy map is
    copied into its HealpixSkyMap and released before the next, so the peak
    is the two synthesized maps plus one copy (6.4 GB per map at nside 8192).
    """
    import healpy as hp
    from spt3g.maps import HealpixSkyMap, MapPolConv

    alm_e, alm_b = hp.read_alm(str(alm_file), hdu=(2, 3))
    lmax = hp.Alm.getlmax(alm_e.size)
    q_map, u_map = hp.alm2map_spin([alm_e, alm_b], nside, 2, lmax)
    del alm_e, alm_b
    maps = {"Q": HealpixSkyMap(q_map, pol_conv=MapPolConv.COSMO)}
    del q_map
    maps["U"] = HealpixSkyMap(u_map, pol_conv=MapPolConv.COSMO)
    del u_map
    return maps


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--flattening", type=parse_flattening, required=True,
                   help="polar | spt3g")
    args, rest = p.parse_known_args(argv if argv is not None else sys.argv[1:])

    # Flattening removes the rotation the projection applies to the Q/U
    # basis. It cannot remove the projection's local shear -- a sheared
    # spin-2 field is not a rotated one -- and the E-to-B leakage the shear
    # leaves is what this project measures.
    def postprocess(frame):
        flatten(frame["Q"], frame["U"], args.flattening)

    return run_leg(LEG, _synthesize, postprocess, argv=rest,
                   record={"flattening": args.flattening})


if __name__ == "__main__":
    main()
