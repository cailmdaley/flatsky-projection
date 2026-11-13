"""Project E/B onto the flat sky -- the leakage-free reference leg.

Synthesizes E and B on an intermediate Healpix grid *as independent spin-0
scalars* from the same per-seed alms the QU leg uses, then resamples each
into the four flat sky projections with no further processing. A spin-0
field cannot mix into another spin-0 field under reprojection, so this leg
has no projection-induced E-to-B leakage by construction. It is the
reference project_qu.py's (leaky) EE/BB is differenced against.

Ported from
/sptlocal/analysis/pol_project2020/workflow/scripts/project_EB_fullsky.py.
"""

from __future__ import annotations

from _project import Leg, run_leg

LEG = Leg(name="eb", keys=("E", "B"))


def _synthesize(alm_file, nside):
    """E, B on the intermediate Healpix grid, each as an independent spin-0 scalar.

    Matches the 2021 project_EB_fullsky.py exactly: hp.alm2map(alm, nside)
    is called once per component with a *single* alm array (not the
    3-element T/E/B list project_qu.py passes), which makes healpy perform a
    plain spin-0 synthesis rather than a polarized (spin-2) one.
    """
    import healpy as hp
    from spt3g.maps import HealpixSkyMap, MapPolConv

    alm_e, alm_b = hp.read_alm(str(alm_file), hdu=(2, 3))
    # One component at a time, releasing each numpy map once its
    # HealpixSkyMap copy exists: at nside 8192 a map is 6.4 GB, and holding
    # both numpy maps through both copies doubles the unit's peak memory.
    maps = {}
    e_map = hp.alm2map(alm_e, nside)
    del alm_e
    maps["E"] = HealpixSkyMap(e_map, pol_conv=MapPolConv.COSMO)
    del e_map
    b_map = hp.alm2map(alm_b, nside)
    del alm_b
    maps["B"] = HealpixSkyMap(b_map, pol_conv=MapPolConv.COSMO)
    del b_map
    return maps


def _postprocess(flatsky_frame):
    # EB leg: E and B were synthesized on the sphere as independent spin-0
    # scalars (see _synthesize above) and projected the same way -- no
    # polarization basis is ever constructed for this leg, so there is
    # nothing for FlattenPol to undo, and it is deliberately not called
    # here. That is exactly what makes this leg leakage-free by
    # construction: reprojecting a spin-0 field cannot mix E and B, no
    # matter how sheared the projection is. Compare project_qu.py, whose
    # spin-2 Q/U leg does call FlattenPol and does leak.
    pass


def main(argv=None):
    return run_leg(LEG, _synthesize, _postprocess, argv=argv)


if __name__ == "__main__":
    main()
