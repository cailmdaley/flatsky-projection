"""Project Q/U onto the flat sky -- the leaky leg.

Synthesizes Q and U on an intermediate Healpix grid from the same per-seed
T/E/B alms the EB leg uses, then resamples them into each of the five flat
sky projections and flattens the polarization basis. This is what a real
flat-sky polarization analysis does, and because Q/U is a spin-2 field, it is
the leg that carries the projection-induced E-to-B leakage this project
measures. Compare against project_eb.py, which projects the same sky as two
independent spin-0 scalars and cannot leak by construction.

Ported from
/sptlocal/analysis/pol_project2020/workflow/scripts/project_QU.py.
"""

from __future__ import annotations

from _project import Leg, run_leg

LEG = Leg(name="qu", keys=("Q", "U"))


def _synthesize(alm_file, nside):
    """Q, U on the intermediate Healpix grid.

    The 2021 project_QU.py called hp.alm2map(alms_TEB, nside, pol=True) and
    kept the polarization pair. hp.alm2map_spin([alm_E, alm_B], nside, 2,
    lmax) is the same spin-2 synthesis -- bit-identical output, verified
    against the pol=True call on this project's alms -- without the T map
    and without returning the three maps as one array. That matters for
    memory, not correctness: at nside 8192 each map is 6.4 GB, and the
    pol=True form held T+Q+U while two HealpixSkyMap copies were made
    (38.8 GB peak per unit, measured). Here each numpy map is copied into
    its HealpixSkyMap and released before the next, so the peak is the two
    synthesized maps plus one copy.
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


def _postprocess(flatsky_frame):
    from spt3g import maps

    # QU leg: this is the leaky leg. Projecting Q/U as a spin-2 pair rotates
    # the polarization basis by a position-dependent angle; FlattenPol undoes
    # that *rotation* so grid Q/U line back up with sky Q/U, but it cannot
    # undo the projection's local *shear* -- a sheared spin-2 field is not a
    # rotated spin-2 field (see analysis/geometry.py). The E-to-B leakage
    # that survives at O(shear - 1) is exactly what this project measures,
    # and it only shows up on this leg because only this leg ever treats
    # Q/U as a spin-2 quantity.
    #
    # The 2021 script called maps.FlattenPol(frame) here. That no longer works:
    # today's FlattenPol does `frame.pop("T")` unconditionally, and this frame
    # carries only Q and U, so it raises KeyError: 'T'. FlattenPol is a thin
    # frame-level wrapper whose only real work is the call below, so calling
    # flatten_pol directly is the same rotation with no T required. (The 2021
    # maps on disk do carry flat_pol=True, so the build of the day tolerated a
    # T-less frame; the wrapper has tightened since.)
    maps.flatten_pol(flatsky_frame["Q"], flatsky_frame["U"])


def main(argv=None):
    return run_leg(LEG, _synthesize, _postprocess, argv=argv)


if __name__ == "__main__":
    main()
