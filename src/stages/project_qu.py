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

    Matches the 2021 project_QU.py exactly: hp.alm2map(alms_TEB, nside,
    pol=True) synthesizes T, Q and U together (a polarized transform,
    because a 3-element alm list is passed); only the polarization pair is
    kept.
    """
    import healpy as hp
    from spt3g.maps import HealpixSkyMap, MapPolConv

    alms_teb = hp.read_alm(str(alm_file), hdu=(1, 2, 3))
    q_map, u_map = hp.alm2map(alms_teb, nside, pol=True)[1:]
    return {
        "Q": HealpixSkyMap(q_map, pol_conv=MapPolConv.COSMO),
        "U": HealpixSkyMap(u_map, pol_conv=MapPolConv.COSMO),
    }


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
