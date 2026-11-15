"""Project the first alm component of a scalar sky as a temperature map.

`--spectrum tt` projects the CMB temperature of the sky_alms output;
`--spectrum white` projects the unit-C_l white skies. The spectrum name
prefixes every map and index file, so both share one output directory.
"""
import argparse
import sys

from _project import Leg, run_leg


def synthesize(path, nside):
    import healpy as hp
    from spt3g.maps import HealpixSkyMap, MapPolConv
    alm = hp.read_alm(str(path), hdu=1)
    sky = hp.alm2map(alm, nside)
    return {"T": HealpixSkyMap(sky, pol_conv=MapPolConv.COSMO)}


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--spectrum", choices=["tt", "white"], required=True)
    p.add_argument("--tt-alms", required=True, help="sky_alms output directory")
    p.add_argument("--white-alms", required=True, help="scalar_white_alms output directory")
    args, rest = p.parse_known_args(argv if argv is not None else sys.argv[1:])
    alms = args.tt_alms if args.spectrum == "tt" else args.white_alms
    return run_leg(Leg(name=args.spectrum, keys=("T",)), synthesize, lambda frame: None,
                   argv=["--alms", alms, *rest], index_tag=args.spectrum)


if __name__ == "__main__":
    main()
