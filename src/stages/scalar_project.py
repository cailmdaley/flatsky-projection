"""Project the first alm component as a scalar temperature map."""
from _project import Leg, run_leg


def synthesize(path, nside):
    import healpy as hp
    from spt3g.maps import HealpixSkyMap, MapPolConv
    alm = hp.read_alm(str(path), hdu=1)
    sky = hp.alm2map(alm, nside)
    return {"T": HealpixSkyMap(sky, pol_conv=MapPolConv.COSMO)}


if __name__ == "__main__":
    run_leg(Leg(name="scalar", keys=("T",)), synthesize, lambda frame: None)
