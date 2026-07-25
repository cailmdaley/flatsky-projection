from pathlib import Path
import numpy as np
import itertools
from tqdm import tqdm

from spt3g import core
from spt3g.core import G3Units, G3Writer, G3Frame
from spt3g.maps import MapProjection, flatten_pol, reproj_map
from spt3g.maps.fitsio import load_skymap_fits
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing.sourcemaps import CreateFieldMapStub
from spt3g.util.files import load_pickle
#core.set_log_level(core.G3LogLevel.LOG_ERROR)

import healpy as hp


basepath = Path("/sptlocal/user/cailmdaley/pol_project/pager/v2/pager_maps")
basepath.mkdir(exist_ok=True)

nside = 8192
interp = True
rebin = 1
res = 2
dec = -57.5
width = 110
seed = 1

lmaxs = [None, 4200]
projs = [0, 1, 2, 3, 4, 5, 6, 7, 9] 

def read_healpix(nside, seed):
    healpix_path = "/spt/simulation/lensing_maps/unlensed_TQU1/TQU1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{}.fits"
    return load_skymap_fits(healpix_path.format(seed), memmap=True,  keys="TQU")
        
def project(healpix_frame, interp, rebin, proj, res_arcmin, dec_deg, width):
    parent = CreateFieldMapStub(res=res_arcmin*G3Units.arcmin,  proj=proj, 
                                dec_center=dec_deg*G3Units.deg, width=width*G3Units.deg)
    return {k : healpix_to_flatsky(f, map_stub=parent.Clone(False), interp=interp, rebin=rebin) 
            for (k, f) in healpix_frame.items()};
    
def write_flatsky_frame(flatsky_frame, filename):
    frame = G3Frame()
    for k in flatsky_frame.keys():
        frame[k] = flatsky_frame[k]
    G3Writer(filename=str(filename))(frame)
    
mask_proj5 = CreateFieldMapStub(proj=MapProjection.ProjZEA)
np.asarray(mask_proj5)[:] = load_pickle("/spt/user/ddutcher/masks/3band_res2_bundle_apod_mask.pkl")

for lmax in tqdm(lmaxs):
    healpix_frame = read_healpix(nside, seed)
    print("made it!")
    if lmax is not None:
        aℓms = hp.map2alm([np.asarray(m) for m in healpix_frame.values()],lmax=lmax,pol=True)
        for k, m in zip("TQU", hp.alm2map(aℓms, 8192, pol=True)):
            np.asarray(healpix_frame[k])[:] = m
    del healpix_frame["T"]
    
    for proj in tqdm(projs):
        directory = basepath / "lmax={}_proj={}".format(lmax, proj)
        directory.mkdir(exist_ok=True)

        flatsky_frame = project(healpix_frame, interp, rebin, 
                                MapProjection.values[proj], res, dec, width)
        flatten_pol(*flatsky_frame.values())
        write_flatsky_frame(flatsky_frame, directory / "seed{}.g3".format(seed))


        mask = CreateFieldMapStub(proj=MapProjection.values[proj], 
                                  res=res*G3Units.arcmin, 
                                  width=width*G3Units.deg)
        reproj_map(mask_proj5, mask)
        write_flatsky_frame({"T": mask}, directory / "mask.g3")
