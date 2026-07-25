import itertools
from pathlib import Path
import numpy as np

from tqdm import tqdm
from spt3g import core
from spt3g.core import G3Units, G3Writer, G3Frame
from spt3g.maps import MapProjection, flatten_pol
from spt3g.maps.fitsio import load_skymap_fits
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing.sourcemaps import CreateFieldMapStub

core.set_log_level(core.G3LogLevel.LOG_ERROR)


basepath = Path("/sptlocal/user/cailmdaley/pager_maps")
nsides = [8192, 8192 // 2, 8192 // 4, 8192 // 8]
interps = [True, False]
rebins = [1, 2, 4, 8]
projs = [MapProjection.ProjZEA, MapProjection.ProjBICEP]
resos = [0.5, 1, 2, 4, 6]
decs = [0, -57.5, 90.0]
rotates = [None, 'flat']
project_options = [interps, rebins, projs, resos, decs]

path = "/spt/simulation/lensing_maps/unlensed_TQU1/TQU1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{}.fits"
def read_healpix(nside, seed):
    return {k : load_skymap_fits(path.format(seed), 
                                 memmap=True, 
                                 keys=k)[k].rebin(8192//nside)
            for k in "QU"}
        
def project(healpix_frame, interp, rebin, proj, res_arcmin, dec_deg):
    parent = CreateFieldMapStub(res=res_arcmin*G3Units.arcmin,  proj=proj, dec_center=dec_deg*G3Units.deg)
    return {k : healpix_to_flatsky(f, map_stub=parent.Clone(False), interp=interp, rebin=rebin) 
            for (k, f) in healpix_frame.items()};
    
def write_flatsky_frame(flatsky_name, filename):
    frame = G3Frame()
    for k in flatsky_frame.keys():
        frame[k] = flatsky_frame[k]
    
    G3Writer(filename=filename)(frame)
    


# for seed, nside in itertools.product(tqdm(seeds), tqdm(nsides), total=40):
# seeds = range(1, 11)
seed = 1
for nside in tqdm(nsides):
    healpix_frame = read_healpix(nside, seed)
    for interp, rebin, proj, res, dec in tqdm(itertools.product(*project_options), total=1920):
      
        dirbase = "nside={}_interp={}_rebin={}_proj={}_res={}_dec={}".format(
            nside, interp, rebin, proj, res, dec)
        d1 = basepath / (dirbase + "_rotate=None"); d1.mkdir(exist_ok=True)
        d2 = basepath / (dirbase + "_rotate=flat"); d2.mkdir(exist_ok=True)
        f1, f2 = d1 / "seed{}.g3".format(seed), d2 / "seed{}.g3".format(seed)
        
        if not (f1.exists() and f2.exists()):
            print("Writing {}\n".format(dirbase))
            
            if proj == MapProjection.ProjBICEP and dec == 90.0:
                dec = 89.99
                
            flatsky_frame = project(healpix_frame, interp, rebin, proj, res, dec)
            write_flatsky_frame(flatsky_frame, str(f1))
            
            flatten_pol(*flatsky_frame.values())
            write_flatsky_frame(flatsky_frame, str(f2))