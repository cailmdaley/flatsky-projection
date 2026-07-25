import os; os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)

from pathlib import Path
import numpy as np
from itertools import product
from tqdm import tqdm

from spt3g import core
from spt3g.core import G3Units, G3Writer, G3Frame
from spt3g import maps
from spt3g.maps.fitsio import load_skymap_fits
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing.sourcemaps import CreateFieldMapStub
from spt3g.util.files import load_pickle
import healpy as hp


def project(healpix_frame, interp, rebin, proj, res_arcmin, dec_deg, size_deg):
    size_pix = int(size_deg * G3Units.deg / (res_arcmin * G3Units.arcmin))
    parent = maps.FlatSkyMap(size_pix, size_pix, res_arcmin*G3Units.arcmin,
                             proj=proj, delta_center=dec_deg*G3Units.deg)
    flatsky_frame = G3Frame()
    for (k, f) in healpix_frame.items():
        flatsky_frame[k] = healpix_to_flatsky(f, map_stub=parent.clone(False), 
                                              interp=interp, rebin=rebin)
    return flatsky_frame

nside = 512
projs = snakemake.config['projs']


print("reading Healpix...")
healpix_frame = load_skymap_fits(snakemake.input[0], keys="TQU") 
pol_conv = healpix_frame['T'].pol_conv

print("rotating Healpix QU to EB...")
healpix_list = [np.asarray(f) for f in healpix_frame.values()]
alms_list = hp.map2alm(healpix_list, pol=True, lmax=lmax)

for queb in snakemake.output.keys():
    maps_list = hp.alm2map(alms_list, nside=nside, lmax=lmax, pol = True if queb == "QU" else False)
    healpix_frame = {k : maps.HealpixSkyMap(f, pol_conv = pol_conv)
                     for (k, f) in zip(queb, maps_list[1:])}
    
    for i in range(len(projs)):
        proj = "ProjBICEP" if projs[i] == "ProjECP" else projs[i]
        flatsky_frame = project(healpix_frame, proj=maps.MapProjection.names[proj], 
                                **snakemake.config['project'])
        if queb == "QU":
            maps.FlattenPol(flatsky_frame)
        G3Writer(filename=snakemake.output[queb][i])(flatsky_frame)