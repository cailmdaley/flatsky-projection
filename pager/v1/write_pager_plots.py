import numpy as np
from scipy.ndimage.interpolation import zoom
from spt3g.core import G3File, G3Units
from spt3g.maps import FlatSkyMap, reproj_map, MapProjection

from spt3g.mapspectra.map_analysis import constructEB
from spt3g.util.files import load_pickle
from spt3g.std_processing.sourcemaps import CreateFieldMapStub
from tqdm import tqdm
import sys
import pathlib

from spt3g import core
core.set_log_level(core.G3LogLevel.LOG_ERROR)


import sys; sys.path.append("/src/spt3g_software/scratch/cailmdaley")
from spt3g_plotting import *
import matplotlib.pyplot as plt

spt_mask_zea = load_pickle("/spt/user/ddutcher/masks/3band_res2_bundle_apod_mask.pkl")
spt_mask_proj5 = CreateFieldMapStub(proj=MapProjection.ProjZEA)
np.asarray(spt_mask_proj5)[:] = spt_mask_zea

spt_mask_proj9 = CreateFieldMapStub(proj=MapProjection.ProjBICEP)
reproj_map(spt_mask_proj5, spt_mask_proj9)
spt_mask_bicep = np.asarray(spt_mask_proj9)

def plot_generic_inset(frame, vlims):
    k1, k2 = list(frame.keys())
    fac = 2*G3Units.arcmin / frame[k1].res
    scale = lambda ind: int(ind*fac)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,8), tight_layout=True)
    plot_FlatSkyMap(frame[k1][scale(200):scale(300), scale(625):scale(725)], vlim=vlims[0],  ax=ax1, title_prefix="Upper Left {} ".format(k1))
    plot_FlatSkyMap(frame[k1][scale(750):scale(850), -scale(200):-scale(100)], vlim=vlims[0],  ax=ax3, title_prefix="Lower Right {} ".format(k1))
    plot_FlatSkyMap(frame[k2][scale(200):scale(300), scale(625):scale(725)], vlim=vlims[1],  ax=ax2, title_prefix="Upper Left {} ".format(k2))
    plot_FlatSkyMap(frame[k2][scale(750):scale(850), -scale(200):-scale(100)], vlim=vlims[1],  ax=ax4, title_prefix="Lower Right {} ".format(k2))
    return fig
    

def plot_QU(writestring, qu_frame):
    fig = plot_generic_inset(qu_frame, ('sym', 'sym'))
    fig.savefig(writestring + "_figtype=QUinset_vlim=99th.png"); plt.close(fig)
    
    fig = plot_generic_inset(qu_frame, ((-15,15), (-15,15)))
    fig.savefig(writestring + "_figtype=QUinset_vlim=fixed.png"); plt.close(fig)


def plot_EB(writestring, qu_frame):
    spt_mask = spt_mask_zea if 'ZEA' in writestring else spt_mask_bicep
    mask = zoom(spt_mask, 2*G3Units.arcmin / qu_frame["Q"].res)
    B_fourier = constructEB(qu_frame, apod_mask=mask)["B"]

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_MapSpectrum2D(B_fourier, title_prefix="B ", ax=ax, vlim=(1e-8, 2e-2))
    fig.savefig(writestring + "_figtype=Bfourierspt_vlim=fixed.png"); plt.close(fig)

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_MapSpectrum2D(B_fourier, title_prefix="B ", ax=ax, vlim='asym')
    fig.savefig(writestring + "_figtype=Bfourierspt_vlim=99th.png"); plt.close(fig)
    del B_fourier

    eb_frame = constructEB(qu_frame)
    B_map = eb_frame["B"].get_rmap()
    np.asarray(B_map)[:] *= mask

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_FlatSkyMap(B_map, vlim=(-1,1),  ax=ax, title_prefix="B ")
    fig.savefig(writestring + "_figtype=Bmapspt_vlim=fixed.png"); plt.close(fig)

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_FlatSkyMap(B_map, vlim='sym', ax=ax, title_prefix="B ")
    fig.savefig(writestring + "_figtype=Bmapspt_vlim=99th.png"); plt.close(fig)
    del B_map
    
    fig = plot_generic_inset({k : eb_frame[k].get_rmap() for k in eb_frame}, ('sym', 'sym'))
    fig.savefig(writestring + "_figtype=EBinset_vlim=99th.png"); plt.close(fig)

    
    fig = plot_generic_inset({k : eb_frame[k].get_rmap() for k in eb_frame}, ((-15,15), (-1,1)))
    fig.savefig(writestring + "_figtype=EBinset_vlim=fixed.png"); plt.close(fig)
    

def write_plots(writepath, dirs):
    writepath.mkdir(exist_ok=True)
    for readdir in tqdm(dirs):
        writestring = str(writepath / readdir.name)
        frame_path = readdir / "seed1.g3" 
        
        if frame_path.exists():
            qu_frame = next(G3File(str(frame_path)))
        else:
            continue

        try:
          plot_EB(writestring, qu_frame)
          plot_QU(writestring, qu_frame)
        except:
          print(frame_path)
        del qu_frame


writepath = pathlib.Path(sys.argv[1])
globstring = sys.argv[2]

dirs = list(pathlib.Path("/sptlocal/user/cailmdaley/pol_project/pager/v1/pager_maps/").glob(globstring))
print("Making plots for the {} directories that match glob(\"{}\")".format(len(dirs), globstring))
write_plots(writepath, dirs)