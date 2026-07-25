import numpy as np
from scipy.ndimage.interpolation import zoom
from spt3g.core import G3File, G3Units
from spt3g.maps import FlatSkyMap, reproj_map, MapProjection, get_ra_dec_map

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


def plot_generic_inset(frame, mask, vlims):
    k1, k2  = list(frame.keys())
    
    ra, dec = [np.asarray(coord) for coord in get_ra_dec_map(mask)]

    mask1_inds = np.logical_and.reduce(
        [((50 - 3.3)*G3Units.deg < ra),  (ra  < 50*G3Units.deg),
         ((-47.5 - 2)*G3Units.deg < dec), (dec < -47.5*G3Units.deg)]
    )
    mask2_inds = np.logical_and.reduce(
        [(-50*G3Units.deg < ra), (ra < (-50 + 5.9)*G3Units.deg),
         (-70*G3Units.deg         < dec), (dec < (-70 + 2.05)*G3Units.deg)]
    )
    
    (ymin1, ymax1), (xmin1, xmax1) = [[int(f(np.where(mask1_inds)[i])) for f in (np.min, np.max)] for i in (0, 1)]
    (ymin2, ymax2), (xmin2, xmax2) = [[int(f(np.where(mask2_inds)[i])) for f in (np.min, np.max)] for i in (0, 1)]
    
    m1, m2 = [frame[k].Clone(False) for k in frame]
    np.asarray(m1)[mask1_inds] = np.asarray(frame[k1])[mask1_inds]
    np.asarray(m1)[mask2_inds] = np.asarray(frame[k1])[mask2_inds]
    np.asarray(m2)[mask1_inds] = np.asarray(frame[k2])[mask1_inds]
    np.asarray(m2)[mask2_inds] = np.asarray(frame[k2])[mask2_inds]
    
    pad=1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,8), tight_layout=True)
    
    m = m1[ymin1-pad:ymax1+pad+1, xmin1-pad:xmax1+pad+1]
    plot_FlatSkyMap(m, vlim=vlims[0],  ax=ax1, title_prefix="Upper Left {} ".format(k1))
    
    m = m1[ymin2-pad:ymax2+pad+1, xmin2-pad:xmax2+pad+1]
    plot_FlatSkyMap(m, vlim=vlims[0],  ax=ax2, title_prefix="Lower Right {} ".format(k1))
    
    m = m2[ymin1-pad:ymax1+pad+1, xmin1-pad:xmax1+pad+1]
    plot_FlatSkyMap(m, vlim=vlims[1],  ax=ax3, title_prefix="Upper Left {} ".format(k2))
    
    m = m2[ymin2-pad:ymax2+pad+1, xmin2-pad:xmax2+pad+1]
    plot_FlatSkyMap(m, vlim=vlims[1],  ax=ax4, title_prefix="Lower Right {} ".format(k2))
    
    return fig


def plot_QU(writestring, qu_frame, mask):
    fig = plot_generic_inset(qu_frame, mask, ('sym', 'sym'))
    fig.savefig(writestring + "_figtype=QUinset_vlim=99th.png"); plt.close(fig)
    
    fig = plot_generic_inset(qu_frame, mask, ((-15,15), (-15,15)))
    fig.savefig(writestring + "_figtype=QUinset_vlim=fixed.png"); plt.close(fig)


def plot_EB(writestring, qu_frame, mask):
    EB_fourier = constructEB(qu_frame, apod_mask=mask)

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_MapSpectrum2D(EB_fourier["B"], title_prefix="B ", ax=ax, vlim=(1e-8, 2e-2))
    fig.savefig(writestring + "_figtype=Bfourierspt_vlim=fixed.png"); plt.close(fig)

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_MapSpectrum2D(EB_fourier["B"], title_prefix="B ", ax=ax, vlim='asym')
    fig.savefig(writestring + "_figtype=Bfourierspt_vlim=99th.png"); plt.close(fig)
    
    
    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_MapSpectrum2D(EB_fourier["E"], title_prefix="E ", ax=ax, vlim=(1e-7, 5e-2))
    fig.savefig(writestring + "_figtype=Efourierspt_vlim=fixed.png"); plt.close(fig)

    fig, ax = plt.subplots(1, figsize=(14,8), tight_layout=True)
    plot_MapSpectrum2D(EB_fourier["E"], title_prefix="E ", ax=ax, vlim='asym')
    fig.savefig(writestring + "_figtype=Efourierspt_vlim=99th.png"); plt.close(fig)
    
    
    del EB_fourier

    
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
    
    fig = plot_generic_inset({k : eb_frame[k].get_rmap() for k in eb_frame}, mask, ('sym', 'sym'))
    fig.savefig(writestring + "_figtype=EBinset_vlim=99th.png"); plt.close(fig)

    
    fig = plot_generic_inset({k : eb_frame[k].get_rmap() for k in eb_frame}, mask, ((-15,15), (-1,1)))
    fig.savefig(writestring + "_figtype=EBinset_vlim=fixed.png"); plt.close(fig) 

def write_plots(writepath, dirs):
    writepath.mkdir(exist_ok=True)
    for readdir in tqdm(dirs):
        writestring = str(writepath / readdir.name)
        
        qu_frame = next(G3File(str(readdir / "seed1.g3" )))
        for k in qu_frame:
            np.asarray(qu_frame[k])[:] = np.nan_to_num(qu_frame[k])
        mask_frame = next(G3File(str(readdir / "mask.g3" )))

        plot_EB(writestring, qu_frame, mask_frame["T"])
        plot_QU(writestring, qu_frame, mask_frame["T"])
        
        del qu_frame, mask_frame



basedir = pathlib.Path("/sptlocal/user/cailmdaley/pol_project/pager/v2/pager_maps")
if __name__ == "__main__":
    writepath = pathlib.Path(sys.argv[1])
    globstring = sys.argv[2]
    
    dirs = list(basedir.glob(globstring))
    print("Making plots for the {} directories that match glob(\"{}\")".format(len(dirs), globstring))
    write_plots(writepath, dirs)