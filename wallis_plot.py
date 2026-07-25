# %% 
import spt3g
from spt3g.core import G3File, G3Units
from spt3g import maps
from spt3g.mapspectra.map_analysis import constructEB

import sys; sys.path.append("/src/spt3g_software/scratch/cailmdaley")
from spt3g_plotting import plot_frames

# %% 
input = ["/sptlocal/user/cailmdaley/pol_project/paper/projected_maps/seed1_proj5_QU.g3", "/sptlocal/user/cailmdaley/pol_project/paper/projected_maps/seed1_proj5_EB.g3"]

frame_QU = next(G3File(input[0]))
frame_EB_healpix = dict(next(G3File(input[1])))
frame_EB_flatsky = {k: f.get_rmap() for (k,f) in constructEB(frame_QU).items()}
frame_EB_diff = {k : frame_EB_healpix[k] - frame_EB_flatsky[k] for k in "EB"}
# %%

frame_EB_healpix['B'].plot()
# (frame_EB_healpix['E'] - frame_EB_flatsky['E']).plot() 

# %%

# %%

# %%
py"""
from spt3g import core
from spt3g.core import G3Units, G3Writer, G3Frame
from spt3g.maps import MapProjection, FlattenPol, reproj_map
from spt3g.maps.fitsio import load_skymap_fits
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing.sourcemaps import CreateFieldMapStub
from spt3g.util.files import load_pickle
import healpy as hp
"""

py"""
healpix_frame = {k: hp.ud_grade(np.asarray(f), 256) 
    for (k, f) in load_skymap_fits("/sptgrid/simulation/lensing_maps/unlensed_TQU1/TQU1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed1.fits", keys="TQU").items()}
"""

py"""
healpix_frame_TEB = dict(zip("TEB", hp.alm2map(
    hp.map2alm([np.asarray(f) for f in healpix_frame.values()], pol=True), 
    256, pol=False)))
"""

py"healpix_frame_TEB"

plot(Map{Float64, RingOrder}(py"healpix_frame['Q']"))

plot(Map{Float64, RingOrder}(py"healpix_frame_TEB['E']"))

# %%
