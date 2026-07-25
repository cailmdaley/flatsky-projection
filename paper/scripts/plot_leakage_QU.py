# %%
import numpy as np
from spt3g.core import G3File, G3Units
from spt3g.maps import get_ra_dec_map, MapProjection, reproj_map, MapPolType
from spt3g.mapspectra.map_analysis import constructEB
from spt3g.mapspectra.basicmaputils import map_to_ft, get_queb_trans_angle
from spt3g.std_processing.sourcemaps import CreateFieldMapStub
from spt3g.util.files import load_pickle

import sys; sys.path.append("/src/spt3g_software/scratch/cailmdaley")
from spt3g_plotting import plot_frames
import matplotlib.pyplot as plt

frame_QU = next(G3File(snakemake.input.projected_QU))

# %%
mask_proj5 = CreateFieldMapStub(proj=MapProjection.ProjZEA,
                                pol_type = MapPolType.none)
np.asarray(mask_proj5)[:] = load_pickle(snakemake.input.mask)

# %% 
mask = frame_QU['Q'].clone(False)
mask.pol_type = MapPolType.none
reproj_map(mask_proj5, mask)

# %%
frame_EB = constructEB(frame_QU, apod_mask=mask)
ϕ = get_queb_trans_angle(frame_EB['B'])

# %%


B_in_QU = {"Q component of B" : -np.sin(2 * ϕ) * frame_EB['B'], 
           "U component of B" :  np.cos(2 * ϕ) * frame_EB['B']}

plot_frames(B_in_QU, vlim=(1e-9,1e-2), suptitle="{}".format(mask.proj))
plt.savefig(snakemake.output.fixed)

plot_frames(B_in_QU, vlim="asym", suptitle="{}".format(mask.proj))
plt.savefig(snakemake.output.dynamic)


# %%

# %%
