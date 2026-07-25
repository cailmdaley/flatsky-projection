from spt3g.core import G3File
from spt3g import maps
from spt3g.mapspectra.map_analysis import constructEB
from spt3g_plotting import *

frame = next(G3File(snakemake.input))

if "E" in snakemake.output.keys() and "E" not in frame.keys():
    for (k, f) in constructEB(frame).items():
        frame[k] = f.get_rmap()

for k in snakemake.output.keys():
    frame[k].plot()
    plt.savefig(snakemake.output[k])
    plt.clf()
