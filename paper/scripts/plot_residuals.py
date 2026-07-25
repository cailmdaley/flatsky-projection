from spt3g.core import G3File
from spt3g import maps
from spt3g.mapspectra.map_analysis import constructEB
from spt3g_plotting import *

projected_QU = next(G3File(snakemake.input.QU))
projected_EB = next(G3File(snakemake.input.EB))
residuals_EB = {k: projected_EB[k] - f.get_rmap()
    for (k, f) in constructEB(projected_QU).items()}

for k in snakemake.output.keys():
    residuals_EB[k].plot()
    plt.savefig(snakemake.output[k])
    plt.clf()

