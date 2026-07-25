from spt3g.core import G3File
from spt3g import maps
from spt3g.mapspectra.map_analysis import constructEB
from spt3g_plotting import *

def plot_and_save_frame(frame, output, **kwargs):
    assert len(frame) == len(output)
    for (i, k) in enumerate(frame):
        frame[k].plot(**kwargs);
        plt.savefig(output[i])
        plt.clf()