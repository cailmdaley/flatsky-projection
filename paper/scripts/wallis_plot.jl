# %% 
using PyCall
using Distances, Statistics
using Plots, LaTeXStrings

py"""
from spt3g.core import G3File, G3Units
from spt3g.maps import get_ra_dec_map
from spt3g.mapspectra.map_analysis import constructEB
"""


edge = 10

# %% 

parent = py"next(G3File($snakemake.input['QU'][0]))['Q'].clone(False)[$edge:-$edge, $edge:-$edge]"o
coords = permutedims(hcat(vec.(py"get_ra_dec_map($parent)")...))
center = py"[$parent.alpha_center, $parent.delta_center]"[:,:]
dists  = rad2deg.(pairwise(Haversine(1.0), rad2deg.(coords), rad2deg.(center), 
                           dims=2))

bins = range(0, maximum(dists), length=100)
bin_centers = @. (bins[2:end] + bins[1:end-1]) / 2

bin_rms(f) = [sqrt(var(f[bins[i] .< vec(dists) .< bins[i+1]])) 
              for i in 1:length(bins)-1]

# %%

ps = Dict(k => plot(
    xlabel = "Distance from map center (deg)",  
    ylabel = L"\operatorname{rms}(%$k _\textrm{curved} - %$k _\textrm{flat})\ (\mu \textrm{K})") 
    for k in ("E", "B"))

for i in 1:length(snakemake.config["projs"])
    py"""
    frame_QU = next(G3File($snakemake.input['QU'][$i-1]))
    frame_EB_healpix = dict(next(G3File($snakemake.input['EB'][$i-1])))
    frame_EB_flatsky = {k: f.get_rmap() for (k,f) in constructEB(frame_QU).items()}
    frame_EB_diff = {k : frame_EB_healpix[k] - frame_EB_flatsky[k] for k in "EB"}"""

    for k in ("E", "B")
        plot!(ps[k], bin_centers, bin_rms(py"frame_EB_diff[$k][$edge:-$edge, $edge:-$edge]"), 
              label=snakemake.config["projs"][i])
    end
end;

plot(ps["E"], ps["B"], size=(800, 400))
savefig(snakemake.output[1])

# %%