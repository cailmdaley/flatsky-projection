from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from matplotlib.pyplot import colorbar, figure, gca, gcf, subplot, tight_layout
from matplotlib.ticker import ScalarFormatter
from numpy import array, broadcast, empty, isfinite, nan, percentile, sqrt
from numpy.fft import ifftshift

from spt3g.maps import FlatSkyMap, G3SkyMapWeights, MapProjection
from spt3g.core import G3Frame, G3FrameObject, G3Units
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum1D, MapSpectrum2D


# todo: properly attach as a method of FlatSkyMap
def plot_FlatSkyMap(
    self,
    vlim=None,
    vscale=None,
    cmap=None,
    title=None,
    title_prefix=None,
    tickunits=G3Units.deg,
    ticklabels=True,
    cbar=True,
    ax=None,
    **kwargs
):

    if ax is None:
        ax = gca()
    if vlim is None:
        vlim = "sym"
    if vscale is None:
        vscale = "linear"
    if cmap is None:
        cmap = "RdBu_r"
    if title is None:
        title = "%sMap (%i×%i @ %.2g')" % (
            title_prefix or "",
            *self.shape,
            self.res / G3Units.arcmin,
        )

    if vscale not in ["log", "linear"]:
        raise ValueError("`vscale` should be 'log' or 'linear'")

    arr = array(self)

    # upper/lower limits
    if vlim == "sym":
        vmax = percentile(abs(arr[isfinite(arr)]), 99.9)
        vmin = -vmax
    elif vlim == "asym":
        (vmin, vmax) = (percentile(arr[isfinite(arr)], q) for q in (0.1, 99.9))
    elif isinstance(vlim, tuple):
        (vmin, vmax) = vlim
    else:
        (vmin, vmax) = -vlim, vlim

    (y_extent, x_extent) = array(self.shape) * (self.res / tickunits) / 2
    extent = [x_extent, -x_extent, -y_extent, y_extent]

    cax = ax.matshow(
        arr,
        vmin=vmin,
        vmax=vmax,
        norm=(LogNorm() if vscale == "log" else None),
        cmap=cmap,
        rasterized=True,
        extent=extent,
        origin="lower",
        **kwargs
    )

    if cbar:
        colorbar(cax, ax=ax)
    if title is not None:
        ax.set_title(title, y=1)

    if ticklabels:

        class MyFmt(ScalarFormatter):
            def __call__(self, v, p=None):
                return (
                    super(MyFmt, self).__call__(v, p)
                    + {G3Units.deg: "°", G3Units.arcmin: "′"}[tickunits]
                )

        ax.xaxis.set_major_formatter(MyFmt())
        ax.yaxis.set_major_formatter(MyFmt())
        ax.tick_params(labeltop=False, labelbottom=True)
    else:
        ax.tick_params(labeltop=False, labelleft=False)


# todo: properly attach as a method of MapSpectrum2D
def plot_MapSpectrum2D(
    self,
    vlim=None,
    vscale=None,
    cmap=None,
    title=None,
    title_prefix=None,
    cbar=True,
    ticklabels=True,
    axeslabels=False,
    ax=None,
    **kwargs
):

    if ax is None:
        ax = gca()
    if vlim is None:
        vlim = "asym"
    if vscale is None:
        vscale = "log"
    if cmap is None:
        cmap = copy(get_cmap("viridis"))
        cmap.set_bad("lightgray")
    if title is None:
        title = "%sFourier (%i×%i @ %.2g')" % (
            title_prefix or "",
            self.map_nx,
            self.map_ny,
            sqrt(self.dx * self.dy) / G3Units.arcmin,
        )

    if vscale not in ["log", "linear"]:
        raise ValueError("`vscale` should be 'log' or 'linear'")
    if not (vlim == "asym" or isinstance(vlim, tuple)):
        raise ValueError("`vlim` should be 'asym' or a tuple of (vmin,vmax)")

    arr = abs(array(self.get_complex()))
    if vscale == "log":
        arr = abs(arr)
        arr[arr == 0] = nan

    lx, ly = self.get_lxly()
    extent = (lx.min(), lx.max(), ly.min(), ly.max())

    if vlim == "asym":
        (vmin, vmax) = (percentile(arr[isfinite(arr)], q) for q in (0.1, 99.9))
    elif isinstance(vlim, tuple):
        (vmin, vmax) = vlim

    cax = ax.matshow(
        ifftshift(arr),
        norm=(LogNorm() if vscale == "log" else None),
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs
    )

    if cbar:
        colorbar(cax, ax=ax)

    if title is not None:
        ax.set_title(title, y=1)

    if ticklabels:
        ax.tick_params(labeltop=False, labelbottom=True)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(45)
    else:
        ax.tick_params(labeltop=False, labelleft=False)

    ax.set_xlabel(r"$\ell_x$")
    ax.set_ylabel(r"$\ell_y$")


def safe_atleast_2d(ary, scalar_types):
    """
    Ensure `ary` is at least a 2d array similar to how `np.atleast_2d` works, but
    treat any type in `scalar_types` as a scalar even if its array-like. E.g
    
        >>> safe_atleast_2d([(1,2)], scalar_types=tuple)
        array([[(1, 2)]], dtype=object)
        
    """

    def is_scalar(x):
        try:
            x[0]
        except:
            return True
        else:
            return isinstance(x, scalar_types)

    if is_scalar(ary):
        (nrows, ncols) = (1, 1)
        ary = [[ary]]
    elif is_scalar(ary[0]):
        (nrows, ncols) = (1, len(ary))
        ary = [ary]
    else:
        (nrows, ncols) = (len(ary), len(ary[0]))

    ary_2d = empty((nrows, ncols), dtype=object)
    ary_2d[:] = ary
    return ary_2d



def plot_frames(
    fields,
    which=None,
    vlim=None,
    vscale=None,
    mult=1,
    title=None,
    suptitle=None,
    cmap=None,
    fig=None,
    plotsize=4,
    **kwargs
):
    """
    Plot a 1d or 2d array of FlatSkyMaps or MapSpectrum2Ds as an array of plots. 
    
    For example, if `m1`, `m2`, `m3`, and `m4` are `FlatSkyMap`s, then
    
        plot_frames([[m1,m2],[m3,m4]])
        
    makes a 2×2 array of plots showing the four maps.
    
    Additionally, the arguments `vlim`, `vscale`, `title`, and `cmap` can each
    be scalars, vectors, or 2d arrays, and "broadcast" across subplots according
    to usual Numpy broadcasting rules. E.g. 
    
        plot_frames([[m1,m2],[m3,m4]], vlim=[10, 5])
        
    makes the first column in the 2×2 array of plots have `vlim=10`, and the
    second column have `vlim=5`.
    
    If plotting a "frame" which has some keys, these are automatically splayed
    out in the x-direction. E.g. if `f1` and `f2` were `{"T":FlatSkyMap(...),
    "Q":FlatSkyMap(...), "U":FlatSkyMap(...)}`, then `plot_frames([f1,f2])`
    would plot a 2×3 array of plots where the first row was `f1`, the second row
    `f2`, and each column corresponded to T, Q, and U. 
    """

    scalar_types = (FlatSkyMap, MapSpectrum2D, tuple, str)
    frame_types = (G3Frame, G3FrameObject, dict)

    def is_frame(x):
        return isinstance(x, frame_types) and not isinstance(x, scalar_types)

    # if we're plotting frames, expand the polarization keys along the x-axis of
    # the plot array
    pol = None
    if is_frame(fields):
        pol = list(which) if which is not None else canonical_pol_order(_keys(fields))
        fields = [_getitem(fields,k) for k in pol]
    elif is_frame(fields[0]):
        pol = (
            list(which)
            if which is not None
            else [canonical_pol_order(_keys(field)) for field in fields]
        )
        fields = [[_getitem(field,k) for k in p] for (p, field) in zip(pol, fields)]

    bc = broadcast(
        *(
            safe_atleast_2d(x, scalar_types)
            for x in (fields, vlim, vscale, title, pol, cmap)
        )
    )

    if fig is None:
        fig = figure(figsize=([1.4, 1] * array(bc.shape)[::-1] * plotsize))

    for i, (field, vlim, vscale, title, pol, cmap) in enumerate(bc, 1):
        if field is not None:
            ax = subplot(*bc.shape, i)
            (mult * field).plot(
                ax=ax,
                vlim=vlim,
                vscale=vscale,
                title=title,
                title_prefix=(pol + " " if pol else None),
                cmap=cmap,
                **kwargs
            )

    fig.suptitle(suptitle)#, y=1.05)
    tight_layout(w_pad=-10)



def canonical_pol_order(pols):
    canonical_pols = ["T", "E", "B", "Q", "U", "TT", "EE", "BB", "QQ", "UU"]
    frame_keys = [k for k in canonical_pols if k in pols]
    for pol in pols:
      if pol not in frame_keys:
        frame_keys.append(pol)
    return frame_keys
    

def _keys(obj):
    try:
        return obj.keys()
    except:
        return obj.__dir__()
    
def _getitem(obj,k):
    try:
        return obj[k]
    except:
        return getattr(obj,k)


# todo: properly make these methods instead of doing this:
FlatSkyMap.plot = plot_FlatSkyMap
MapSpectrum2D.plot = plot_MapSpectrum2D
