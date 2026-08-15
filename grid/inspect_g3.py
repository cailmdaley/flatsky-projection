"""Spot-verify a projected .g3 map returned from the grid.

Named inspect_g3.py, not inspect.py: a module named ``inspect`` on sys.path
shadows the stdlib ``inspect`` that numpy imports, and the whole spt3g import
fails with a circular-import error.

Graduated from osg_test/probe9_verify.py. Prints frame structure and, per map
key, the projection, resolution, patch centre, polarization flags, and simple
pixel statistics. Expect flat_pol=True on the QU leg (keys Q, U) and no
flat_pol on the EB leg, whose E and B are plain spin-0 T-like maps.

Run under the spt3g env-shell inside a small memory scope -- a single 3300 x
3300 float64 map is only ~87 MB, so 4 GiB is generous:

  systemd-run --user --scope -p MemoryMax=4G \
    ~/code/python/spt3g_software/build/env-shell.sh \
    python3 grid/inspect_g3.py \
      /sptlocal/user/cailmdaley/pol_project/results/highell/projected_eb/eb_ProjSFL_seed0000.g3
"""

import sys

import numpy as np
from spt3g import core, maps  # maps import registers FlatSkyMap for deserialization

assert maps  # silence linters; the import is load-bearing

path = sys.argv[1]
for i, frame in enumerate(core.G3File(path)):
    print(f"--- frame {i}: {frame.type}")
    print("keys:", list(frame.keys()))
    for key in frame:
        m = frame[key]
        a = np.asarray(m)
        print(f"  {key}: {type(m).__name__} shape={a.shape} proj={m.proj} "
              f"res={m.res / core.G3Units.arcmin:.3f} arcmin "
              f"alpha={m.alpha_center / core.G3Units.deg:.3f} deg "
              f"delta={m.delta_center / core.G3Units.deg:.3f} deg "
              f"flat_pol={getattr(m, 'flat_pol', None)} "
              f"pol_conv={getattr(m, 'pol_conv', None)}")
        print(f"     finite={np.isfinite(a).all()} nonzero={np.count_nonzero(a)} "
              f"mean={np.nanmean(a):.6e} std={np.nanstd(a):.6e} "
              f"min={np.nanmin(a):.6e} max={np.nanmax(a):.6e}")
