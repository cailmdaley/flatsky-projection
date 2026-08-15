"""Verify the .g3 map probe9 brought back from the grid.

Run under the spt3g env-shell inside a 4 GiB memory scope:
  systemd-run --user --scope -p MemoryMax=4G \
    ~/code/python/spt3g_software/build/env-shell.sh \
    python3 osg_test/probe9_verify.py \
      osg_test/scratch/probe9_out/qu_ProjSFL_seed0000.g3
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
