"""Check beam-map cropping against production masks and pure-B workspaces."""

from pathlib import Path
import sys

import numpy as np
import pymaster as nmt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'stages'))
from beam_reprojection import mask_slices


def run():
    # A broadcast view tests the full-map crop without allocating a full map.
    full = np.broadcast_to(0., (2, 3300, 3300))
    res = np.deg2rad(2 / 60)
    lo = np.arange(0., 6000., 50.)
    bins = nmt.NmtBinFlat(lo, lo + 50)
    for proj in ('ProjZEA', 'ProjSTG'):
        for maskdir, workdir, index in (
                ('flatsky_masks', 'coupling_matrices', 5),
                ('field_flatsky_masks', 'field_coupling', 0)):
            metadata = np.load(ROOT / 'results' / 'highell' / maskdir / f'mask_{proj}_{index}.npz')
            mask = metadata['mask']
            ys, xs = mask_slices(metadata)
            pair = full[:, ys, xs]
            assert pair.shape == (2, *mask.shape)
            ny, nx = mask.shape
            lx, ly = float(metadata['lx_rad']), float(metadata['ly_rad'])
            np.testing.assert_allclose((lx, ly), (nx * res, ny * res))
            field = nmt.NmtFieldFlat(lx, ly, mask, list(pair), purify_b=True)
            coupled = nmt.compute_coupled_cell_flat(field, field, bins)
            workspace = nmt.NmtWorkspaceFlat()
            workspace.read_from(str(ROOT / 'results' / 'highell' / workdir / f'w22_{proj}_{index}.fits'))
            decoupled = workspace.decouple_cell(coupled)
            np.testing.assert_array_equal(decoupled, np.zeros((4, 120)))
            print(proj, maskdir, mask.shape, 'crop and workspace pass', flush=True)


if __name__ == '__main__':
    run()
