"""Gate exact Fourier-cell rebinning and coupling-response aggregation on toy maps.

Run with `OMP_NUM_THREADS=2 bash env.sh python3 analysis/check_fine_estimator.py`.
"""
import sys
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'stages'))
import numpy as np
import pymaster as nmt
from astropy.io import fits
from tempfile import TemporaryDirectory
from fine_pseudo_spectra import mode_counts
from spectra import rebin_pseudo, rebin_flat_response, decouple_fine


def main():
    rng = np.random.default_rng(1904)
    for shape in [(32, 34), (33, 35)]:
        lx, ly = .8, .7
        yy, xx = np.indices(shape)
        mask = np.sin(np.pi * yy / (shape[0]-1))**4 * np.sin(np.pi * xx / (shape[1]-1))**4
        fine_edges = np.arange(0, 101)
        source_edges = np.arange(0, 101, 10)
        edges = np.arange(0, 101, 50)
        fine = nmt.NmtBinFlat(fine_edges[:-1], fine_edges[1:])
        source = nmt.NmtBinFlat(source_edges[:-1], source_edges[1:])
        coarse = nmt.NmtBinFlat(edges[:-1], edges[1:])
        counts = mode_counts(shape, lx, ly, 100)
        for spin in (0, 2):
            field = nmt.NmtFieldFlat(lx, ly, mask,
                                     rng.normal(size=(1 if spin == 0 else 2, *shape)),
                                     purify_b=spin == 2)
            cf = nmt.compute_coupled_cell_flat(field, field, fine)
            cc = nmt.compute_coupled_cell_flat(field, field, coarse)
            np.testing.assert_allclose(rebin_pseudo(cf, counts, edges), cc, atol=1e-14, rtol=1e-12)
            ws, wc = nmt.NmtWorkspaceFlat(), nmt.NmtWorkspaceFlat()
            ws.compute_coupling_matrix(field, field, source)
            wc.compute_coupling_matrix(field, field, coarse)
            with TemporaryDirectory() as tmp:
                a, b = Path(tmp)/'a.fits', Path(tmp)/'b.fits'
                ws.write_to(str(a)); wc.write_to(str(b))
                with fits.open(a) as fa, fits.open(b) as fb:
                    matrix = rebin_flat_response(fa['MCM_BINNED'].data, source_edges,
                                                 edges, counts, 1 if spin == 0 else 4)
                    np.testing.assert_allclose(matrix, fb['MCM_BINNED'].data, atol=1e-13, rtol=1e-10)
                    np.testing.assert_array_equal(np.diff(np.r_[0,counts.cumsum()][edges]),
                                                  fb['N_CELLS'].data.field(0))
            if spin == 2:
                with TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    ws_path = tmp / 'w00_Toy_0.fits'
                    wc.write_to(str(ws_path))
                    with fits.open(ws_path, mode='update') as hdus:
                        hdus['FS_INFO'].header['NX'] = shape[1] + 1
                    pseudo_path = tmp / 'pseudo.npz'
                    np.savez(pseudo_path, pol=cf, tt=cf[:1], counts=counts,
                             projection='Toy', mask_index=0, purify_b=True,
                             nx=shape[1], ny=shape[0], lx_rad=lx, ly_rad=ly,
                             mask_sha256='a'*64)
                    try:
                        decouple_fine(pseudo_path, workspace=tmp)
                    except ValueError as error:
                        assert 'geometry differs' in str(error), str(error)
                    else:
                        raise AssertionError('wrong-footprint workspace must raise')
                    scalar = nmt.NmtFieldFlat(lx, ly, mask, rng.normal(size=(1, *shape)))
                    tt = nmt.compute_coupled_cell_flat(scalar, scalar, fine)
                    mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()
                    np.savez(pseudo_path, pol=cf, tt=tt, counts=counts,
                             projection='Toy', mask_index=0, purify_b=True,
                             nx=shape[1], ny=shape[0], lx_rad=lx, ly_rad=ly,
                             mask_sha256=mask_hash)
                    mask_dir = tmp / 'masks'
                    mask_dir.mkdir()
                    np.savez(mask_dir / 'mask_Toy_0.npz', mask=mask,
                             ylo=0, yhi=shape[0], xlo=0, xhi=shape[1], lx_rad=lx, ly_rad=ly)
                    result = decouple_fine(pseudo_path, masks_dir=mask_dir, cache_dir=tmp/'cache')
                    from spt3g.mapspectra.pixel_window import pixwin_flatsky
                    window = pixwin_flatsky(coarse.get_effective_ells(), 2.)**2
                    expected = wc.decouple_cell(cc)
                    np.testing.assert_allclose(result['ee'], expected[0]/window, atol=1e-13, rtol=1e-10)
                    np.testing.assert_allclose(result['bb'], expected[3]/window, atol=1e-13, rtol=1e-10)
                    assert (tmp/'cache'/'d50'/'Toy'/mask_hash/'w22_Toy_0.fits').exists()
            print(f'{shape} spin{spin}: exact pseudo-spectrum and response rebinning passed')
    # Low-ell TT can exceed high-ell TT by many orders of magnitude.
    dynamic = np.array([[1e20, 1e20, 1., 2.]])
    np.testing.assert_array_equal(rebin_pseudo(dynamic, np.ones(4), [0, 2, 4]), [[1e20, 1.5]])
    try:
        rebin_pseudo(cf, counts, np.arange(101))
    except ValueError:
        pass
    else:
        raise AssertionError('unit-width empty target bins must raise')


if __name__ == '__main__':
    main()
