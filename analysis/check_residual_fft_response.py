"""Independent 2D FFT gate for the radial NaMaster response.

Adapted from the independent Opus review's reviewer_fft_response.py.
This checks Fourier-cell weighting, normalization and scalar/EE mask coupling
on the real production grid. It does not assert stationarity of projected skies.
The padded FFT and NaMaster radial approximations need not agree exactly.
"""
from pathlib import Path
import json
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import pymaster as nmt
import spectra as S

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')


def main():
    reference = np.load(ROOT/'residual_estimator_response.npz')
    rows = []
    bins = nmt.NmtBinFlat(np.arange(0., 6000., 50.), np.arange(50., 6050., 50.))
    ell = np.arange(8001.)
    for proj in S.PROJS:
        with np.load(S.ROOT/'results/highell/field_flatsky_masks'/f'mask_{proj}_0.npz') as d:
            mask = d['mask'].astype(float)
            lx, ly = float(d['lx_rad']), float(d['ly_rad'])
        ny, nx = mask.shape
        dx, dy = lx/nx, ly/ny
        kx, ky = np.meshgrid(fftfreq(nx, 1/nx)*2*np.pi/lx,
                             fftfreq(ny, 1/ny)*2*np.pi/ly)
        k, psi = np.hypot(kx, ky), np.arctan2(ky, kx)
        index = (k/50).astype(int).ravel()
        count = np.bincount(index, minlength=120)[:120]

        def binmean(power):
            return np.bincount(index, power.ravel(), minlength=120)[:120]/count

        # No fitted normalization: compare one identical masked white map.
        sky = np.random.default_rng(0).normal(size=mask.shape)
        field = nmt.NmtFieldFlat(lx, ly, mask, [sky])
        nmt_power = nmt.compute_coupled_cell_flat(field, field, bins)[0]
        own_power = binmean(abs(fft2(mask*sky)*dx*dy)**2/(lx*ly))
        clean = (S.ELL >= 100) & (S.ELL < 4800)
        np.testing.assert_allclose(nmt_power[clean], own_power[clean], rtol=1e-10, atol=0.)
        del field, sky
        # The mask autocorrelation and covariance use a twice-padded grid.
        # Evaluate the result at the original Fourier cells (even indices).
        NY, NX = 2*ny, 2*nx
        correlation = ifft2(abs(fft2(mask, s=(NY, NX)))**2).real*dx*dy
        qx, qy = np.meshgrid(fftfreq(NX, 1/NX)*np.pi/lx,
                             fftfreq(NY, 1/NY)*np.pi/ly)
        q, phi = np.hypot(qx, qy), np.arctan2(qy, qx)

        def pseudo(cq):
            covariance = ifft2(cq).real*NX*NY/(4*lx*ly)
            return (fft2(correlation*covariance).real*dx*dy/(lx*ly))[::2, ::2]

        for spec, spin in [('TT', 0), ('EE', 2)]:
            native = reference[f'{proj}_{spec}_native']
            cq = np.interp(q, ell, native)
            if spin == 0:
                power = pseudo(cq)
            else:
                power = .5*(pseudo(cq)+np.cos(4*psi)*pseudo(cq*np.cos(4*phi))+
                            np.sin(4*psi)*pseudo(cq*np.sin(4*phi)))
            ws = nmt.NmtWorkspaceFlat()
            ws.read_from(str(S.ROOT/'results/highell/field_coupling'/f'w{spin}{spin}_{proj}_0.fits'))
            cl = np.zeros((1 if spin == 0 else 4, len(ell)))
            cl[0] = native
            coupled = ws.couple_cell(ell, cl)[0]
            ratio = binmean(power)/coupled-1
            # Independent quadrature, not an algebraic couple/decouple
            # identity. Review measured <0.3% per bin for ZEA/STG; retain a
            # conservative 0.5% envelope for all four production grids.
            np.testing.assert_allclose(ratio[clean], 0., rtol=0., atol=.005)
            rows.append(dict(projection=proj, spectrum=spec,
                             max_coupled_fraction=float(abs(ratio[clean]).max()),
                             rms_coupled_fraction=float(np.sqrt(np.mean(ratio[clean]**2)))))
            print(rows[-1], flush=True)
    (ROOT/'residual_fft_gate.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
