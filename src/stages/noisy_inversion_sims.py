"""Noisy SPT-3G-like field skies for inverting the projection response.

One lensed Gaussian sky per unit (TT, EE, BB, TE from the lensed spectra),
convolved on the sphere with a Gaussian beam, projected with the production
interpolation and flattening, then observed as two independent noise splits.
Noise is white and drawn per grid pixel, never projected. Its variance per
pixel scales as the pixel's area magnification |det J| (plane area over sky
area): a fixed instrument depth per sky area, so equal-area charts get uniform
noise and STG's outer pixels, which cover less sky, are noisier.

Kept per projection, all unit-width coupled pseudo-spectra on the field mask:
  tt_cross, pol_cross   split 1 x split 2 (no noise bias)
  tt_signal, pol_signal the noise-free sky, for attributing variance
"""
from pathlib import Path
import argparse
import time

from _decisions import parse_flattening, parse_lmax, parse_nside, parse_purify_b, parse_interpolation, set_threads, write_index
from _project import flatten
from namaster_spectra import load_mask, finite_in_mask, atomic_save
from bin_covariance_sims import mask_stub, project_array
from _footprints import DEC_DEG, FIELD, RES_ARCMIN


def area_magnification(proj, stub):
    """Plane area over sky area for every pixel of the stub."""
    import numpy as np
    from spt3g import core, maps
    if proj != 'ProjSTG':
        return np.ones(stub.shape)  # ZEA and SFL preserve area.
    ra, dec = (np.asarray(a)/core.G3Units.rad for a in maps.get_ra_dec_map(stub))
    d0 = np.deg2rad(DEC_DEG)
    cos_theta = np.sin(dec)*np.sin(d0) + np.cos(dec)*np.cos(d0)*np.cos(ra)
    # STG: r = 2 tan(theta/2), area scale sec^4(theta/2) = (2/(1+cos theta))^2.
    return (2/(1+cos_theta))**2


def run(args):
    set_threads(args.max_procs)
    import numpy as np
    import healpy as hp
    import pymaster as nmt
    from spt3g import maps
    from fine_pseudo_spectra import mode_counts

    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.theory / 'theory_spectra.npz') as z:
        cls = [z[f'lensed_{k}'][:args.lmax+1] for k in ('TT', 'EE', 'BB', 'TE')]
    beam = hp.gauss_beam(np.deg2rad(args.beam_fwhm/60), lmax=args.lmax)
    bins = nmt.NmtBinFlat(np.arange(args.lmax, dtype=float), np.arange(1, args.lmax+1, dtype=float))
    interpolation = parse_interpolation(args.interpolation)
    pixel_arcmin = RES_ARCMIN
    setup = {}
    for proj in args.projections:
        mask, window, lx, ly = load_mask(args.masks, proj, FIELD)
        stub = mask_stub(proj, window)
        setup[proj] = (mask, stub, lx, ly, np.sqrt(area_magnification(proj, stub)))

    # Offset +200000 separates these seeds from every other ensemble.
    seed = 200000+args.seed
    np.random.seed(seed)
    at, ae, ab = hp.synalm(cls, lmax=args.lmax, new=True)
    for a in (at, ae, ab):
        hp.almxfl(a, beam, inplace=True)
    noise = np.random.default_rng(seed)
    out = {proj: {} for proj in setup}

    def spectra(proj, comps, sigma, spin):
        mask, stub, lx, ly, scale = setup[proj]
        noisy = [[c + noise.standard_normal(c.shape)*sigma*scale for c in comps] for _ in range(2)]
        kw = dict(purify_b=args.purify_b) if spin else {}
        f0, f1, f2 = [nmt.NmtFieldFlat(lx, ly, mask, m, lite=True, **kw)
                      for m in (comps, *noisy)]
        key = 'pol' if spin else 'tt'
        out[proj][key+'_signal'] = nmt.compute_coupled_cell_flat(f0, f0, bins)
        out[proj][key+'_cross'] = nmt.compute_coupled_cell_flat(f1, f2, bins)

    q, u = hp.alm2map_spin([ae, ab], args.nside, 2, args.lmax)
    del ae, ab
    hq = maps.HealpixSkyMap(q, pol_conv=maps.MapPolConv.COSMO)
    del q
    hu = maps.HealpixSkyMap(u, pol_conv=maps.MapPolConv.COSMO)
    del u
    sigma_pol = args.noise_pol*np.sqrt(2)/pixel_arcmin  # per split, per pixel
    for proj, (mask, stub, *_) in setup.items():
        qmap, umap = [project_array(h, stub, interpolation) for h in (hq, hu)]
        flatten(qmap, umap, args.flattening)
        comps = [finite_in_mask(np.asarray(m), mask) for m in (qmap, umap)]
        spectra(proj, comps, sigma_pol, True)
        print(f'{proj} polarization seed {args.seed}: {time.monotonic()-start:.1f}s', flush=True)
    del hq, hu
    t = hp.alm2map(at, args.nside, lmax=args.lmax)
    del at
    ht = maps.HealpixSkyMap(t, pol_conv=maps.MapPolConv.COSMO)
    del t
    sigma_t = sigma_pol/np.sqrt(2)  # temperature depth is polarization's over sqrt(2)
    paths = []
    for proj, (mask, stub, lx, ly, _) in setup.items():
        comps = [finite_in_mask(np.asarray(project_array(ht, stub, interpolation)), mask)]
        spectra(proj, comps, sigma_t, False)
        path = args.out/f'noisy_{proj}_seed{args.seed:04d}.npz'
        atomic_save(path, **out[proj], counts=mode_counts(mask.shape, lx, ly, args.lmax),
                    projection=proj, seed=args.seed, nside=args.nside, lmax=args.lmax,
                    purify_b=args.purify_b, interpolation=args.interpolation,
                    beam_fwhm_arcmin=args.beam_fwhm, noise_pol_uk_arcmin=args.noise_pol,
                    noise_t_uk_arcmin=args.noise_pol/np.sqrt(2), splits=2,
                    lx_rad=lx, ly_rad=ly, pixel_window='none')
        paths.append(path.name)
        print(f'{proj} TT seed {args.seed}: {time.monotonic()-start:.1f}s', flush=True)
    write_index(args.out, dict(seed=args.seed, spectra=paths, nside=args.nside,
                               lmax=args.lmax, seconds=time.monotonic()-start),
                name=f'index_seed{args.seed:04d}.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--theory', type=Path, required=True, help='theory_spectra output directory')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--nside', type=parse_nside, required=True)
    p.add_argument('--lmax', type=parse_lmax, required=True)
    p.add_argument('--purify-b', type=parse_purify_b, required=True)
    p.add_argument('--interpolation', required=True)
    p.add_argument('--flattening', type=parse_flattening, required=True)
    p.add_argument('--noise-pol', type=float, default=5., help='coadded polarization depth, uK-arcmin')
    p.add_argument('--beam-fwhm', type=float, default=1.2, help='arcmin')
    p.add_argument('--projections', nargs='+', default=['ProjZEA', 'ProjSFL', 'ProjSTG'])
    p.add_argument('--max-procs', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
