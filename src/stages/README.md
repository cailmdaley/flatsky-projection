# Stage scripts — the data contract

Each file in `src/stages/` is a standalone `argparse` CLI. `astra.yaml` invokes them
through the dual-mode launcher at the repo root:

```
bash env.sh python3 src/stages/<stage>.py …
```

Every input arrives as an explicit flag, so a stage runs the same under lightcone,
on the grid, or by hand. The pipeline has no data inputs: every product descends
from the decisions, starting with the cosmology.

## Rules every stage follows

1. **`--out` is a directory.** The stage creates it and writes all its products inside.
   Lightcone hashes the directory, so nothing outside it counts as output.
2. **Every stage writes an index** — `<out>/index.json`, or `index_<coords>.json` for a
   unit job — naming the files it produced and the resolved parameters it used.
   Downstream stages read the index rather than globbing or reconstructing filenames.
3. **Decision flags take ASTRA option *ids*, not values.** `{decisions.ell_max}`
   substitutes the literal string `l6000`, so `--lmax` accepts `l4000` / `l6000` *and*
   a bare integer. The same for `--nside` (`nside4096` | `nside8192` | `4096`),
   `--interpolation` (`interp_rebin8` | `interp_rebin1` | `nearest`),
   `--apodization` (`c2_2p5deg` | `c1_2p5deg` | `c2_5deg`),
   `--purify-b` (`purify` | `standard` | `true` | `false`),
   `--pixel-window` (`flat_only` | `none` | `healpix_only` | `healpix_and_flat`),
   `--delta-ell` (`d50` | `d10`), `--cosmology` (`planck2018`),
   `--flattening` (`polar` | `spt3g`).
   That parsing lives in `src/stages/_decisions.py`, with the cosmology's CAMB
   parameters; import it, do not duplicate it.
4. **Threading is bounded.** Set `OMP_NUM_THREADS` from `--max-procs` (default 4)
   *before* importing numpy/healpy/spt3g.
5. **Seeds are explicit**: `--seeds 3` names a unit's seed; `--seed-start 0 --seed-stop 20`
   a range, inclusive-exclusive.
6. **Idempotent.** A per-seed product that already exists is skipped, and products are
   written through a temporary file renamed into place, so a killed run resumes.
7. **No relative-path magic.** Every path arrives through a flag, theory spectra
   included (`--theory`, the `theory_spectra` output directory holding
   `theory_spectra.npz`); a stage slices the multipoles it needs itself.

## Grid and geometry, fixed for every stage

`_footprints.py` holds the patch centre (RA 0, dec -56), the flat-sky grid
(`GRID_PIX` = 4500 pixels of `RES_ARCMIN` = 2 arcmin, 150 deg across) and the
footprint registry. Projections are named by `spt3g.maps.MapProjection`:
`ProjSFL ProjSTG ProjZEA ProjBICEP`.

## The files, in DAG order

### `theory_spectra.py`
`--cosmology --out`

CAMB 1.5.4 for the cosmology decision's parameters: lensed and unlensed TT, EE, BB,
TE as raw C_ell in uK^2 indexed from ell = 0 to 10000, written to
`<out>/theory_spectra.npz` with the parameters in a pickled `header`.

### `sky_alms.py`
`--spectrum cmb|white --theory --lmax --seeds --out`

One Gaussian sky per seed with `healpy.synalm`, band-limited at `--lmax`, numpy's
generator seeded from `SeedSequence((namespace, seed))`. `cmb`: T, E, B from the
unlensed TT, EE, TE with BB = 0, so B is exactly zero. `white`: one scalar sky with
C_ell = 1 for 2 <= ell <= lmax. Written by `hp.write_alm` to
`<out>/alms_seed{N:04d}.fits`, one HDU per component.

### `make_masks.py`
`--nside --apodization --footprint NAME… --out`

Footprints are named in `_footprints.py`, the one registry: seven RA/dec rectangles
about the patch centre, self-similar to SPT's 1500d field (`field_1500d`, scale 1),
the others named by their unapodized sky fraction (`rect_0p77` … `rect_10p06`). Each
is apodized analytically on the sphere with NaMaster's C1/C2 kernel at each pixel's
geodesic distance to the boundary and written as `<out>/mask_{footprint}.g3.gz` (a
`G3Frame` with key `T` holding a ring-sparse `HealpixSkyMap`). `index_{footprint}.json`
records its scale and box and the realized `fsky_binary`, `fsky_w1 = mean(mask)`,
`fsky_w2 = mean(mask**2)`; the apodized fractions are smaller than the label and are
the numbers downstream code must use.

### `project_qu.py` / `project_eb.py`
`--alms --projections --interpolation --nside [--flattening] --seeds --max-procs --out`

QU leg: `hp.alm2map_spin([alm_E, alm_B], nside, 2)` → `HealpixSkyMap(..., pol_conv=COSMO)`
for Q and U → `maps.maputils.healpix_to_flatsky` per projection → `_project.flatten`.
Flattening (`--flattening`) rotates Q/U by the polar-decomposition angle of the
projection's Jacobian (`polar`, a numpy rotation from `_project.polar_angle`), or by
`maps.flatten_pol`'s averaged angle (`spt3g`).
EB leg: `hp.alm2map(alm_E)` and `hp.alm2map(alm_B)` as **spin-0 scalars**, projected the
same way, with no flattening — this leg is leakage-free by construction.

Write `<out>/qu_{proj}_seed{N:04d}.g3` and `<out>/eb_{proj}_seed{N:04d}.g3`.

### `project_masks.py`
`--healpix-masks --footprint NAME… --projections --interpolation --out`

Projects each footprint's Healpix mask into each projection, zeroes NaNs, keeps the
heaviest connected component, and cuts the window NaMaster analyses: the tightest
even-sided rectangle around the support (the window sets the Fourier grid and
therefore the bandpowers). Writes `<out>/mask_{proj}_{footprint}.npz` with `mask`
(2-D float64), `lx_rad`, `ly_rad` and the window bounds `ylo yhi xlo xhi`, and
`index_{footprint}.json`, which also records each window's margin to the grid edge.

### `coupling_matrices.py`
`--masks --projections --footprint NAME… --purify-b --lmax --delta-ell --max-procs --out`

Per (projection, footprint): `nmt.NmtBinFlat(l0, lf)` with `l0 = arange(nbins)*delta_ell`,
`lf = (arange(nbins)+1)*delta_ell`, `nbins = lmax // delta_ell`; dummy spin-2 and
spin-0 fields over the mask; `w.compute_coupling_matrix(f, f, b)`, written to
`<out>/w22_{proj}_{footprint}.fits` and `<out>/w00_{proj}_{footprint}.fits`.

### `namaster_spectra.py`
`--qu-maps --eb-maps --masks --workspaces --projections --seeds --purify-b --pixel-window --lmax --delta-ell --max-procs --out`

For each leg, projection, footprint the workspaces index lists, and seed: cut the map
to that footprint's window, build `nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=…)`
(Q/U leg) or two spin-0 fields (E/B leg), then
`w.decouple_cell(nmt.compute_coupled_cell_flat(f, f, b))`.

Writes `<out>/bandpowers_{proj}_seed{N:04d}.npz` per unit:

```
ell_eff     (nbins,)
ee          (nleg=2, nproj, nfootprint, nseed, nbins)
bb          (same shape)
legs        ['qu', 'eb']  projections  footprints (names, registry order)  seeds
```

One array, explicitly indexed, with the leg, projection and footprint names stored
beside it, so downstream code cannot mis-index them.

Pixel-window handling: `flat_only` (the production mode) divides by the azimuthally
averaged flat-sky window from `spt3g.mapspectra.pixel_window.pixwin_flatsky(ells,
res_arcmin)` — bare arcmin, not G3Units — **squared** (it is a C_ell).
`healpix_only` divides by `hp.pixwin(8192)` binned onto the bandpowers, squared;
`healpix_and_flat` divides by both. The skies are drawn in harmonic space and carry
no HEALPix window, so the healpix modes over-correct and exist only for the
comparison.

### `scalar_project.py` / `scalar_spectra.py`
`--spectrum tt|white` selects the scalar sky (the T alms of `sky_alms`, or the white
alms). Maps are `<out>/{spectrum}_{proj}_seed{N:04d}.g3`; bandpowers are
`<out>/bandpowers_{spectrum}_{proj}_seed{N:04d}.npz` with `cl` of shape
`(nfootprint, nbins)` and `footprints`.

### The field suites

`fine_pseudo_spectra.py` and `coupling_matrices.py --delta-ell 5` give unit-width
1500d spectra of the production skies; `bin_covariance_sims.py`,
`bin_covariance_controls.py` and `noisy_inversion_sims.py` draw their own skies on
the sphere and pass them through the field pipeline for the bandpower covariance and
the response inversion. `astra.yaml` describes each one and gives its flags; the
module docstrings give the method.
