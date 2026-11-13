# Stage scripts — the data contract

Each file in `src/stages/` is a standalone `argparse` CLI. `astra.yaml` invokes them
through the dual-mode launcher at the repo root:

```
bash env.sh python3 src/stages/<stage>.py …
```

Every input arrives as an explicit flag, so a stage runs the same under lightcone,
on the grid, or by hand.

## Rules every stage follows

1. **`--out` is a directory.** The stage creates it and writes all its products inside.
   Lightcone hashes the directory, so nothing outside it counts as output.
2. **Every stage writes `<out>/index.json`** — a manifest naming the files it produced
   and the resolved parameters it used (see below). Downstream stages read the manifest
   rather than globbing or reconstructing filenames.
3. **Decision flags take ASTRA option *ids*, not values.** `{decisions.ell_max}`
   substitutes the literal string `l4000`, so `--lmax` accepts `l4000` / `l6000` *and*
   a bare integer. The same for `--nside` (`nside4096` | `nside8192` | `4096`),
   `--interpolation` (`interp_rebin8` | `interp_rebin1` | `nearest`),
   `--apodization` (`c2_2p5deg` | `c1_2p5deg` | `c2_5deg`),
   `--purify-b` (`purify` | `standard` | `true` | `false`),
   `--pixel-window` (`flat_only` | `none` | `healpix_only` | `healpix_and_flat`),
   `--delta-ell` (`d50` | `d10`).
   Put that parsing in `src/stages/_decisions.py` and import it; do not duplicate it.
4. **Threading is bounded.** Set `OMP_NUM_THREADS` from `--max-procs` (default 4)
   *before* importing numpy/healpy/spt3g. amundsen is a shared 24-core login node;
   nothing here may exceed 4 cores without being asked to.
5. **Seeds are explicit and inclusive-exclusive**: `--seed-start 0 --seed-stop 20`.
6. **Idempotent.** If a per-seed product already exists and is newer than its input,
   skip it. Long stages must be resumable after a kill.
7. **No `snakemake`, no `spt3g_plotting`, no relative-path magic.** Every path arrives
   through a flag, theory spectra included (`--theory`, the `theory_spectra` input,
   `data/theory_spectra.npz`); a stage slices the multipoles it needs itself.

## Grid and geometry, fixed for every stage

The flat-sky patch is 110 deg across at 2 arcmin, i.e. **3300 x 3300 pixels**, centred
on RA 0, dec -56. Projections are named by `spt3g.maps.MapProjection`:
`ProjSFL ProjSTG ProjZEA ProjBICEP`.

## The files, in DAG order

### `preprocess_healpix.py`
`--sims-dir --seed-start --seed-stop --lmax --max-procs --out`

Reads `<sims-dir>/TQU1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{N}.fits`
with `spt3g.maps.fitsio.load_skymap_fits(..., keys="TQU")`, runs
`hp.map2alm([T,Q,U], lmax=lmax, pol=True)`, zeroes the monopole and dipole with
`hp.almxfl`, writes `hp.write_alm` to `<out>/alms_seed{N:04d}.fits` (3 HDUs: T, E, B).

### `make_masks.py`
`--nside --apodization --footprint NAME… --out`

Footprints are named in `_footprints.py`, the one registry of the analysis
footprints: seven disks (`disk_0p75` … `disk_10`) and SPT's 1500d field
(`field_1500d`), with each one's shape, flat-sky cut and nominal sky fraction, and
the patch centre. Each footprint is apodized analytically on the sphere with
NaMaster's C1/C2 kernel and written as `<out>/mask_{footprint}.g3.gz` (a `G3Frame`
with key `T` holding a ring-sparse `HealpixSkyMap`). `index_{footprint}.json`
records, per mask: its shape, `fsky_binary = mean(mask_binary)`,
`fsky_w1 = mean(mask_apodized)`, `fsky_w2 = mean(mask_apodized**2)`. The disks'
nominal 0.75…10 % labels come from the *unapodized* disk area; the realized
`fsky_w2` is smaller and is the number downstream code and the paper must use.

### `project_qu.py` / `project_eb.py`
`--alms --projections --interpolation --nside --res-arcmin --size-deg --ra-deg --dec-deg --seed-start --seed-stop --max-procs --out`

QU leg: `hp.alm2map(alms_TEB, nside, pol=True)[1:]` → `HealpixSkyMap(..., pol_conv=COSMO)`
for Q and U → `maps.maputils.healpix_to_flatsky` per projection → `maps.FlattenPol`.
EB leg: `hp.alm2map(alm_E)` and `hp.alm2map(alm_B)` as **spin-0 scalars**, projected the
same way, with *no* `FlattenPol` — this leg is leakage-free by construction.

Write `<out>/qu_{proj}_seed{N:04d}.g3` and `<out>/eb_{proj}_seed{N:04d}.g3`.

### `project_masks.py`
`--healpix-masks --footprint NAME… --projections --interpolation --res-arcmin --size-deg --out`

Projects each footprint's Healpix mask into each projection, zeroes NaNs, then cuts
the window NaMaster analyses. A disk's window is its square `lo:hi` from the registry
(a hand-maintained table; the window sets the Fourier grid and therefore the
bandpowers); the field's is the tightest even-sided rectangle around its support.
Writes `<out>/mask_{proj}_{footprint}.npz` with `mask` (2-D float64), `lx_rad`,
`ly_rad` and the window bounds `ylo yhi xlo xhi` (plus `lo hi` for a square), and
`index_{footprint}.json`.

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

Pixel-window handling: `flat_only` (the production mode for high-ell work) divides
by the azimuthally averaged flat-sky window from
`spt3g.mapspectra.pixel_window.pixwin_flatsky(ells, res_arcmin)` — signature is
`(ells, reso_arcmin)` with bare arcmin, not G3Units — **squared** (it is a C_ell).
`healpix_only` divides by `hp.pixwin(nside_input)[:lmax+1]` binned onto the
bandpowers, squared; `healpix_and_flat` divides by both. The healpix modes
over-correct — the unlensed_TQU inputs carry no Healpix window (Cl of the
preprocessed alms matches theory flat to ell = 6000) — and exist only for the
explicit comparison.

### `scalar_project.py` / `scalar_spectra.py`
`--spectrum tt|white` selects the scalar sky (the preprocessed T alms, or unit-C_l
white alms). Maps are `<out>/{spectrum}_{proj}_seed{N:04d}.g3`; bandpowers are
`<out>/bandpowers_{spectrum}_{proj}_seed{N:04d}.npz` with `cl` of shape
`(nfootprint, nbins)` and `footprints`.

### The simulation suites

The remaining stages each produce one suite of the paper's controls and corollaries
(covariance skies and controls, noisy inversion skies, the BB-floor controls, the
masked template, lensing reconstruction, compact sources, the beam test, spherical
pixel tests, lognormal filtering). `astra.yaml` describes each one and gives its
flags; the module docstrings give the method.
