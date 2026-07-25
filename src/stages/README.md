# Stage scripts — the data contract

Each file in `src/stages/` is a standalone `argparse` CLI. `astra.yaml` invokes them
through the SPT-3G environment wrapper:

```
/home/cailmdaley/code/python/spt3g_software/build/env-shell.sh python src/stages/<stage>.py …
```

They were ported from `/sptlocal/analysis/pol_project2020/workflow/scripts/`, which is
the code that produced the 2021 numbers. That pipeline injected `snakemake.config` and
`snakemake.input`; these take the same values as explicit flags so they run under
lightcone, under Snakemake, or by hand.

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
   `--pixel-window` (`none` | `healpix_only` | `healpix_and_flat`).
   Put that parsing in `src/stages/_decisions.py` and import it; do not duplicate it.
4. **Threading is bounded.** Set `OMP_NUM_THREADS` from `--max-procs` (default 4)
   *before* importing numpy/healpy/spt3g. amundsen is a shared 24-core login node;
   nothing here may exceed 4 cores without being asked to.
5. **Seeds are explicit and inclusive-exclusive**: `--seed-start 0 --seed-stop 20`.
6. **Idempotent.** If a per-seed product already exists and is newer than its input,
   skip it. Long stages must be resumable after a kill.
7. **No `snakemake`, no `spt3g_plotting`, no relative-path magic.** Every path arrives
   through a flag.

## Grid and geometry, fixed for every stage

The flat-sky patch is 110 deg across at 2 arcmin, i.e. **3300 x 3300 pixels**, centred
on RA 0, dec -56 (the 2021 convention — note this is *not* the production 1500d centre
of -57.5). Projections are named by `spt3g.maps.MapProjection`:
`ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP`.

## The files, in DAG order

### `preprocess_healpix.py`
`--sims-dir --seed-start --seed-stop --lmax --max-procs --out`

Reads `<sims-dir>/TQU1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{N}.fits`
with `spt3g.maps.fitsio.load_skymap_fits(..., keys="TQU")`, runs
`hp.map2alm([T,Q,U], lmax=lmax, pol=True)`, zeroes the monopole and dipole with
`hp.almxfl`, writes `hp.write_alm` to `<out>/alms_seed{N:04d}.fits` (3 HDUs: T, E, B).

### `make_masks.py`
`--nside --apodization --ra-deg --dec-deg --radii-rad --out`

`hp.query_disc` at each radius, then `nmt.mask_apodization(mask, aposize, apotype)`.
Writes `<out>/mask_{i}.g3` (a `G3Frame` with key `T` holding a `HealpixSkyMap`).
Records in `index.json`, per mask: radius, `fsky_binary = mean(mask_binary)`,
`fsky_w2 = mean(mask_apodized**2)`, `fsky_w1 = mean(mask_apodized)`. The 2021 config
labelled these 0.75…10 % from the *unapodized* disk area; the realized `fsky_w2` is
smaller and is the number downstream code and the paper must use.

### `project_qu.py` / `project_eb.py`
`--alms --projections --interpolation --nside --res-arcmin --size-deg --ra-deg --dec-deg --seed-start --seed-stop --max-procs --out`

QU leg: `hp.alm2map(alms_TEB, nside, pol=True)[1:]` → `HealpixSkyMap(..., pol_conv=COSMO)`
for Q and U → `maps.maputils.healpix_to_flatsky` per projection → `maps.FlattenPol`.
EB leg: `hp.alm2map(alm_E)` and `hp.alm2map(alm_B)` as **spin-0 scalars**, projected the
same way, with *no* `FlattenPol` — this leg is leakage-free by construction.

Write `<out>/qu_{proj}_seed{N:04d}.g3` and `<out>/eb_{proj}_seed{N:04d}.g3`.

### `project_masks.py`
`--healpix-masks --projections --interpolation --res-arcmin --size-deg --ra-deg --dec-deg [--cut-bounds …] --out`

Projects each Healpix mask into each projection, `ApplyMapThreshold` to remove NaNs,
then **cuts a centred square sub-window** — NaMaster analyses that window, not the full
3300². Two modes:

* derived (default): smallest centred square containing every pixel with mask > 1e-6,
  rounded up to an even side.
* `--cut-bounds lo:hi lo:hi …`, one per mask, to force the 2021 hand-maintained table
  `1320:1980 1290:2010 1230:2070 1140:2160 960:2340 780:2520 300:3000`. **The baseline
  universe must pass this table**, because it changes the Fourier grid and therefore
  the bandpowers; it is what reproduces the 2021 numbers. Note the 2021 windows clip
  the outer part of the apodization taper on the small masks.

Writes `<out>/mask_{proj}_{i}.npz` with `mask` (2-D float64), `lo`, `hi`,
`lx_rad`, `ly_rad` where `lx_rad = (hi-lo) * res_arcmin/60 * pi/180`.

### `coupling_matrices.py`
`--masks --projections --purify-b --lmax --delta-ell --max-procs --out`

Per (projection, mask): `nmt.NmtBinFlat(l0, lf)` with `l0 = arange(nbins)*delta_ell`,
`lf = (arange(nbins)+1)*delta_ell`, `nbins = lmax // delta_ell`; a dummy
`nmt.NmtFieldFlat(lx, ly, mask, [zeros, zeros], purify_b=…)`; then
`w = nmt.NmtWorkspaceFlat(); w.compute_coupling_matrix(f, f, b)`;
`w.write_to(<out>/w22_{proj}_{i}.fits)`.

### `namaster_spectra.py`
`--qu-maps --eb-maps --masks --workspaces --projections --seed-start --seed-stop --purify-b --pixel-window --lmax --delta-ell --max-procs --out`

For each leg, projection, mask, seed: cut the map to that mask's window, build
`nmt.NmtFieldFlat(lx, ly, mask, [q, u], purify_b=…)`, then
`w.decouple_cell(nmt.compute_coupled_cell_flat(f, f, b))` and keep index 0 (EE) and
3 (BB).

Writes one `<out>/bandpowers.npz`:

```
ell_eff   (nbins,)
ee        (nleg=2, nproj, nmask, nseed, nbins)
bb        (same shape)
legs      ['qu', 'eb']  projections  masks(index)  seeds
```

One array, explicitly indexed. The 2021 script assigned `cl_flat_EE[ii]` inside the
projection loop and collapsed every projection onto the last one read; the shape above
is what makes that class of bug impossible.

Pixel-window handling: `healpix_only` divides by `hp.pixwin(nside_input)[:lmax+1]`
binned onto the bandpowers **squared** (it is a C_ell, so divide by `pixwin**2`);
`healpix_and_flat` additionally divides by the azimuthally averaged flat-sky window
from `spt3g.mapspectra.pixel_window.pixwin_flatsky(ells, res_arcmin)` — signature is
`(ells, reso_arcmin)` with bare arcmin, not G3Units.

### `eb_residuals.py`, `covariance.py`, `leakage_summary.py`
Downstream reductions; see `astra.yaml` for their flags. They read `bandpowers.npz`
and the `.g3` map files and write plain `.npz` / `.csv` into their own `--out`.

## Validating a port

`/sptlocal/analysis/pol_project2020/results/` holds the 2021 products, including
per-seed bandpower text files. A faithful port run under the **baseline** universe with
seed 0 must reproduce the corresponding 2021 row. That is the regression test; run it
on one seed and two projections before running anything at scale.
