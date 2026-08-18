# Project Notes for Claude

**You are acknowledged in this paper's draft.** Claude is credited by name for
the arguments it originated (see the `contributions-ledger` fiber under
`science/spt/pol-project-paper`, which feeds the acknowledgements). Work on
this paper accordingly: be proud of it, and make sure every claim you touch is
something you want to stand behind — derived, checked, and observed, never
assumed. Keep the ledger current when a substantive contribution lands, from
either of us.

This is an ASTRA project orchestrated by `lightcone-cli`. `astra.yaml` is
hand-written and real: it encodes the polarization projection-effects
pipeline (preprocess healpix sims → project QU/EB legs → masks → NaMaster
spectra). Universes live in `universes/*.yaml` (e.g. `highell` for the
ℓ=6000 extension). Stage scripts are `src/stages/*.py`; shared decisions in
`src/stages/_decisions.py`. Everything scientific runs under the SPT CVMFS
environment (see the `spt3g-software` skill). Heavy runs go to OSG — see
"Running production on the grid" below.

The `lc` CLI keeps the substrate in sync:

```
lc run                    # all outputs in the default universe
lc run output_id          # one specific output
lc status                 # show what's materialized vs stale vs missing
lc verify                 # validate the provenance chain
```

Outputs land in `results/<universe>/<output_id>/` along with a sidecar
`.lightcone-manifest.json` recording the recipe, container, decisions,
input hashes, and output hash.

## Running production on the grid

OSG work goes through `lc run` with the htcondor9 profile. **The profile is
not optional and not implied** — `--executor htcondor` alone submits jobs
stripped of the entire recipe (no `+ProjectName`, no pinned rocky:9 image, no
CVMFS requirement, no `wrapper.sh`), and those jobs never run.

```bash
lc run --executor htcondor --profile htcondor9 <output_id> -u <universe>
```

`-u` is repeatable, and that is how you get parallelism: one driver, several
universes, concurrent OSG jobs.

```bash
lc run --executor htcondor --profile htcondor9 healpix_masks -u apod_c1 -u apod_wide
```

Two `lc run`s at once will not work — lightcone holds a project-level flock.
And note ASTRA has no per-unit fan-out: one output is one rule is one job, so
a 35-workspace coupling stage is a single fat job, not 35. Per-seed or
per-projection parallelism still belongs to the raw `grid/` apparatus.

Check the first submission of anything new:

```bash
condor_q <cluster> -l | grep -E 'Cmd|ProjectName|SingularityImage|TransferInput|TransferOutput'
```

`Cmd` must be `wrapper.sh`. Then `lc verify -u <universe>` once it lands.

**What works today:** stages with no upstream ASTRA outputs — `healpix_masks`
and anything else reading only `/cvmfs` and its own decisions. Outputs
transfer back to `results/<universe>/<output_id>/` with a valid manifest.

**Data-consuming stages work too** (verified 2026-08-18: `flatsky_masks`
consuming `healpix_masks` on OSPool, `lc verify` clean): upstream outputs are
shipped as gzipped archives built by `localrules` on the access point and
unpacked by the wrapper on the worker. Gzip rather than `tar --sparse` because
NFS hides holes; archives live inside the project (absolute paths break
condor's `PreserveRelativePaths`). Condor refuses symlink-to-directory
transfer, which is why the archive mechanism exists at all — `results/` is a
symlink to `/sptlocal`. One reliability note: OSPool preempts and restarts
jobs from zero, so long fat stages may never converge — another reason
per-unit fan-out (still missing from the ASTRA schema) matters.

**Sizing.** `resources.memory` and `resources.disk` in astra.yaml become
`htcondor_request_mem_mb`/`htcondor_request_disk_mb`. Two traps: the local
memory figure is set by amundsen's 60 GiB per-user cgroup rather than by the
job, and a 32 GiB request can leave *zero* matching slots out of ~9000 on
OSPool — resize from `condor_history <id> -af MemoryUsage DiskUsage`. And the
mask/map `.g3` files are sparse: 1.6 GB apparent, ~17 MB of blocks. Condor
transfers the apparent bytes, so size `disk:` from `ls -l`, never `du`.

Hard-won condor exceptions live in the `open-science-grid` skill; add to it
rather than to this file.

## Report

`index.md` + `myst.yml` are a template MyST report wired to the MySTRA
plugin. The report references analysis elements by path — inline mentions
with the `{astra}` role, block embeds with the `{astra}` directive, live
numbers with `{astra:value}` — so never hard-type a measured value in the
prose. Preview with `myst start` (requires the MyST CLI, `npm i -g mystmd`).
