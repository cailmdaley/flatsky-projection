# Project Notes for Claude

This is an ASTRA project orchestrated by `lightcone-cli`. `astra.yaml` is
hand-written and real: it encodes the polarization projection-effects
pipeline (preprocess healpix sims → project QU/EB legs → masks → NaMaster
spectra). Universes live in `universes/*.yaml` (e.g. `highell` for the
ℓ=6000 extension). Stage scripts are `src/stages/*.py`; shared decisions in
`src/stages/_decisions.py`. Everything scientific runs under the SPT CVMFS
environment (see the `spt3g-software` skill). Heavy runs go to OSG via the
snakemake htcondor9 profile (`~/.config/snakemake/htcondor9/`); `lc run
--executor htcondor` selects it.

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

## Report

`index.md` + `myst.yml` are a template MyST report wired to the MySTRA
plugin. The report references analysis elements by path — inline mentions
with the `{astra}` role, block embeds with the `{astra}` directive, live
numbers with `{astra:value}` — so never hard-type a measured value in the
prose. Preview with `myst start` (requires the MyST CLI, `npm i -g mystmd`).
