# Closed-form distortion of projected flat-sky CMB spectra

This repository holds the paper *Closed-form distortion of projected flat-sky CMB spectra* (Cail Daley, Caterina Umiltà and Gil Holder) together with every computation behind it.
Each figure, each number in the text, and the PDF itself are outputs of one declared pipeline, and you can rebuild them from scratch.

## What the paper says

A flat-sky analysis projects a patch of the sphere onto a plane.
Near every point the projection rotates, magnifies and shears the sky by amounts its Jacobian fixes in closed form, before anything is observed.
Polarization pipelines undo the rotation ("flattening").
The shear survives that step and acts like a lens the analyst chose: it averages each power spectrum over a band of multipoles, and it turns the E/B basis so that a fraction $2\langle g^2\rangle$ of EE appears as BB, where $\langle g^2\rangle$ is the mean squared shear under the squared mask.
The paper derives both effects as a coupling matrix computed from a mask and a theory spectrum, checks it on simulations of four projections (SFL, STG, ZEA, BICEP) over seven footprints, and concludes that the projection is best chosen rather than corrected.

## Layout

| Path | What it is |
|---|---|
| `astra.yaml` | The analysis specification (ASTRA format): its decisions, its outputs and the recipe that makes each one. There are no data inputs. Everything descends from the decisions, starting with the cosmology. |
| `universes/fiducial.yaml` | The decision values the paper uses. |
| `src/stages/` | The pipeline stages: CAMB spectra, Gaussian skies, projected maps, masks, NaMaster spectra. `src/stages/README.md` gives each stage's contract. |
| `analysis/` | Summaries, macro writers and figure scripts that read the stage outputs. |
| `paper/` | `main.tex` and its bibliography. Numbers enter the text only as macros written by the outputs. |
| `projcoupling/` | A small numpy-only package that computes a projection's coupling matrix from any grid's pixel coordinates and a mask (Sect. 5.2 of the paper). |
| `env.sh`, `wrapper.sh` | The launcher every recipe runs through, locally and on the Open Science Pool. |

## Rebuilding it

The pipeline is driven by `lightcone-cli` (`lc`), which turns `astra.yaml` into a Snakemake workflow and records provenance for every output.

```bash
lc status -u fiducial                                       # what is built, stale or missing
lc run paper -u fiducial                                    # build the letter and everything it needs
lc run --executor htcondor --profile htcondor9 bandpowers -u fiducial   # heavy stages on the Open Science Pool
lc verify -u fiducial                                       # recompute hashes along the provenance chain
```

Outputs land in `results/fiducial/<output_id>/`, and each one carries a `.lightcone-manifest.json` that records its recipe, decisions, input hashes and output hash.
The letter is `results/fiducial/paper/paper.pdf`.

The stages run inside the SPT-3G software environment (`spt3g_software` and the CVMFS `py3-v5` toolset, which provides NaMaster 2.0 and healpy).
`env.sh` activates it on the SPT analysis machines.
On a grid worker it unpacks a shipped build of `spt3g_software`, which is why `spt3g_build.tar.gz` and `lightcone_py.tar.gz` are symlinks into the SPT data disk here.
Outside SPT you need an equivalent environment: the projections come from `spt3g.maps`, and everything else is standard Python (numpy, healpy, pymaster, camb).
The heavy stages (coupling matrices, bandpowers, the covariance suites) fan out into one grid job per projection, mask or seed.

## Browsing it

Lightcone Studio is a browser view of this specification. It lays out the decisions and the output graph, shows each figure and macro table with the manifest behind it, and connects the paper's claims to the outputs that support them.
On the SPT analysis machine `amundsen` it runs from this checkout at `http://127.0.0.1:5317/?universe=fiducial`.

## History

The project began in 2020 as a NaMaster study by Cail Daley and Caterina Umiltà and was revived in 2026.
The history keeps the dead ends: appendices, sub-analyses and branches that did not make it into the letter.
Two branches that never merged are kept as the tags `archive/lightcone-gfal` and `archive/mainline-osg-proof`.
