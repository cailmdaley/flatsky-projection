#!/bin/bash
# job_wrapper for the htcondor9 Snakemake executor profile (lightcone jobs).
#
# The plugin makes this script the job executable; $@ is the per-job
# snakemake argv (minus the "python -m snakemake" prefix). We activate the
# CVMFS py3-v5 toolset (snakemake 9.10.1 + healpy + pymaster on the worker),
# unpack the transferred lightcone payload so the generated Snakefile's
# `from lightcone.engine.runner import run_rule` resolves, and hand off.
# The spt3g build tarball is unpacked lazily by env.sh inside the recipe.
set -euo pipefail

export OS_ARCH=RHEL_9_x86_64
eval "$(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"

if [ -f lightcone_py.tar.gz ] && [ ! -d lcpy ]; then
    mkdir -p lcpy
    tar -xzf lightcone_py.tar.gz -C lcpy
fi
export PYTHONPATH=$PWD/lcpy${PYTHONPATH:+:$PYTHONPATH}

exec snakemake "$@"
