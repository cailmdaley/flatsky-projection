#!/bin/bash
# job_wrapper for the htcondor9 Snakemake executor profile.
#
# The htcondor executor plugin submits `sys.executable` (the python on the
# submit host) as the job executable by default -- but OSG workers have no
# shared filesystem, so that python does not exist on the worker. Setting
# `job_wrapper` in the profile makes THIS script the executable instead; the
# plugin strips the "python -m snakemake" prefix from the per-job command and
# passes the rest as arguments here ($@). We activate the validated CVMFS
# py3-v5 toolset (which provides snakemake 9.10.1 + pymaster 1.5.1 on the
# worker) and then hand off to snakemake.
set -euo pipefail

export OS_ARCH=RHEL_9_x86_64
eval "$(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"

exec snakemake "$@"
