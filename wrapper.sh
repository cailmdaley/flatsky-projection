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
# setup.sh picks its output syntax from `case ${SHELL##*/}` and emits csh
# `setenv` lines for tcsh/csh. OSPool workers do not agree on $SHELL -- some
# hand out csh -- and the failure is `setenv: command not found` from a shell
# that was always bash. Pin it before the eval, not after.
export SHELL=/bin/bash
eval "$(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"

# Some OSPool sites (seen: NWICG_NDCMS, msu-lm01.osris.org) run the container
# so that the sandbox's absolute path is not usable from Python: snakemake's
# os.makedirs(os.path.abspath(".snakemake/...")) dies on "Read-only file
# system: '/scratch'" while relative paths work, and bash's own `[ -d
# "$(pwd -P)" ]` does not catch it. So probe with the same interpreter and
# the same call snakemake makes. If it fails and the sandbox is also visible
# at a resolvable alias (OSG's /srv convention, or the condor scratch
# variable), work from there; otherwise fail fast with the declared output
# directories created, so condor does not hold the job for a missing output
# transfer (a 4 h timeout on the access point) and snakemake's retries
# resubmit it, most likely to a different site.
# Some sites hand the job a sandbox whose absolute path is a symlink into a
# host path the container cannot traverse (seen on MI-HORUS, CHTC, Colgate:
# cwd /srv/scratch works, relative paths work, but anything that resolves
# the symlink -- snakemake's Persistence does -- dies on "Read-only file
# system: '/var/lib/condor/execute/...'"). Detect it by comparing the cwd
# with its realpath. If they differ, do the work in a fresh real directory
# and copy the declared outputs back into the sandbox for transfer.
SANDBOX=$PWD
REAL=$(python3 -c 'import os; print(os.path.realpath("."))')
echo "wrapper.sh: cwd=$PWD realpath=$REAL _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-} TMPDIR=${TMPDIR:-} APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-}" >&2
RELOCATED=0
if [ "$REAL" != "$PWD" ]; then
    WORK=""
    for base in /tmp /var/tmp "$HOME"; do
        [ -d "$base" ] && [ -w "$base" ] || continue
        cand=$(mktemp -d "$base/lcwork.XXXXXX" 2>/dev/null) || continue
        if [ "$(python3 -c "import os; print(os.path.realpath('$cand'))")" = "$cand" ]; then WORK=$cand; break; fi
        rmdir "$cand"
    done
    if [ -z "$WORK" ]; then
        echo "wrapper.sh: sandbox $PWD resolves to $REAL and no real working directory is available; failing fast" >&2
        if [ -n "${_CONDOR_JOB_AD:-}" ]; then
            tr -d '"' < "$_CONDOR_JOB_AD" | awk -F' = ' '$1=="TransferOutput"{print $2}' \
                | tr ',' '\n' | sed 's/^ *//; s/ *$//' | while read -r out; do
                [ -n "$out" ] && mkdir -p "$out" || true
            done
        fi
        exit 85
    fi
    echo "wrapper.sh: relocating sandbox to $WORK" >&2
    cp -a "$SANDBOX"/. "$WORK"/
    cd "$WORK"
    export TMPDIR="$WORK/.tmp"; mkdir -p "$TMPDIR"
    export PYTHONPATH=$PWD/lcpy${PYTHONPATH:+:$PYTHONPATH}
    RELOCATED=1
fi

# Snakemake's Persistence makes its apptainer-image directory at
# $APPTAINER_CACHEDIR when that is set (DeploymentSettings.apptainer_prefix),
# and some glideins (AMNH, IRISHEP-SSL river, Rhodes-HPC; likely MWT2) export
# it as the sandbox's *host* path, which is read-only inside the container:
# the job dies on "Read-only file system: '/pilot'" or '/scratch_space' with
# cwd, realpath and TMPDIR all /srv/scratch. Nothing here uses apptainer, so
# point it inside the sandbox.
export APPTAINER_CACHEDIR="$PWD/.apptainer"

if [ -f lightcone_py.tar.gz ] && [ ! -d lcpy ]; then
    mkdir -p lcpy
    tar -xzf lightcone_py.tar.gz -C lcpy
fi
export PYTHONPATH=$PWD/lcpy${PYTHONPATH:+:$PYTHONPATH}

# Diagnostic: some sites die in snakemake's Persistence with "Read-only file
# system: '/scratch'" although cwd, TMPDIR and relative paths all work. Log
# the exact path os.makedirs fails on, and the cwd at that moment, so the
# .err says which path snakemake resolved and from where.
mkdir -p lcpy
cat > lcpy/sitecustomize.py <<'PY'
import os, sys
_makedirs = os.makedirs
def makedirs(name, mode=0o777, exist_ok=False):
    try:
        return _makedirs(name, mode, exist_ok)
    except OSError as e:
        print(f"sitecustomize: os.makedirs({name!r}) failed: {e}; cwd={os.getcwd()} "
              f"TMPDIR={os.environ.get('TMPDIR')} exists(cwd)={os.path.exists(os.getcwd())}",
              file=sys.stderr, flush=True)
        raise
os.makedirs = makedirs
PY

# Staged upstream outputs. A remote job cannot be handed `results/...`
# directly: `results` is a symlink to /sptlocal and condor refuses to transfer
# symlinks to directories. lightcone therefore ships each upstream output as
# a gzipped archive named `<universe>__<output_id>.tar.gz`, whose members are
# already rooted at `results/<universe>/<output_id>/`. Unpacking here in the
# job sandbox recreates exactly the relative path the recipe expects, so the
# stage scripts need no notion of being on a worker.
#
# The `__` in the glob is load-bearing: it distinguishes these from the
# project's own payload tarballs (spt3g_build.tar.gz, lightcone_py.tar.gz),
# which are unpacked elsewhere and must not be touched here.
#
# Look in both places: condor's PreserveRelativePaths recreates a relative
# input at its relative location, so an archive declared as
# `.lightcone/stage/<...>` lands there rather than at the sandbox root. An
# absolute one would land at the root by basename. Extraction happens with
# the sandbox as CWD either way, which is what puts the members at
# `results/<universe>/<output_id>/`.
shopt -s nullglob
for staged in *__*.tar.gz .lightcone/stage/*__*.tar.gz; do
    tar -xzf "$staged"
done
shopt -u nullglob

if [ "$RELOCATED" = 1 ]; then
    set +e; snakemake "$@"; status=$?; set -e
    tr -d '"' < "$_CONDOR_JOB_AD" | awk -F' = ' '$1=="TransferOutput"{print $2}' \
        | tr ',' '\n' | sed 's/^ *//; s/ *$//' | while read -r out; do
        [ -n "$out" ] || continue
        mkdir -p "$SANDBOX/$(dirname "$out")"
        rm -rf "$SANDBOX/$out"
        [ -e "$WORK/$out" ] && cp -a "$WORK/$out" "$SANDBOX/$out" || mkdir -p "$SANDBOX/$out"
    done
    echo "wrapper.sh: copied outputs back to $SANDBOX; snakemake exit $status" >&2
    exit $status
fi
exec snakemake "$@"
