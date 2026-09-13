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
probe_abspath() {
    python3 -c 'import os, sys
d = os.path.abspath(".snakemake/probe")
try:
    os.makedirs(d, exist_ok=True)
except OSError as e:
    print(f"wrapper.sh: abspath probe failed: cwd={os.getcwd()} exists={os.path.exists(os.getcwd())} err={e}", file=sys.stderr)
    sys.exit(1)'
}
if ! probe_abspath; then
    moved=0
    for cand in /srv "${_CONDOR_SCRATCH_DIR:-}" "${TMPDIR:-}"; do
        if [ -n "$cand" ] && [ -d "$cand" ] \
           && [ "$(stat -Lc %d:%i "$cand")" = "$(stat -Lc %d:%i .)" ]; then
            cd "$cand"
            if probe_abspath; then moved=1; break; fi
        fi
    done
    if [ "$moved" = 0 ]; then
        echo "wrapper.sh: sandbox $(pwd -P) unusable in this container and no alias found; failing fast" >&2
        if [ -n "${_CONDOR_JOB_AD:-}" ]; then
            tr -d '"' < "$_CONDOR_JOB_AD" | awk -F' = ' '$1=="TransferOutput"{print $2}' \
                | tr ',' '\n' | sed 's/^ *//; s/ *$//' | while read -r out; do
                [ -n "$out" ] && mkdir -p "$out" || true
            done
        fi
        exit 85
    fi
    echo "wrapper.sh: working from $(pwd -P) (sandbox alias)" >&2
fi

if [ -f lightcone_py.tar.gz ] && [ ! -d lcpy ]; then
    mkdir -p lcpy
    tar -xzf lightcone_py.tar.gz -C lcpy
fi
export PYTHONPATH=$PWD/lcpy${PYTHONPATH:+:$PYTHONPATH}

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

exec snakemake "$@"
