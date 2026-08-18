#!/bin/bash
# Production worker for the highell projection stage on OSG.
#
# Graduated from osg_test/probe9.sh. Runs ONE (leg, projection, seed) job:
#   worker.sh <stage_script> <projection> <seed>
# e.g. worker.sh project_eb.py ProjSFL 0
#
# Environment: rocky:9 singularity + SPT CVMFS py3-v5 toolset + the
# self-contained spt3g build tarball transferred in alongside the stage
# sources. The seed's alms file is transferred into the scratch dir, so the
# stage runs with --alms . -- see grid/project.sub.
#
# The heavy step is the Healpix synthesis at nside 8192 (three/two float64
# maps of ~6.4 GB each plus healpy scratch). Validated true peak for the QU
# leg is 38.8 GB via /usr/bin/time -v; request_memory is 64GB.

set -o pipefail

STAGE=$1
PROJ=$2
SEED=$3
SEEDPAD=$(printf '%04d' "$SEED")
LEG=${STAGE#project_}; LEG=${LEG%.py}
OUTNAME="${LEG}_${PROJ}_seed${SEEDPAD}.g3"

say() { echo "### $1  t=$(date -u +%FT%TZ)"; }

say JOB
echo "STAGE=$STAGE  LEG=$LEG  PROJ=$PROJ  SEED=$SEED  OUTNAME=$OUTNAME"
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || echo unknown)"
echo "CPUS=$(nproc)  MEMFREE=$(awk '/MemAvailable/{print $2/1048576" GiB"}' /proc/meminfo)"
df -h . | tail -1

say INPUTS
ls -la

say UNPACK
tar -xzf spt3g_build.tar.gz || { echo "UNPACK_OK NO"; exit 1; }
echo "UNPACK_OK yes  dangling_symlinks=$(find build -xtype l | wc -l)"

say TOOLSET
export OS_ARCH=RHEL_9_x86_64
# setup.sh picks its output syntax from `case ${SHELL##*/}` and emits csh
# `setenv` lines for tcsh/csh. OSPool workers do not agree on $SHELL -- some
# hand out csh -- and the failure is `setenv: command not found` from a shell
# that was always bash. Pin it before the eval, not after.
export SHELL=/bin/bash
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

export SPT3G_SOFTWARE_BUILD_PATH=$PWD/build
export PYTHONPATH=$PWD/build:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
export PATH=$PWD/build/bin:$PATH

say RUN
mkdir -p out
# highell universe decisions: nside 8192, interp_rebin8, 2 arcmin, 110 deg,
# ra 0, dec -56 (lmax l6000 is baked into the transferred alms).
/usr/bin/time -v python3 "$STAGE" \
    --alms . \
    --projections "$PROJ" \
    --interpolation interp_rebin8 \
    --nside nside8192 \
    --res-arcmin 2.0 \
    --size-deg 110.0 \
    --ra-deg 0.0 \
    --dec-deg -56.0 \
    --seed-start "$SEED" --seed-stop "$((SEED + 1))" \
    --max-procs 4 \
    --out out
rc=$?
echo "run_rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

say OUTPUT
ls -la out/
[ -s "out/$OUTNAME" ] || { echo "OUTPUT_OK NO"; exit 1; }
# Hoist the single map out of out/ so the submit file transfers exactly one
# file and can remap it into results/highell/projected_<leg>/. out/index.json
# is deliberately left behind: every job would write its own single-entry
# manifest to the same name, and the stage's real provenance lives in the
# lightcone manifest for the assembled output directory.
mv "out/$OUTNAME" "./$OUTNAME"
echo "OUTPUT_OK yes"
ls -la "./$OUTNAME"
md5sum "./$OUTNAME"

say DONE
