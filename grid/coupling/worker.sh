#!/bin/bash
# Production worker for the highell coupling-matrix stage on OSG.
#
# One job = one (projection, mask index) pair:
#   worker.sh <projection> <mask_index> [threads]
# e.g. worker.sh ProjSIN 0        (threads defaults to 8)
#      worker.sh ProjBICEP 6 16   (idx=6 rescue: 16-core, see coupling.sub)
#
# Unlike grid/worker.sh (the projection stage), this stage needs no spt3g at
# all -- coupling_matrices.py only imports numpy and pymaster, both already
# in the CVMFS py3-v5 toolset, so there is no build tarball to unpack and no
# PYTHONPATH/LD_LIBRARY_PATH surgery. The only transferred inputs are the
# stage script, _decisions.py, and the single mask npz for this pair (see
# grid/coupling/coupling.sub).
#
# coupling_matrices.py discovers which mask indices exist for a projection
# by globbing --masks for mask_{proj}_*.npz (falling back from its
# index.json-manifest path, which isn't present in the job sandbox since
# only one mask file is transferred). Pointing --masks at a directory
# containing exactly one npz therefore makes it process exactly this one
# (projection, index) pair and nothing else -- no code changes needed.
#
# Threading: OMP_NUM_THREADS / --max-procs come from $3, defaulting to 8.
# /usr/bin/time -v on the completed idx=4/5 windows showed ~700-734% CPU at
# 8 cores (87-92% of ideal 8x -- an Amdahl fit puts the serial fraction at
# ~2%), so doubling to 16 buys a genuine ~1.76x, not just contention. That's
# the idx=6 rescue's whole point: idx=5's worst case (ProjSFL) already took
# 18:58:15 wall at 8 cores, grazing the 20h Medium cap it was held under
# before this file added +JobDurationCategory=Long; idx=6 is bigger still,
# so 16 cores buys real margin under Long's 40h rather than 8 cores hoping
# the 40h ceiling is enough. Request_cpus in coupling.sub must match $3.

set -o pipefail

PROJ=$1
IDX=$2
THREADS=${3:-8}
OUTW22="w22_${PROJ}_${IDX}.fits"
OUTW00="w00_${PROJ}_${IDX}.fits"

say() { echo "### $1  t=$(date -u +%FT%TZ)"; }

say JOB
echo "PROJ=$PROJ  IDX=$IDX"
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || echo unknown)"
echo "CPUS=$(nproc)  MEMFREE=$(awk '/MemAvailable/{print $2/1048576" GiB"}' /proc/meminfo)"
df -h . | tail -1

say INPUTS
ls -la

say TOOLSET
export OS_ARCH=RHEL_9_x86_64
# setup.sh picks its output syntax from `case ${SHELL##*/}` and emits csh
# `setenv` lines for tcsh/csh. OSPool workers do not agree on $SHELL -- some
# hand out csh -- and the failure is `setenv: command not found` from a shell
# that was always bash. Pin it before the eval, not after.
export SHELL=/bin/bash
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
# setup.sh exports OMP_NUM_THREADS=1; the big windows (4-6) exceed OSPool's
# 20-hour execute-duration cap single-threaded, so thread them.
export OMP_NUM_THREADS=$THREADS
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

say RUN
mkdir -p masks out
mv "mask_${PROJ}_${IDX}.npz" "masks/"

/usr/bin/time -v python3 coupling_matrices.py \
    --masks masks \
    --projections "$PROJ" \
    --purify-b purify \
    --lmax l6000 --delta-ell 50 \
    --max-procs "$THREADS" \
    --out out
rc=$?
echo "run_rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

say OUTPUT
ls -la out/
[ -s "out/$OUTW22" ] || { echo "OUTPUT_OK NO ($OUTW22 missing)"; exit 1; }
[ -s "out/$OUTW00" ] || { echo "OUTPUT_OK NO ($OUTW00 missing)"; exit 1; }
# Hoist the two workspace files out of out/ so the submit file transfers
# exactly them and can remap onto results/highell/coupling_matrices/.
# out/index.json is deliberately left behind (single-pair manifest, not the
# real provenance record).
mv "out/$OUTW22" "./$OUTW22"
mv "out/$OUTW00" "./$OUTW00"
echo "OUTPUT_OK yes"
ls -la "./$OUTW22" "./$OUTW00"
md5sum "./$OUTW22" "./$OUTW00"

say DONE
