#!/bin/bash
# Local (access-point) runner for the idx=6 coupling-matrix stragglers that
# never converge on OSPool within the 20h execute-duration cap even with
# 8-core threading (see coupling rescue session 2026-08-21). Same env/args
# as grid/coupling/worker.sh, minus the condor wrapper: writes straight to
# the final results dir instead of hoisting files for transfer, since
# coupling_matrices.py is already resumable/atomic (temp-file + rename) and
# skips any w22/w00 that already exists.
#
# Usage: local_run.sh <projection>   e.g. local_run.sh ProjBICEP

set -o pipefail

PROJ=$1
REPO=/home/cailmdaley/work/pol_project
MASKS=/sptlocal/user/cailmdaley/pol_project/results/highell/flatsky_masks
OUT=/sptlocal/user/cailmdaley/pol_project/results/highell/coupling_matrices

say() { echo "### $1  t=$(date -u +%FT%TZ)"; }

say JOB
echo "PROJ=$PROJ (idx=6 only -- 0-5 already on disk, script skips existing)"
echo "HOST=$(hostname)"

say TOOLSET
export OS_ARCH=RHEL_9_x86_64
export SHELL=/bin/bash
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
export OMP_NUM_THREADS=8
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

say RUN
cd "$REPO/src/stages" || exit 1
/usr/bin/time -v python3 coupling_matrices.py \
    --masks "$MASKS" \
    --projections "$PROJ" \
    --purify-b purify \
    --lmax l6000 --delta-ell 50 \
    --max-procs 8 \
    --out "$OUT"
rc=$?
echo "run_rc=$rc"

say OUTPUT
ls -la "$OUT/w22_${PROJ}_6.fits" "$OUT/w00_${PROJ}_6.fits" 2>&1
[ "$rc" -eq 0 ] && [ -s "$OUT/w22_${PROJ}_6.fits" ] && [ -s "$OUT/w00_${PROJ}_6.fits" ] \
    && echo "OUTPUT_OK yes" || echo "OUTPUT_OK NO"

say DONE
exit "$rc"
