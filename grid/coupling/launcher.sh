#!/bin/bash
# Census-and-submit launcher for the highell coupling-matrix stage on OSG.
#
# Builds a joblist of every (projection, mask_index) pair whose w22/w00
# outputs are not already present in results/highell/coupling_matrices
# (grid or local -- same directory, same filenames, so this naturally skips
# whatever the concurrent local run has already finished), then submits one
# condor job per remaining pair.
#
# request_memory is scaled per mask index: the flat-sky mask footprint grows
# with index (radii 0.175 .. 0.65 rad -> cut-bounds widths 660..2700 px,
# ~16x in linear size from index 0 to 6, matching the mask npz size growing
# 3.5MB -> 58MB), so the coupling-matrix FFT working set grows with it too.
# Tiers below are sized off that npz-size proxy with generous headroom;
# refine after the probe job reports a true /usr/bin/time -v peak.
#
# Usage: grid/coupling/launcher.sh [batch_name]   (default: batch)
#        grid/coupling/launcher.sh probe          (writes lists/probe.txt with just ProjSIN 0)
#        EXCLUDE=$'ProjSIN,0' grid/coupling/launcher.sh   (newline-separated proj,idx pairs to
#                                                           skip even if not yet on disk -- e.g.
#                                                           a probe job still in flight)

set -u

REPO=/home/cailmdaley/work/pol_project
MASKS=/sptlocal/user/cailmdaley/pol_project/results/highell/flatsky_masks
RESULTS=/sptlocal/user/cailmdaley/pol_project/results/highell/coupling_matrices
LISTS=$REPO/grid/coupling/lists

PROJECTIONS="ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP"
INDICES="0 1 2 3 4 5 6"

mem_for_idx() {
    case "$1" in
        0|1|2) echo 4GB ;;
        3) echo 6GB ;;
        4) echo 8GB ;;
        5) echo 12GB ;;
        6) echo 20GB ;;
    esac
}

cd "$REPO" || exit 1
mkdir -p "$LISTS" grid/coupling/logs "$RESULTS"

BATCH=${1:-batch}
list=$LISTS/$BATCH.txt
: >"$list"

if [ "$BATCH" = "probe" ]; then
    echo "ProjSIN,0,4GB" >>"$list"
else
    for proj in $PROJECTIONS; do
        for idx in $INDICES; do
            if [ -n "${EXCLUDE:-}" ] && echo "$EXCLUDE" | grep -qx "$proj,$idx"; then
                echo "skip $proj $idx (excluded, e.g. an in-flight probe)" >&2
                continue
            fi
            w22="$RESULTS/w22_${proj}_${idx}.fits"
            w00="$RESULTS/w00_${proj}_${idx}.fits"
            if [ -s "$w22" ] && [ -s "$w00" ]; then
                echo "skip $proj $idx (outputs exist)" >&2
                continue
            fi
            echo "$proj,$idx,$(mem_for_idx "$idx")" >>"$list"
        done
    done
fi

n=$(wc -l <"$list")
echo "joblist $list: $n job(s)" >&2
if [ "$n" -eq 0 ]; then
    echo "nothing to submit" >&2
    exit 0
fi
cat "$list" >&2

condor_submit "$REPO/grid/coupling/coupling.sub" -a "joblist = $list"
