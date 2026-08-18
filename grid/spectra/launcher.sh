#!/bin/bash
# Census-and-submit launcher for the highell namaster_spectra stage on OSG.
#
# Builds a joblist of every (projection, seed) pair whose bandpowers output
# is not already present in results/highell/namaster_spectra (grid or
# local -- same directory, same filename, so this naturally skips whatever a
# concurrent local run has already finished), then submits one condor job
# per remaining pair. Each job internally spans all 7 mask indices and both
# legs for its one seed -- see grid/spectra/worker.sh.
#
# request_memory is a single flat tier, unlike grid/coupling's per-index
# tiering: every namaster_spectra job processes every mask index 0..6 in
# sequence (the stage script gives no way to select a mask-index subset), so
# every job's peak is set by the same largest index (6) regardless of which
# (projection, seed) it is -- there's no per-job axis left to tier on. The
# tier below is sized generously off two knowns: coupling_matrices' own
# idx-6 workspace-build tier (20GB, for building the same-size w22/w00 this
# stage only *reads*) and the ~500MB of decompressed 5500x5500 float64 Q/U
# or E/B arrays per seed. Refine after the probe job reports a true
# /usr/bin/time -v peak -- it should come in well under 20GB since this
# stage does no workspace *construction*, only decoupling.
#
# Usage: grid/spectra/launcher.sh [batch_name]   (default: batch)
#        grid/spectra/launcher.sh probe          (writes lists/probe.txt with just ProjSIN seed 0)
#        EXCLUDE=$'ProjSIN,0' grid/spectra/launcher.sh   (newline-separated proj,seed pairs to
#                                                          skip even if not yet on disk -- e.g.
#                                                          a probe job still in flight)

set -u

REPO=/home/cailmdaley/work/pol_project
MASKS=/sptlocal/user/cailmdaley/pol_project/results/highell/flatsky_masks
WORKSPACES=/sptlocal/user/cailmdaley/pol_project/results/highell/coupling_matrices
QU=/sptlocal/user/cailmdaley/pol_project/results/highell/projected_qu
EB=/sptlocal/user/cailmdaley/pol_project/results/highell/projected_eb
RESULTS=/sptlocal/user/cailmdaley/pol_project/results/highell/namaster_spectra
LISTS=$REPO/grid/spectra/lists

PROJECTIONS="ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP"
SEEDS="$(seq 0 19)"
INDICES="0 1 2 3 4 5 6"
MEM=16GB

cd "$REPO" || exit 1
mkdir -p "$LISTS" grid/spectra/logs "$RESULTS"

BATCH=${1:-batch}
list=$LISTS/$BATCH.txt
: >"$list"

check_inputs_ready() {
    # All 7 mask npz + w22/w00 fits for this projection, plus this seed's
    # qu/eb g3 pair, must exist before the job can be submitted.
    local proj=$1 pad=$2 i
    for i in $INDICES; do
        [ -s "$MASKS/mask_${proj}_${i}.npz" ] || return 1
        [ -s "$WORKSPACES/w22_${proj}_${i}.fits" ] || return 1
        [ -s "$WORKSPACES/w00_${proj}_${i}.fits" ] || return 1
    done
    [ -s "$QU/qu_${proj}_seed${pad}.g3" ] || return 1
    [ -s "$EB/eb_${proj}_seed${pad}.g3" ] || return 1
    return 0
}

if [ "$BATCH" = "probe" ]; then
    if ! check_inputs_ready ProjSIN 0000; then
        echo "probe: inputs for ProjSIN seed 0 not yet complete -- see check_inputs_ready" >&2
        exit 1
    fi
    echo "ProjSIN,0,0000,$MEM" >>"$list"
else
    for proj in $PROJECTIONS; do
        for seed in $SEEDS; do
            pad=$(printf '%04d' "$seed")
            if [ -n "${EXCLUDE:-}" ] && echo "$EXCLUDE" | grep -qx "$proj,$seed"; then
                echo "skip $proj $seed (excluded, e.g. an in-flight probe)" >&2
                continue
            fi
            out="$RESULTS/bandpowers_${proj}_seed${pad}.npz"
            if [ -s "$out" ]; then
                echo "skip $proj $seed (output exists)" >&2
                continue
            fi
            if ! check_inputs_ready "$proj" "$pad"; then
                echo "skip $proj $seed (inputs not yet complete)" >&2
                continue
            fi
            echo "$proj,$seed,$pad,$MEM" >>"$list"
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

condor_submit "$REPO/grid/spectra/spectra.sub" -a "joblist = $list"
