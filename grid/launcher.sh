#!/bin/bash
# Unattended launcher for the highell projection stage.
#
# The local preprocessing run drops one alms_seed%04d.fits about every 31
# minutes. This script waits for each seed's file to appear AND be complete,
# then submits that seed's 10 jobs (5 projections x 2 legs) to OSG. It exits
# once all seeds are submitted.
#
# Completeness test: every seed's alms file has an identical shape, so a file
# is complete exactly when its size equals the reference size below (taken
# from the already-finished seed 0). Poll interval is 60 s.
#
# Idempotent: a (leg, projection, seed) whose .g3 already exists in
# results/highell/projected_<leg>/ is not submitted again, so the launcher can
# be killed and restarted safely.
#
# Usage:  grid/launcher.sh [seed_start] [seed_stop]     (default 0 20)
# Detached:  setsid nohup grid/launcher.sh </dev/null >/dev/null 2>&1 &

set -u

REPO=/home/cailmdaley/work/pol_project
ALMS=/sptlocal/user/cailmdaley/pol_project/preprocessed_alms_l6000
RESULTS=/sptlocal/user/cailmdaley/pol_project/results/highell
LOG=$REPO/grid/launcher.log
LISTS=$REPO/grid/lists

SEED_START=${1:-0}
SEED_STOP=${2:-20}
POLL=60
PROJECTIONS="ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP"
STAGES="project_qu.py project_eb.py"

# Per-leg request_memory. The OSpool is memory-poor (median slot 4 GB; only
# ~56 slots offer 64 GB against ~89 at 32 GB), so each leg asks for a little
# over its measured true peak instead of one worst-case number. Peaks come
# from /usr/bin/time -v on the grid, never from condor's sampled MemoryUsage,
# which reads low. QU: 38.8 GB measured (3 nside-8192 float64 maps from a
# polarized alm2map). EB: 24.7 GiB measured (two spin-0 alm2map calls), so
# 32GB gives ~30% headroom. Condor's own sample for that job read 12.7 GB --
# half the truth, which is why the sampled number is never used here.
MEM_qu=${MEM_QU:-64GB}
MEM_eb=${MEM_EB:-32GB}

# Reference size of a complete alms file, in bytes (seed 0, lmax 6000).
REF_SIZE=$(stat -c %s "$ALMS/alms_seed0000.fits" 2>/dev/null || echo 1080555840)

cd "$REPO" || exit 1
mkdir -p "$LISTS" grid/logs "$RESULTS/projected_qu" "$RESULTS/projected_eb"

say() { echo "[$(date -u +%FT%TZ)] $*" >>"$LOG"; }

say "launcher start pid=$$ seeds ${SEED_START}..$((SEED_STOP - 1)) ref_size=$REF_SIZE"

for seed in $(seq "$SEED_START" $((SEED_STOP - 1))); do
    pad=$(printf '%04d' "$seed")
    f=$ALMS/alms_seed$pad.fits

    waited=0
    while :; do
        size=$(stat -c %s "$f" 2>/dev/null || echo 0)
        [ "$size" -eq "$REF_SIZE" ] && break
        [ "$waited" -eq 0 ] && say "seed $pad: waiting (size=$size / $REF_SIZE)"
        waited=$((waited + POLL))
        sleep "$POLL"
    done
    say "seed $pad: alms complete after ${waited}s of waiting"

    list=$LISTS/seed$pad.txt
    : >"$list"
    for stage in $STAGES; do
        leg=${stage#project_}; leg=${leg%.py}
        for proj in $PROJECTIONS; do
            out=$RESULTS/projected_$leg/${leg}_${proj}_seed$pad.g3
            if [ -s "$out" ]; then
                say "seed $pad: skip $leg/$proj (output exists)"
            else
                mem=MEM_$leg
                echo "$stage,$leg,$proj,$seed,$pad,${!mem}" >>"$list"
            fi
        done
    done

    n=$(wc -l <"$list")
    if [ "$n" -eq 0 ]; then
        say "seed $pad: nothing to submit"
        continue
    fi
    if out=$(condor_submit "$REPO/grid/project.sub" -a "joblist = $list" 2>&1); then
        say "seed $pad: submitted $n jobs -- $(echo "$out" | tail -1)"
    else
        say "seed $pad: SUBMIT FAILED -- $out"
    fi
done

say "launcher done: all seeds through $((SEED_STOP - 1)) submitted"
