#!/bin/bash
# Census: what exists, what's in flight, what failed, what it cost.
#   grid/check.sh [stage ...]     (default: every stage submit.sh knows)
#
# The per-stage done/flight/missing counts come from grid/submit.sh -n, so
# there is exactly one definition of "which units exist" in the apparatus.
# "done" there means present on dCache *or* /sptlocal; the line below each
# census says how many are still only on dCache and want a grid/pull.sh.
set -u
REPO=/home/cailmdaley/work/pol_project
LOGS=$REPO/grid/logs
cd "$REPO" || exit 1

for stage in "${@:-project spectra}"; do
    echo "== $stage =="
    grid/submit.sh "$stage" -n 2>&1 >/dev/null | grep -v '^skip \|^in flight: '
    grid/pull.sh "$stage" -n 2>&1 >/dev/null | grep 'to pull'
done

echo
echo "== queue =="
condor_q -af:h ClusterId ProcId JobStatus PolUnit RequestMemory MemoryUsage NumJobStarts 2>&1
# JobStatus 5 = held. With the hold/release policy in grid/common.sub a job
# lands here only after 6 failed starts, so a hold is a real failure to read.
condor_q -held -af:h ClusterId PolUnit HoldReason 2>&1 | grep -v '^$'

echo
echo "== failures in logs =="
grep -l 'run_rc=[^0]' "$LOGS"/*.out 2>/dev/null | while read -r f; do
    printf '%-46s %s\n' "$(basename "$f")" "$(grep -o 'run_rc=[0-9]*' "$f" | tail -1)"
done
grep -h 'OUTPUT_OK NO\|UNPACK_OK NO\|TOOLSET_OK NO\|INDEX_OK NO' "$LOGS"/*.out 2>/dev/null | sort | uniq -c

echo
# /usr/bin/time -v writes to stderr, so peaks land in .err. Condor's sampled
# MemoryUsage reads about half the truth -- size requests from this column.
echo "== true peak memory (Maximum resident set size, GiB) =="
printf '%-46s %10s %10s\n' UNIT PEAK_GIB WALL
for f in "$LOGS"/*.err; do
    [ -e "$f" ] || continue
    kb=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "$f" | tail -1 | grep -o '[0-9]*$')
    [ -n "$kb" ] || continue
    wall=$(grep -o 'Elapsed (wall clock) time.*: .*' "$f" | tail -1 | sed 's/.*: //')
    printf '%-46s %10.1f %10s\n' "$(basename "$f" .err)" \
           "$(awk -v k="$kb" 'BEGIN{print k/1048576}')" "$wall"
done | sort -k2 -n
