#!/bin/bash
# Post-hoc verifier for the highell projection run on OSG.
#
# Counts expected vs present outputs per leg/projection, reports non-zero
# exits found in the job logs, and reports the TRUE peak memory recorded by
# /usr/bin/time -v (condor's sampled MemoryUsage reads low -- never size from
# it).
#
# Usage: grid/check.sh [seed_stop]   (default 20 seeds)

set -u

REPO=/home/cailmdaley/work/pol_project
RESULTS=/sptlocal/user/cailmdaley/pol_project/results/highell
LOGS=$REPO/grid/logs
NSEEDS=${1:-20}
PROJECTIONS="ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP"

echo "== outputs (expected $NSEEDS seeds per leg/projection) =="
printf '%-6s %-10s %8s %8s   %s\n' LEG PROJ PRESENT MISSING "missing seeds"
total_present=0
total_expected=0
for leg in qu eb; do
    for proj in $PROJECTIONS; do
        present=0; missing=""
        for seed in $(seq 0 $((NSEEDS - 1))); do
            pad=$(printf '%04d' "$seed")
            if [ -s "$RESULTS/projected_$leg/${leg}_${proj}_seed$pad.g3" ]; then
                present=$((present + 1))
            else
                missing="$missing $pad"
            fi
        done
        printf '%-6s %-10s %8d %8d  %s\n' "$leg" "$proj" "$present" \
               $((NSEEDS - present)) "$missing"
        total_present=$((total_present + present))
        total_expected=$((total_expected + NSEEDS))
    done
done
echo "TOTAL $total_present / $total_expected"

echo
echo "== non-zero exits in logs =="
grep -l 'run_rc=[^0]' "$LOGS"/*.out 2>/dev/null | while read -r f; do
    printf '%-40s %s\n' "$(basename "$f")" "$(grep -o 'run_rc=[0-9]*' "$f" | tail -1)"
done
grep -h 'OUTPUT_OK NO\|UNPACK_OK NO\|TOOLSET_OK NO' "$LOGS"/*.out 2>/dev/null | sort | uniq -c

echo
# /usr/bin/time -v writes to stderr, so the peak lands in the .err log while
# run_rc and the output check land in the .out log.
echo "== true peak memory (Maximum resident set size, GiB) =="
printf '%-46s %10s %10s\n' JOB PEAK_GIB WALL
for f in "$LOGS"/*.err; do
    [ -e "$f" ] || continue
    kb=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "$f" | tail -1 | grep -o '[0-9]*$')
    wall=$(grep -o 'Elapsed (wall clock) time.*: .*' "$f" | tail -1 | sed 's/.*: //')
    [ -n "$kb" ] || continue
    printf '%-46s %10.1f %10s\n' "$(basename "$f" .err)" \
           "$(awk -v k="$kb" 'BEGIN{print k/1048576}')" "$wall"
done | sort -k2 -n

echo
echo "== queue =="
condor_q -totals 2>&1 | grep cailmdaley
