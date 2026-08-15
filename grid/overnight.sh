#!/bin/bash
# Overnight chain for the highell universe's amundsen-side stages.
#
# Waits for the l6000 preprocessing run to release the memory cgroup, then
# runs masks -> flatsky masks -> coupling matrices; waits for the 200 grid
# projection outputs; then runs the NaMaster bandpowers. Every stage is
# skipped if its index.json already exists, so re-running this script after
# any failure is a safe sweep (coupling_matrices and namaster_spectra are
# additionally resumable file-by-file internally).
#
# Launch detached:  setsid nohup bash grid/overnight.sh > grid/overnight.log 2>&1 &
set -u

ENV=/home/cailmdaley/code/python/spt3g_software/build/env-shell.sh
REPO=/home/cailmdaley/work/pol_project
RESULTS=/sptlocal/user/cailmdaley/pol_project/results/highell
PREPROC_PID=${PREPROC_PID:-2454648}
ALMS=/sptlocal/user/cailmdaley/pol_project/preprocessed_alms_l6000

say() { echo "[$(date -u +%FT%TZ)] $*"; }

cd "$REPO"

# --- 1. wait for the preprocessing run to free the cgroup -------------------
while kill -0 "$PREPROC_PID" 2>/dev/null; do
    say "waiting on preprocessing (pid $PREPROC_PID)"
    sleep 300
done
say "preprocessing exited"

# sanity: all 20 alms present and full-size
nalms=$(find "$ALMS" -name 'alms_seed*.fits' -size +1000M 2>/dev/null | wc -l)
if [ "$nalms" -lt 20 ]; then
    say "FATAL: only $nalms/20 full-size alms in $ALMS -- preprocessing incomplete"
    exit 1
fi
say "all 20 alms present"

run_stage() {  # run_stage <name> <outdir> <cmd...>
    local name=$1 outdir=$2; shift 2
    if [ -f "$outdir/index.json" ]; then
        say "$name: skip (index.json exists)"
        return 0
    fi
    say "$name: start"
    if "$@" >> "grid/overnight_$name.log" 2>&1; then
        say "$name: done"
    else
        say "FATAL: $name failed (rc=$?), see grid/overnight_$name.log and dmesg for oom"
        exit 1
    fi
}

# --- 2. healpix masks (nside 8192, sphere apodization) ----------------------
run_stage healpix_masks "$RESULTS/healpix_masks" \
    $ENV python src/stages/make_masks.py \
    --nside nside8192 --apodization c2_2p5deg \
    --ra-deg 0 --dec-deg -56 \
    --radii-rad 0.175 0.2 0.225 0.25 0.35 0.45 0.65 \
    --max-procs 4 --out "$RESULTS/healpix_masks"

# --- 3. flat-sky masks ------------------------------------------------------
run_stage flatsky_masks "$RESULTS/flatsky_masks" \
    $ENV python src/stages/project_masks.py \
    --healpix-masks "$RESULTS/healpix_masks" \
    --projections ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP \
    --interpolation interp_rebin8 \
    --res-arcmin 2 --size-deg 110 --ra-deg 0 --dec-deg -56 \
    --cut-bounds 1320:1980 1290:2010 1230:2070 1140:2160 960:2340 780:2520 300:3000 \
    --max-procs 2 --out "$RESULTS/flatsky_masks"

# --- 4. coupling matrices (lmax 6000, the expensive one) --------------------
run_stage coupling_matrices "$RESULTS/coupling_matrices" \
    $ENV python src/stages/coupling_matrices.py \
    --masks "$RESULTS/flatsky_masks" \
    --projections ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP \
    --purify-b purify \
    --lmax l6000 --delta-ell 10 \
    --max-procs 4 --out "$RESULTS/coupling_matrices"

# --- 5. wait for the grid projection outputs (200 = 2 legs x 5 proj x 20) ---
while :; do
    ng=$(find "$RESULTS/projected_qu" "$RESULTS/projected_eb" -name '*.g3' -size +50M 2>/dev/null | wc -l)
    [ "$ng" -ge 200 ] && break
    say "waiting on grid projections ($ng/200)"
    sleep 600
done
say "grid projections complete ($ng/200)"

# --- 6. NaMaster bandpowers -------------------------------------------------
run_stage bandpowers "$RESULTS/bandpowers" \
    $ENV python src/stages/namaster_spectra.py \
    --qu-maps "$RESULTS/projected_qu" \
    --eb-maps "$RESULTS/projected_eb" \
    --masks "$RESULTS/flatsky_masks" \
    --workspaces "$RESULTS/coupling_matrices" \
    --projections ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP \
    --seed-start 0 --seed-stop 20 \
    --purify-b purify --pixel-window healpix_and_flat \
    --lmax l6000 --delta-ell 10 \
    --max-procs 4 --out "$RESULTS/bandpowers"

say "overnight chain complete"
