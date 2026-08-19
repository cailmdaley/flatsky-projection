#!/bin/bash
# Worker for the projection stage: one (leg, projection, seed).
#   project.sh <stage_script> <projection> <seed>
# The heavy step is the nside-8192 Healpix synthesis; see the memory note in
# grid/submit.sh. Decisions below are the highell universe's.
set -o pipefail
STAGE=$1 PROJ=$2 SEED=$3
SEEDPAD=$(printf '%04d' "$SEED")
LEG=${STAGE#project_}; LEG=${LEG%.py}
OUTNAME="${LEG}_${PROJ}_seed${SEEDPAD}.g3"

source ./env.sh

say RUN
mkdir -p out
/usr/bin/time -v python3 "$STAGE" \
    --alms . --projections "$PROJ" \
    --interpolation interp_rebin8 --nside nside8192 \
    --res-arcmin 2.0 --size-deg 110.0 --ra-deg 0.0 --dec-deg -56.0 \
    --seed-start "$SEED" --seed-stop "$((SEED + 1))" \
    --max-procs 4 --out out
rc=$?
echo "run_rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

say OUTPUT
# Hoist the single map out of out/ so condor transfers exactly the one file
# common.sub remaps. out/index.json is left behind: every job would write its
# own single-entry manifest to the same name, and the real provenance lives in
# the lightcone manifest for the assembled output directory.
[ -s "out/$OUTNAME" ] || { echo "OUTPUT_OK NO"; exit 1; }
mv "out/$OUTNAME" "./$OUTNAME"
push_output "$OUTNAME" || exit 1
say DONE
