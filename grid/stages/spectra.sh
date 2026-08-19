#!/bin/bash
# Worker for the NaMaster bandpowers stage: one (projection, seed) over all 7
# mask indices and both legs.  spectra.sh <projection> <seed>
#
# Splitting by mask index instead would re-transfer the same seed's ~350 MB of
# qu/eb maps once per index for no shrink in per-job memory.
set -o pipefail
PROJ=$1 SEED=$2
SEEDPAD=$(printf '%04d' "$SEED")
OUTNAME="bandpowers_${PROJ}_seed${SEEDPAD}.npz"

source ./env.sh

say STAGE
mkdir -p masks workspaces qu eb out
for i in 0 1 2 3 4 5 6; do
    mv "mask_${PROJ}_${i}.npz" masks/  || { echo "MISSING mask_${PROJ}_${i}.npz"; exit 1; }
    mv "w22_${PROJ}_${i}.fits" workspaces/ || { echo "MISSING w22_${PROJ}_${i}.fits"; exit 1; }
    mv "w00_${PROJ}_${i}.fits" workspaces/ || { echo "MISSING w00_${PROJ}_${i}.fits"; exit 1; }
done
mv "qu_${PROJ}_seed${SEEDPAD}.g3" qu/ || { echo "MISSING qu map"; exit 1; }
mv "eb_${PROJ}_seed${SEEDPAD}.g3" eb/ || { echo "MISSING eb map"; exit 1; }

# namaster_spectra.py discovers available mask indices from
# <--workspaces>/index.json. The real one describes the full production
# directory the worker cannot see, so synthesize a minimal one listing exactly
# the 7 entries transferred in. Only these fields are read back.
python3 - "$PROJ" <<'PYEOF'
import json, sys
proj = sys.argv[1]
json.dump({"lmax": 6000, "delta_ell": 10, "purify_b": True,
           "workspaces": [{"projection": proj, "mask_index": i,
                           "w22": f"w22_{proj}_{i}.fits",
                           "w00": f"w00_{proj}_{i}.fits"} for i in range(7)]},
          open("workspaces/index.json", "w"))
PYEOF
[ -s workspaces/index.json ] || { echo "INDEX_OK NO"; exit 1; }

say RUN
/usr/bin/time -v python3 namaster_spectra.py \
    --qu-maps qu --eb-maps eb --masks masks --workspaces workspaces \
    --projections "$PROJ" \
    --seed-start "$SEED" --seed-stop "$((SEED + 1))" \
    --purify-b purify --pixel-window healpix_and_flat \
    --lmax l6000 --delta-ell 10 --max-procs 2 --out out
rc=$?
echo "run_rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

say OUTPUT
# Hoist under the unit-qualified name so jobs don't collide on bandpowers.npz.
[ -s out/bandpowers.npz ] || { echo "OUTPUT_OK NO"; exit 1; }
mv out/bandpowers.npz "./$OUTNAME"
push_output "$OUTNAME" || exit 1
say DONE
