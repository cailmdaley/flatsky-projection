#!/bin/bash
# Production worker for the highell namaster_spectra stage on OSG.
#
# One job = one (projection, seed) pair, spanning ALL 7 mask indices and
# both legs (qu/eb) internally -- namaster_spectra.py's --projections/
# --seed-start/--seed-stop flags don't let a caller select a mask-index
# subset; it always processes every mask index it finds for the requested
# projection(s) in the --workspaces manifest. Splitting by mask index (the
# way grid/coupling does) would mean re-transferring the same seed's
# ~350MB of qu/eb .g3 maps once per index, 7x the traffic for no shrink in
# per-job memory (every job still has to walk up to the largest index
# eventually). Splitting by (projection, seed) instead transfers each g3
# pair exactly once and keeps jobs small (single seed, ~seconds of FFT
# decoupling per mask x leg): worker.sh <projection> <seed>
# e.g. worker.sh ProjSIN 0
#
# Unlike grid/coupling/worker.sh (numpy + pymaster only), this stage needs
# spt3g itself: `from spt3g import core, maps` registers FlatSkyMap so
# G3File can deserialize the projected Q/U/E/B maps. So, like
# grid/worker.sh (the projection stage), this unpacks the self-contained
# spt3g build tarball transferred in alongside the stage sources.
#
# namaster_spectra.py discovers which mask indices are available for a
# projection via <--workspaces>/index.json (written by coupling_matrices.py
# -- see load_workspace_manifest()). Only one real index.json exists, for
# the FULL production coupling_matrices output directory, which this job
# cannot see (workers only get the files condor transfers). So this script
# synthesizes a minimal index.json in a local workspaces/ dir listing
# exactly the 7 (projection, mask_index) entries transferred in -- the only
# fields namaster_spectra.py's load_workspace_manifest()/main() actually
# read: lmax, delta_ell, purify_b, and each entry's projection/mask_index/
# w22/w00. nx/ny/lx_rad/ly_rad are written by coupling_matrices.py but never
# read back by namaster_spectra.py (it loads geometry straight from the
# mask npz), so they're omitted here.

set -o pipefail

PROJ=$1
SEED=$2
SEEDPAD=$(printf '%04d' "$SEED")
OUTNAME="bandpowers_${PROJ}_seed${SEEDPAD}.npz"
INDICES="0 1 2 3 4 5 6"

say() { echo "### $1  t=$(date -u +%FT%TZ)"; }

say JOB
echo "PROJ=$PROJ  SEED=$SEED  OUTNAME=$OUTNAME"
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || echo unknown)"
echo "CPUS=$(nproc)  MEMFREE=$(awk '/MemAvailable/{print $2/1048576" GiB"}' /proc/meminfo)"
df -h . | tail -1

say INPUTS
ls -la

say UNPACK
tar -xzf spt3g_build.tar.gz || { echo "UNPACK_OK NO"; exit 1; }
echo "UNPACK_OK yes  dangling_symlinks=$(find build -xtype l | wc -l)"

say TOOLSET
export OS_ARCH=RHEL_9_x86_64
# setup.sh picks its output syntax from `case ${SHELL##*/}` and emits csh
# `setenv` lines for tcsh/csh. OSPool workers do not agree on $SHELL -- some
# hand out csh -- and the failure is `setenv: command not found` from a shell
# that was always bash. Pin it before the eval, not after.
export SHELL=/bin/bash
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

export SPT3G_SOFTWARE_BUILD_PATH=$PWD/build
export PYTHONPATH=$PWD/build:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
export PATH=$PWD/build/bin:$PATH

say STAGE
mkdir -p masks workspaces qu eb out
for i in $INDICES; do
    mv "mask_${PROJ}_${i}.npz" "masks/" || { echo "MISSING mask_${PROJ}_${i}.npz"; exit 1; }
    mv "w22_${PROJ}_${i}.fits" "workspaces/" || { echo "MISSING w22_${PROJ}_${i}.fits"; exit 1; }
    mv "w00_${PROJ}_${i}.fits" "workspaces/" || { echo "MISSING w00_${PROJ}_${i}.fits"; exit 1; }
done
mv "qu_${PROJ}_seed${SEEDPAD}.g3" "qu/" || { echo "MISSING qu_${PROJ}_seed${SEEDPAD}.g3"; exit 1; }
mv "eb_${PROJ}_seed${SEEDPAD}.g3" "eb/" || { echo "MISSING eb_${PROJ}_seed${SEEDPAD}.g3"; exit 1; }

python3 - "$PROJ" <<'PYEOF'
import json, sys
proj = sys.argv[1]
manifest = {
    "lmax": 6000,
    "delta_ell": 10,
    "purify_b": True,
    "workspaces": [
        {
            "projection": proj,
            "mask_index": i,
            "w22": f"w22_{proj}_{i}.fits",
            "w00": f"w00_{proj}_{i}.fits",
        }
        for i in range(7)
    ],
}
with open("workspaces/index.json", "w") as f:
    json.dump(manifest, f)
PYEOF
[ -s workspaces/index.json ] || { echo "INDEX_OK NO"; exit 1; }
echo "INDEX_OK yes"

say RUN
/usr/bin/time -v python3 namaster_spectra.py \
    --qu-maps qu \
    --eb-maps eb \
    --masks masks \
    --workspaces workspaces \
    --projections "$PROJ" \
    --seed-start "$SEED" --seed-stop "$((SEED + 1))" \
    --purify-b purify \
    --pixel-window healpix_and_flat \
    --lmax l6000 --delta-ell 10 \
    --max-procs 2 \
    --out out
rc=$?
echo "run_rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

say OUTPUT
ls -la out/
[ -s "out/bandpowers.npz" ] || { echo "OUTPUT_OK NO (bandpowers.npz missing)"; exit 1; }
# Hoist under a (projection, seed)-qualified name so the submit file
# transfers exactly this one file and can remap it into
# results/highell/namaster_spectra/ without every job colliding on the same
# "bandpowers.npz" name. out/index.json and out/partial/ are deliberately
# left behind: they describe only this single-seed run, not the assembled
# 5x20 production manifest.
mv "out/bandpowers.npz" "./$OUTNAME"
echo "OUTPUT_OK yes"
ls -la "./$OUTNAME"
md5sum "./$OUTNAME"

say DONE
