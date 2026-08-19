#!/bin/bash
# Worker for the probe stage: assert the pinned toolset resolved, then write a
# small file and hand it to push_output. Whatever this job does with its output
# is exactly what a real stage does with a 350 MB map.
set -o pipefail
OUTNAME=$1
source ./env.sh

say RUN
python3 -c "import numpy, pymaster; print('numpy', numpy.__version__, 'pymaster', pymaster.__version__)" || exit 1
{
    echo "unit=$OUTNAME host=$(hostname) site=${GLIDEIN_ResourceName:-unset} t=$(date -u +%FT%TZ)"
    echo "gfal_dest=${GFAL_DEST:-<condor remap>}"
} > "$OUTNAME"

say OUTPUT
push_output "$OUTNAME" || exit 1
say DONE
