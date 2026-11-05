#!/bin/bash
# Restore dCache-staged upstream outputs in a Condor job sandbox.
set -euo pipefail
shopt -s nullglob

for marker in .lightcone/stage/*__*.stage; do
    mapfile -t receipt < "$marker"
    if [ "${#receipt[@]}" -ne 2 ]; then
        echo "stage-inputs.sh: malformed staging receipt $marker" >&2
        exit 1
    fi
    archive="$(basename "${marker%.stage}").tar.gz"
    copied=0
    for attempt in 1 2 3; do
        if gfal-copy -f -p "${receipt[0]}" "$PWD/$archive"; then
            copied=1
            break
        fi
        sleep $((attempt * 3))
    done
    if [ "$copied" -ne 1 ]; then
        echo "stage-inputs.sh: gfal download failed for ${receipt[0]}" >&2
        exit 1
    fi
    actual=$(sha256sum "$archive" | cut -d' ' -f1)
    if [ "$actual" != "${receipt[1]}" ]; then
        echo "stage-inputs.sh: checksum mismatch for $marker" >&2
        exit 1
    fi
    tar -xzf "$archive"
    rm "$archive"
done
