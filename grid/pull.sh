#!/bin/bash
# Bring a stage's products down from dCache to /sptlocal, where analysis lives.
#
#   grid/pull.sh <stage> [glob ...]      copy everything not already local
#   grid/pull.sh <stage> -n [glob ...]   list what would come down
#
# Jobs push to dCache (grid/submit.sh); this is the other half of that flow.
# One batched gfal-copy --from-file per stage rather than a copy per file.
# Needs a live personal token -- unlike the jobs, which carry condor's own.
set -u

REPO=/home/cailmdaley/work/pol_project
RES=/sptlocal/user/cailmdaley/pol_project/results/highell
GFAL_BASE=${GFAL_BASE:-davs://osg-gridftp.grid.uchicago.edu:2880/sptgrid/user/cailmdaley/pol_project/highell}

# Where each stage's products belong once local. A stage whose products fan out
# into per-leg directories needs the leg back off the filename.
localdir() {  # localdir <stage> <filename>
    case "$1" in
        project) echo "$RES/projected_${2%%_*}" ;;
        spectra) echo "$RES/namaster_spectra" ;;
        *)       echo "$RES/grid_$1" ;;
    esac
}

stage=${1:-}
[ -n "$stage" ] || { grep '^#' "$0" | sed 1d | cut -c3-; exit 2; }
shift
dry=no; [ "${1:-}" = "-n" ] && { dry=yes; shift; }

eval "$(OS_ARCH=RHEL_9_x86_64 SHELL=/bin/bash /cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"
if [ "$(token-info -timeleft 2>/dev/null || echo 0)" -le 0 ]; then
    echo "grid token missing or expired -- mint one (token-init-poll.sh in the open-science-grid skill)" >&2
    exit 1
fi

src=$GFAL_BASE/$stage
# gfal-copy --from-file takes source URLs only plus ONE destination directory,
# so group the sources by destination -- in practice a stage has one or two
# (project fans out into projected_qu / projected_eb).
work=$(mktemp -d) ; trap 'rm -rf "$work"' EXIT
n=0; have=0
for f in $(gfal-ls "$src" 2>/dev/null); do
    if [ $# -gt 0 ]; then
        matched=no
        for pat in "$@"; do case "$f" in $pat) matched=yes ;; esac; done
        [ "$matched" = yes ] || continue
    fi
    dst=$(localdir "$stage" "$f")
    [ -s "$dst/$f" ] && { have=$((have + 1)); continue; }
    # Index batches by a counter, not by a mangled form of the path: the
    # destinations contain underscores and slashes and round-tripping them
    # through tr corrupts the path.
    key=$(grep -Fxn "$dst" "$work/dsts" 2>/dev/null | cut -d: -f1)
    [ -n "$key" ] || { echo "$dst" >>"$work/dsts"; key=$(grep -c . "$work/dsts"); }
    echo "$src/$f" >>"$work/batch.$key"
    echo "$f -> $dst" >&2
    n=$((n + 1))
done

echo "$stage: $n to pull, $have already local" >&2
[ "$n" -gt 0 ] || exit 0
[ "$dry" = no ] || exit 0

rc=0
for batch in "$work"/batch.*; do
    dst=$(sed -n "${batch##*.}p" "$work/dsts")
    mkdir -p "$dst"
    echo "== $(wc -l <"$batch") file(s) -> $dst" >&2
    gfal-copy -p -f -t 1800 --from-file "$batch" "$dst" || rc=1
done
exit $rc
