#!/bin/bash
# The one entry point for pol_project grid work.
#
#   grid/submit.sh <stage> [glob ...]        submit every missing unit
#   grid/submit.sh <stage> -n [glob ...]     census only, submit nothing
#   grid/submit.sh <stage> --local [glob ...] outputs by condor transfer instead
#
# A *unit* is one grid job's identity (e.g. qu_ProjSIN_seed0003). For each
# stage this script enumerates every unit, drops the ones that are already
# done (product present on dCache *or* on /sptlocal) or already in flight (a
# queued job carries +PolUnit = "<unit>"), keeps only those matching the
# optional shell globs, writes the survivors to grid/lists/<stage>-<stamp>.txt,
# and submits exactly that list. Then it exits.
#
# It is therefore safe -- and expected -- to re-run at any time, including
# while earlier jobs are still running. There is no chain, no lock, no
# daemon: run it when you want more work started.
#
#   grid/submit.sh project                   everything still missing
#   grid/submit.sh project 'qu_*'            just the QU leg
#   grid/submit.sh spectra 'bandpowers_ProjSIN_seed000[0-4]'
#   grid/pull.sh   project 'qu_*'           bring products down for analysis
#   grid/submit.sh spectra -n                what would go, and why not
#   grid/submit.sh probe                     validate the apparatus end to end
#
# Two stores, one direction of flow. Jobs write their products to dCache
# ($GFAL_BASE) by gfal-copy -- the right path for products that are big or
# numerous, and it keeps volume off the access point's shared transfer
# plumbing. /sptlocal stays where *analysis* happens: `grid/pull.sh <stage>`
# batch-copies anything on dCache that is missing locally. Verified in a live
# job 2026-08-19 (cluster 2679537): dCache accepts the AP-minted SciToken from
# $_CONDOR_CREDS, so nothing needs a personal token minted or shipped.
#
# --local falls back to condor transfer + a remap onto /sptlocal, for a stage
# whose products are small enough not to care.
#
# The census reads dCache with one gfal-ls, which *interactively* on amundsen
# does need a live personal token (`token-init`) -- checked below, loudly.
# Jobs need nothing: the CVMFS gfal wrapper picks up condor's own credential. A unit
# counts as done if its product is on *either* store, so products produced
# before this apparatus (or pulled down already) are never redone.
#
# Each unit row is CSV:  unit,outpath,outfile,mem,<stage-specific fields>
# consumed by the matching `queue ... from` line in grid/stages/<stage>.sub.

set -u

REPO=/home/cailmdaley/work/pol_project
RES=/sptlocal/user/cailmdaley/pol_project/results/highell
ALMS=/sptlocal/user/cailmdaley/pol_project/preprocessed_alms_l6000
GFAL_BASE=${GFAL_BASE:-davs://osg-gridftp.grid.uchicago.edu:2880/sptgrid/user/cailmdaley/pol_project/highell}
PROJECTIONS=${PROJECTIONS:-"ProjSFL ProjSIN ProjSTG ProjZEA ProjBICEP"}
SEEDS=${SEEDS:-"$(seq 0 19)"}

# --- stage unit enumerators -------------------------------------------------
# Each prints one CSV row per unit that *could* run: identity, destination,
# scratch-dir filename, request_memory, then whatever its .sub's queue line
# needs. Units whose own inputs are not on disk yet are skipped here (with a
# note on stderr) rather than submitted into a held job.
#
# Memory is per-leg / per-stage and comes from measured `/usr/bin/time -v`
# peaks, never from condor's sampled MemoryUsage, which reads about half the
# truth. QU: 38.8 GB measured (three nside-8192 float64 maps from a polarized
# alm2map) -> 64GB. EB: 24.7 GiB measured (two spin-0 alm2map calls) -> 32GB.
# The OSPool is memory-poor, so each leg asks for what it peaks at rather than
# one worst-case number.

units_project() {
    local seed pad leg proj mem alms
    for seed in $SEEDS; do
        pad=$(printf '%04d' "$seed")
        alms=$ALMS/alms_seed$pad.fits
        # A complete alms file matches seed 0's size exactly (identical shape).
        if [ ! -s "$alms" ] || [ "$(stat -c %s "$alms")" -ne "${REF_SIZE:-1080555840}" ]; then
            echo "skip seed $pad: alms missing or still being written" >&2
            continue
        fi
        for leg in qu eb; do
            [ "$leg" = qu ] && mem=${MEM_QU:-64GB} || mem=${MEM_EB:-32GB}
            for proj in $PROJECTIONS; do
                echo "${leg}_${proj}_seed${pad},$RES/projected_$leg/${leg}_${proj}_seed${pad}.g3,${leg}_${proj}_seed${pad}.g3,$mem,project_$leg.py,$leg,$proj,$seed,$pad"
            done
        done
    done
}

units_spectra() {
    # One unit = one (projection, seed), spanning all 7 mask indices and both
    # legs internally -- namaster_spectra.py offers no way to select a subset
    # of mask indices, so every job's peak is set by the same largest index
    # and there is no per-job axis left to tier memory on.
    local proj seed pad i ok
    for proj in $PROJECTIONS; do
        for seed in $SEEDS; do
            pad=$(printf '%04d' "$seed")
            ok=yes
            for i in 0 1 2 3 4 5 6; do
                [ -s "$RES/flatsky_masks/mask_${proj}_${i}.npz" ]      || ok=no
                [ -s "$RES/coupling_matrices/w22_${proj}_${i}.fits" ]  || ok=no
                [ -s "$RES/coupling_matrices/w00_${proj}_${i}.fits" ]  || ok=no
            done
            [ -s "$RES/projected_qu/qu_${proj}_seed${pad}.g3" ] || ok=no
            [ -s "$RES/projected_eb/eb_${proj}_seed${pad}.g3" ] || ok=no
            if [ "$ok" = no ]; then
                echo "skip $proj seed $pad: inputs not complete" >&2
                continue
            fi
            echo "bandpowers_${proj}_seed${pad},$RES/namaster_spectra/bandpowers_${proj}_seed${pad}.npz,bandpowers_${proj}_seed${pad}.npz,${MEM_SPECTRA:-16GB},$proj,$seed,$pad"
        done
    done
}

units_probe() {
    # Two throwaway units that exercise every moving part of the apparatus.
    local i
    for i in 0 1; do
        echo "probe_$i,$RES/grid_probe/probe_$i.txt,probe_$i.txt,1GB"
    done
}

# --- driver -----------------------------------------------------------------
stage=${1:-}
[ -n "$stage" ] || { grep '^#' "$0" | sed 1d | cut -c3-; exit 2; }
shift
type "units_$stage" >/dev/null 2>&1 || { echo "unknown stage: $stage" >&2; exit 2; }
[ -f "$REPO/grid/stages/$stage.sub" ] || { echo "no grid/stages/$stage.sub" >&2; exit 2; }

dry=no; gfal=yes
while :; do
    case "${1:-}" in
        -n) dry=yes; shift ;;
        --local) gfal=no; shift ;;
        --gfal) gfal=yes; shift ;;
        *) break ;;
    esac
done

cd "$REPO" || exit 1
mkdir -p grid/lists grid/logs

# Units already queued or running anywhere in the schedd, by their +PolUnit
# stamp. This is the whole of the double-submit guard: output-on-disk covers
# finished work, +PolUnit covers work in flight.
queued=$(condor_q -af PolUnit 2>/dev/null | grep -v '^undefined$' | sort -u)

# In gfal mode the destination -- and therefore the definition of "already
# done" -- is dCache, so list it once up front instead of stat-ing /sptlocal.
dest=$GFAL_BASE/$stage
landed=
if [ "$gfal" = yes ]; then
    eval "$(OS_ARCH=RHEL_9_x86_64 SHELL=/bin/bash /cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"
    # Jobs need no credential (the CVMFS gfal wrapper reads condor's own), but
    # this census reads dCache from the access point, which needs a live
    # personal token. token-info knows both whether one exists and whether it
    # has time left -- ask it rather than stat-ing the file.
    if [ "$(token-info -timeleft 2>/dev/null || echo 0)" -le 0 ]; then
        echo "grid token missing or expired -- mint one (token-init-poll.sh in the open-science-grid skill) or pass --local" >&2
        exit 1
    fi
    landed=$(gfal-ls "$dest" 2>/dev/null)
    echo "dCache $dest: $(echo "$landed" | grep -c .) product(s) already there" >&2
fi

# The community site blacklist, ANDed into every job's requirements. Shared
# with every other SPT user; several entries were added for gfal failures.
EXCLUDE_FILE=${EXCLUDE_FILE:-/sptlocal/analysis/condor_exclude_sites.txt}
siteexcl=True
if [ -r "$EXCLUDE_FILE" ]; then
    sites=$(sed 's/#.*//' "$EXCLUDE_FILE" | tr -d ' \t' | grep .)
    for site in $sites; do
        siteexcl="$siteexcl && (GLIDEIN_ResourceName =!= \"$site\")"
    done
    echo "excluding $(echo "$sites" | grep -c .) site(s) from $EXCLUDE_FILE" >&2
else
    echo "WARNING: $EXCLUDE_FILE unreadable -- submitting with no site exclusions" >&2
fi

list=grid/lists/$stage-$(date -u +%Y%m%dT%H%M%SZ).txt
: >"$list"
ndone=0; nflight=0; nfilter=0
while IFS=, read -r unit outpath rest; do
    # Done if the product exists on either store. Checking /sptlocal too is what
    # keeps products made before this apparatus (or already pulled down) from
    # being recomputed.
    read -r outfile _ <<<"$(echo "$rest" | tr , ' ')"
    if [ -s "$outpath" ] || { [ -n "$landed" ] && echo "$landed" | grep -qxF "$outfile"; }; then
        ndone=$((ndone + 1)); continue
    fi
    if echo "$queued" | grep -qxF "$unit"; then
        nflight=$((nflight + 1)); echo "in flight: $unit" >&2; continue
    fi
    if [ $# -gt 0 ]; then
        matched=no
        for pat in "$@"; do case "$unit" in $pat) matched=yes ;; esac; done
        [ "$matched" = yes ] || { nfilter=$((nfilter + 1)); continue; }
    fi
    echo "$unit,$outpath,$rest" >>"$list"
done < <("units_$stage")

n=$(wc -l <"$list")
echo "$stage: $n to submit, $ndone done, $nflight in flight, $nfilter filtered out" >&2
[ "$n" -gt 0 ] || { rm -f "$list"; exit 0; }
cut -d, -f1 "$list" >&2

if [ "$dry" = yes ]; then
    rm -f "$list"
    exit 0
fi
if [ "$gfal" = yes ]; then
    gfal-mkdir -p "$dest" 2>/dev/null
    condor_submit "grid/stages/$stage.sub" -a "joblist = $list" -a "siteexcl = $siteexcl" \
        -a "gfaldest = $dest" -a 'outfiles = ""' -a 'transfer_output_remaps = ""'
else
    condor_submit "grid/stages/$stage.sub" -a "joblist = $list" -a "siteexcl = $siteexcl"
fi
