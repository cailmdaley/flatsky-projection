# Worker-side environment, sourced by every stage worker after landing in the
# condor scratch dir. Pins the toolset (OS_ARCH, SHELL) per the
# open-science-grid skill, and unpacks the self-contained spt3g build if the
# stage shipped one (numpy/pymaster-only stages don't).
say() { echo "### $1  t=$(date -u +%FT%TZ)"; }

say JOB
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || echo unknown)"
echo "CPUS=$(nproc)  MEMFREE=$(awk '/MemAvailable/{print $2/1048576" GiB"}' /proc/meminfo)"
df -h . | tail -1
ls -la

# setup.sh picks its output syntax from `case ${SHELL##*/}` and emits csh
# `setenv` lines for tcsh/csh; OSPool workers do not agree on $SHELL. Pin it
# before the eval, not after.
say TOOLSET
export OS_ARCH=RHEL_9_x86_64
export SHELL=/bin/bash
eval "$(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

if [ -f spt3g_build.tar.gz ]; then
    say UNPACK
    tar -xzf spt3g_build.tar.gz || { echo "UNPACK_OK NO"; exit 1; }
    echo "UNPACK_OK yes  dangling_symlinks=$(find build -xtype l | wc -l)"
    export SPT3G_SOFTWARE_BUILD_PATH=$PWD/build
    export PYTHONPATH=$PWD/build:$PYTHONPATH
    export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
    export PATH=$PWD/build/bin:$PATH
fi

push_output() {  # push_output <file>
    # With GFAL_DEST set (the default -- see grid/submit.sh), push the product to
    # dCache and remove it, so condor transfers nothing back. With GFAL_DEST
    # empty (--local) leave it for condor to transfer and remap onto /sptlocal.
    #
    # No credential setup: gfal-copy on CVMFS is a symlink to
    # /cvmfs/spt.opensciencegrid.org/py3-v5/gfal_run.sh, which does the auth
    # itself -- inside a condor job it reads $_CONDOR_CREDS/scitokens.use,
    # extracts .access_token with jq and exports BEARER_TOKEN. Setting anything
    # ourselves is redundant (and was subtly wrong: the file is a JSON wrapper,
    # not a bare token). Verified bare in clusters 2679542 and 2679543.
    local f=$1 rc
    [ -s "$f" ] || { echo "OUTPUT_OK NO ($f)"; return 1; }
    ls -la "$f"; md5sum "$f"
    [ -n "${GFAL_DEST:-}" ] || { echo "OUTPUT_OK yes (condor transfer)"; return 0; }
    say GFAL
    # Retry with a random backoff: gfal failures against dCache are usually
    # transient (spt3g_software's grid_tools.retry_command does the same).
    for attempt in 1 2 3; do
        gfal-copy -p -f -t 1800 "$f" "$GFAL_DEST/$f" && rc=0 && break
        rc=1; echo "gfal attempt $attempt failed"; sleep $((RANDOM % 30 + 1))
    done
    [ "$rc" -eq 0 ] || { echo "GFAL_OK NO"; return 1; }
    rm -f "$f"
    echo "OUTPUT_OK yes (gfal -> $GFAL_DEST/$f)"
}
