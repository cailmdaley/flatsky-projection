#!/bin/bash
# Step 2b: does pinning make the CVMFS toolset deterministic across sites?
#
# probe2 found that `eval $(py3-v5/setup.sh)` picks its tree from os_arch.sh,
# which falls back to the *host kernel* string when lsb_release is absent in the
# container. Five sites gave RHEL_9, RHEL_8, RHEL_7 and one silent no-op --
# including two nominally-identical Rocky 9 containers that resolved to
# different trees. This probe tests the three-part fix:
#
#   1. pin the container image in the submit file (deterministic userspace),
#   2. export OS_ARCH before the eval -- setup.sh uses `: ${OS_ARCH=...}`,
#      a default-if-unset, so an explicit value wins,
#   3. assert afterwards that python3 really resolves under the CVMFS tree,
#      because setup.sh exits 0 even when the tree does not exist.

say() { echo "### $1"; }

WANT_OS_ARCH=RHEL_9_x86_64
SROOTBASE=/cvmfs/spt.opensciencegrid.org/py3-v5

say SITE
echo "HOST=$(hostname)"
echo "SITE=${GLIDEIN_ResourceName:-unset}"
echo "OS_RELEASE=$(cat /etc/redhat-release 2>/dev/null || cat /etc/os-release 2>/dev/null | grep PRETTY | cut -d= -f2)"
echo "KERNEL=$(uname -r)"
echo "APPTAINER=${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-none}}"
echo "LSB_RELEASE=$([ -x /usr/bin/lsb_release ] && echo present || echo absent)"

say AUTODETECT
# What os_arch.sh would have chosen on its own -- recorded so we can see how
# often the pin actually changes the answer.
( . $SROOTBASE/os_arch.sh >/dev/null 2>&1; echo "auto_OS_ARCH=$OS_ARCH" )

say PINNED_ACTIVATION
unset PYTHONPATH
export OS_ARCH=$WANT_OS_ARCH
eval $($SROOTBASE/setup.sh 2>/tmp/setup.err)
echo "eval_rc=$?"
echo "OS_ARCH=$OS_ARCH"
sed 's/^/setup_stderr: /' /tmp/setup.err 2>/dev/null | head -3

say ASSERT_TOOLSET
# The check every job wrapper must make. setup.sh exits 0 on a missing tree.
PY=$(command -v python3)
echo "which_python3=$PY"
case "$PY" in
    $SROOTBASE/$WANT_OS_ARCH/*) echo "TOOLSET_OK yes" ;;
    *)                          echo "TOOLSET_OK NO -- python3 is not from the pinned tree" ;;
esac
python3 -c "import sys; print('PY', sys.version.split()[0])" 2>&1 | head -1

say LIBRARIES
python3 - <<'PY' 2>&1 | head -12
import importlib.metadata as md
for m in ["numpy", "scipy", "healpy", "astropy", "snakemake"]:
    try:
        mod = __import__(m)
        print(f"OK   {m:10s} {md.version(m)}")
    except Exception as e:
        print(f"FAIL {m:10s} {type(e).__name__}: {e}")
try:
    import pymaster
    print("OK   pymaster   dunder=", getattr(pymaster, "__version__", "ABSENT"),
          " meta=", md.version("pymaster"),
          " flat=", hasattr(pymaster, "NmtWorkspaceFlat"))
except Exception as e:
    print("FAIL pymaster  ", type(e).__name__, e)
PY

say SNAKEMAKE_CLI
# The no-shared-fs executor path needs snakemake ON THE WORKER at the AP's version.
snakemake --version 2>&1 | head -1

say DONE
