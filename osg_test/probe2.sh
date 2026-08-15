#!/bin/bash
# Step 2 of the OSG ladder: does a CVMFS toolset actually activate on a worker,
# and what else is true there that shapes how we write rules?
#
# Deliberately does NOT exit on error -- every section reports independently,
# so one failure does not hide the answers after it.

say() { echo "### $1"; }

say SITE
echo "HOST=$(hostname)"
echo "SITE=${GLIDEIN_ResourceName:-unset}"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || uname -a)"
echo "CPUS=$(nproc)"
echo "MEM_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)"
echo "SCRATCH=$(pwd)"
echo "SCRATCH_FREE_GB=$(df -BG --output=avail . | tail -1 | tr -d ' G')"
echo "APPTAINER=${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-none}}"

say CVMFS_REPOS
for r in spt.opensciencegrid.org oasis.opensciencegrid.org singularity.opensciencegrid.org; do
    if [ -d "/cvmfs/$r" ]; then
        echo "OK   /cvmfs/$r"
    else
        echo "MISS /cvmfs/$r"
    fi
done

say TOOLSET_ACTIVATION
# The whole point of step 2. py3-v5 is the recommended base.
if [ -f /cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh ]; then
    eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh) 2>/tmp/setup.err
    rc=$?
    echo "eval_rc=$rc"
    head -5 /tmp/setup.err 2>/dev/null | sed 's/^/setup_stderr: /'
    echo "which_python3=$(command -v python3)"
    python3 -c "import sys; print('PY', sys.version.split()[0])" 2>&1 | head -2
else
    echo "MISSING py3-v5/setup.sh"
fi

say IMPORTS
python3 - <<'PY' 2>&1 | head -20
mods = ["numpy", "scipy", "healpy", "pymaster", "astropy", "matplotlib", "camb", "h5py"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"OK   {m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
PY

say NUMPY_THREADS
python3 -c "import numpy as np, os; print('OMP_NUM_THREADS=', os.environ.get('OMP_NUM_THREADS','unset')); a=np.random.rand(600,600); print('dgemm ok', float((a@a).sum())>0)" 2>&1 | head -3

say TRANSFER_PLUGINS
# Which schemes can this worker's condor pull? Decides the data-staging design.
if command -v condor_config_val >/dev/null 2>&1; then
    condor_config_val FILETRANSFER_PLUGINS 2>&1 | tr ',' '\n' | sed 's/^ *//' | head -30
else
    echo "no condor_config_val on worker"
fi
echo "--- plugin binaries in PATH ---"
for p in pelican_plugin osdf_plugin stash_plugin curl_plugin; do
    command -v $p >/dev/null 2>&1 && echo "OK   $p" || echo "MISS $p"
done

say OSDF_CLIENT
for c in pelican stashcp osdf; do
    command -v $c >/dev/null 2>&1 && echo "OK   $c -> $(command -v $c)" || echo "MISS $c"
done

say NETWORK
# Can the worker reach the outside world directly? Matters for any pull-based staging.
timeout 20 curl -sS -o /dev/null -w "pypi_http=%{http_code}\n" https://pypi.org/simple/ 2>&1 | head -2
timeout 20 getent hosts osg-gridftp.grid.uchicago.edu 2>&1 | head -2 || echo "gridftp_dns=FAIL"

say DONE
