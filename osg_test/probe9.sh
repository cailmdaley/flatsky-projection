#!/bin/bash
# Step 6: a production-shaped projection job on the grid.
#
# One seed, one projection: project_qu.py for seed 0, ProjSFL, on the highell
# intermediate grid (nside 8192, lmax 6000 alms), interp_rebin8, 2 arcmin, 110
# deg, dec -56. Inputs are the 57 MB spt3g build tarball (probe8), the three
# stage sources, and the 1.03 GB seed-0 alms file. Output is the .g3 map plus
# index.json, transferred back.
#
# The heavy step is hp.alm2map(alms_TEB, 8192, pol=True): three nside-8192
# float64 maps are ~6.4 GB each, hence request_memory = 64GB.

set -o pipefail
say() { echo "### $1  t=$(date -u +%FT%TZ)"; }

say SITE
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
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

export SPT3G_SOFTWARE_BUILD_PATH=$PWD/build
export PYTHONPATH=$PWD/build:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
export PATH=$PWD/build/bin:$PATH

say RUN
mkdir -p out
/usr/bin/time -v python3 project_qu.py \
    --alms . \
    --projections ProjSFL \
    --interpolation interp_rebin8 \
    --nside nside8192 \
    --res-arcmin 2.0 \
    --size-deg 110.0 \
    --ra-deg 0.0 \
    --dec-deg -56.0 \
    --seed-start 0 --seed-stop 1 \
    --max-procs 4 \
    --out out
rc=$?
echo "run_rc=$rc"

say OUTPUT
ls -la out/
[ -s out/qu_ProjSFL_seed0000.g3 ] && echo "OUTPUT_OK yes" || { echo "OUTPUT_OK NO"; exit 1; }
md5sum out/*.g3
cat out/index.json

say DONE
