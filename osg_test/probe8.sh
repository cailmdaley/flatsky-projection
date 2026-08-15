#!/bin/bash
# Step 5: ship the local spt3g_software build to an OSG worker via HTCondor
# file transfer and import it against the CVMFS toolset.
#
# The build tree at ~/code/python/spt3g_software/build symlinks every python
# package back into the source tree (build/spt3g/core -> .../core/python, 50
# links in all), so the tarball must be made with `tar -h` (dereference) or the
# worker gets 50 dangling links. Dereferenced it is 57 MB -- trivial next to the
# ~1 GB alms files the pipeline already transfers.
#
# amundsen is AlmaLinux 9 and the pinned worker image is htc/rocky:9, both
# RHEL_9_x86_64, so the compiled .so files should load against the same CVMFS
# toolset libraries they were built against. This probe tests exactly that.

say() { echo "### $1"; }

say SITE
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}  DATE=$(date -u +%FT%TZ)"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || echo unknown)"

say UNPACK
ls -la spt3g_build.tar.gz
tar -xzf spt3g_build.tar.gz || { echo "UNPACK_OK NO"; exit 1; }
echo "UNPACK_OK yes ($(find build -type f | wc -l) files, $(du -sh build | cut -f1))"
echo "dangling_symlinks=$(find build -xtype l | wc -l)"

say TOOLSET
export OS_ARCH=RHEL_9_x86_64
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac
python3 -c "import sys; print('python', sys.version.split()[0])"

say SPT3G_ENV
export SPT3G_SOFTWARE_BUILD_PATH=$PWD/build
export PYTHONPATH=$PWD/build:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
export PATH=$PWD/build/bin:$PATH
echo "SPT3G_SOFTWARE_BUILD_PATH=$SPT3G_SOFTWARE_BUILD_PATH"

say IMPORT
python3 -c "
from spt3g import core, maps
print('core', core.__file__)
print('maps', maps.__file__)
from spt3g.mapspectra.pixel_window import pixwin_flatsky
print('pixwin_flatsky([2000,3000,4000], 2.) =', pixwin_flatsky([2000,3000,4000], 2.))
"
rc=$?
echo "import_rc=$rc"
if [ $rc -ne 0 ]; then
    say LDD_DIAGNOSTIC
    for so in build/spt3g/libspt3g-core.so build/lib/libspt3g-core.so build/spt3g/core/_libcore.so; do
        [ -e "$so" ] && { echo "--- $so"; ldd "$so" 2>&1 | grep -i "not found"; }
    done
    echo "--- all build .so with missing deps"
    find build -name '*.so' | while read f; do
        miss=$(ldd "$f" 2>/dev/null | grep -i "not found")
        [ -n "$miss" ] && echo "$f: $miss"
    done
    exit 1
fi

say DONE
