#!/bin/bash
# env.sh -- run a project command under the spt3g stack, locally or on an
# OSG worker. One command works in both places:
#
#   bash env.sh python src/stages/<stage>.py ...
#
# Worker mode: an HTCondor-transferred spt3g_build.tar.gz sits in the job
# scratch (cwd). Untar it once, activate the CVMFS py3-v5 toolset (which
# carries python + healpy + pymaster), and point the spt3g env at the
# unpacked build.
#
# Local mode (amundsen): no tarball in cwd; defer to the real build's
# env-shell.sh.
set -euo pipefail

if [ -f spt3g_build.tar.gz ] && [ ! -d build ]; then
    tar -xzf spt3g_build.tar.gz
fi

if [ -d build ] && [ ! -e /home/cailmdaley/code/python/spt3g_software/build/env-shell.sh ]; then
    export OS_ARCH=RHEL_9_x86_64
    # setup.sh emits csh `setenv` lines when $SHELL looks like tcsh/csh, and
    # OSPool workers do not agree on $SHELL. Pin it before the eval.
    export SHELL=/bin/bash
    eval "$(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)"
    export SPT3G_SOFTWARE_PATH=$PWD
    export SPT3G_SOFTWARE_BUILD_PATH=$PWD/build
    export PYTHONPATH=$PWD/build:${PYTHONPATH:-}
    export LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-}
    export PATH=$PWD/build/bin:$PATH
    exec "$@"
fi

exec /home/cailmdaley/code/python/spt3g_software/build/env-shell.sh "$@"
