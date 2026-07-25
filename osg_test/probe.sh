#!/bin/bash
echo "HOST=$(hostname)"
echo "OS=$(cat /etc/redhat-release 2>/dev/null || uname -a)"
echo "CVMFS_SPT=$(ls /cvmfs/spt.opensciencegrid.org 2>&1 | head -3 | tr '\n' ' ')"
echo "CPUS=$(nproc)"
python3 -c "import sys; print('PY', sys.version.split()[0])" 2>&1
echo "DONE"
