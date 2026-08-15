#!/bin/bash
# Step 4: the round-trip the pipeline actually needs, with real data.
#
# dCache token auth is broken server-side (probe5/6) and the VOMS-proxy route
# needs a new certificate (human gate), but the pipeline's per-job inputs are
# well under MAX_TRANSFER_INPUT_MB=5000: a 2021 alms file is 458 MB, projected
# maps are 166 MB. So plain HTCondor file transfer from the submit node — which
# reads /sptlocal fine — carries the production workload with no credential at
# all. This probe proves it end to end: ship a real 458 MB alms FITS out,
# verify its checksum on the worker, open it with healpy under the pinned
# CVMFS toolset, and bring a derived product back.

say() { echo "### $1"; }

EXPECT_MD5=249b878848d85290e939311d16a9642a

say SITE
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}  DATE=$(date -u +%FT%TZ)"

say INPUT_INTEGRITY
ls -la almsTQU_seed0.fits
got=$(md5sum almsTQU_seed0.fits | cut -d' ' -f1)
echo "md5=$got (expect $EXPECT_MD5)"
[ "$got" = "$EXPECT_MD5" ] && echo "TRANSFER_OK yes" || echo "TRANSFER_OK NO"

say TOOLSET
export OS_ARCH=RHEL_9_x86_64
eval $(/cvmfs/spt.opensciencegrid.org/py3-v5/setup.sh)
case "$(command -v python3)" in
    /cvmfs/spt.opensciencegrid.org/py3-v5/*) echo "TOOLSET_OK yes" ;;
    *) echo "TOOLSET_OK NO ($(command -v python3))"; exit 1 ;;
esac

say COMPUTE
# Read the alms, compute per-spin power in one coarse band as a real product.
python3 <<'PY'
import healpy as hp, numpy as np
alms = hp.read_alm("almsTQU_seed0.fits", hdu=(1, 2, 3))
lmax = hp.Alm.getlmax(alms[0].size)
print("lmax =", lmax)
cls = hp.alm2cl(alms)
np.savetxt("probe7_cls_seed0.txt", np.asarray(cls[:3]).T[:, :3],
           header="TT EE BB from almsTQU_seed0.fits (2021 intermediates)")
ell = np.arange(lmax + 1)
band = (ell >= 500) & (ell < 1500)
for name, cl in zip(("TT", "EE", "BB"), cls[:3]):
    print(f"mean D_l {name} [500,1500) = {np.mean((ell*(ell+1)/(2*np.pi)*cl)[band]):.4e}")
PY
rc=$?
echo "compute_rc=$rc"
[ -s probe7_cls_seed0.txt ] && echo "OUTPUT_OK yes ($(wc -l < probe7_cls_seed0.txt) lines)" || echo "OUTPUT_OK NO"

say DONE
