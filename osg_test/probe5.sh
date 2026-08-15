#!/bin/bash
# Step 3b: stage from the SPT dCache *inside the job*, after probe4 showed
# HTCondor's own curl_plugin cannot.
#
# probe4 put a davs:// URL in transfer_input_files and all three jobs went on
# hold: two with "Peer's Certificate issuer is not recognized", one with
# "401 login failed". The first is a TLS trust problem -- dCache presents an
# IGTF-signed certificate that the worker container's default CA bundle does not
# contain -- and file-transfer plugins run in the starter's environment, so a job
# cannot hand curl_plugin a CA path.
#
# Doing the pull inside the job fixes that (CVMFS carries the IGTF bundle at
# /cvmfs/spt.opensciencegrid.org/data/CA/certificates, 1264 certs, and setup.sh
# already exports X509_CERT_DIR to it) and has a second benefit: an in-job pull
# is not an HTCondor transfer at all, so MAX_TRANSFER_INPUT_MB (5000, and not
# overridable) does not apply to it.

say() { echo "### $1"; }

EXPECT_MD5=15609acd1c5d4003a6544a4618dcf99e
DCACHE=https://osg-gridftp.grid.uchicago.edu:2880
SRC=/sptgrid/user/cailmdaley/tarballs/spt3g.tgz
CADIR=/cvmfs/spt.opensciencegrid.org/data/CA/certificates

say SITE
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}"

say TOKEN
echo "_CONDOR_CREDS=${_CONDOR_CREDS:-unset}"
ls -la "${_CONDOR_CREDS:-/nonexistent}" 2>&1 | head -6
TOK=""
for cand in "$_CONDOR_CREDS"/scitokens.use "$_CONDOR_CREDS"/*.use; do
    [ -f "$cand" ] && { TOK="$cand"; break; }
done
if [ -n "$TOK" ]; then
    echo "token_file=$TOK ($(stat -c%s "$TOK") bytes)"
    /usr/bin/python3 - "$TOK" <<'PY' 2>&1 | head -8
import base64, json, sys
raw = open(sys.argv[1]).read().strip()
p = raw.split('.')[1]; p += '=' * (-len(p) % 4)
c = json.loads(base64.urlsafe_b64decode(p))
for k in ("iss", "aud", "scope", "exp", "sub", "ver"):
    if k in c: print(f"  claim {k}: {c[k]}")
PY
else
    echo "NO TOKEN -- AddSciToken did not deliver a credential to the job"
fi

say CA_BUNDLE
if [ -d "$CADIR" ]; then
    echo "CA dir present, $(ls "$CADIR" | wc -l) files"
else
    echo "CA dir ABSENT at $CADIR"
fi

say PULL_ANONYMOUS
# Does TLS alone work once we trust the IGTF bundle? Expect 401, not a cert error.
code=$(curl -sS --capath "$CADIR" -o /dev/null -w '%{http_code}' -I "$DCACHE$SRC" 2>/tmp/anon.err)
echo "HEAD (no token) -> HTTP ${code:-none}"
head -2 /tmp/anon.err 2>/dev/null | sed 's/^/  err: /'

say PULL_WITH_TOKEN
if [ -n "$TOK" ]; then
    code=$(curl -sS --capath "$CADIR" -H "Authorization: Bearer $(cat "$TOK")" \
             -o spt3g.tgz -w '%{http_code}' "$DCACHE$SRC" 2>/tmp/tok.err)
    echo "GET (token) -> HTTP ${code:-none}"
    head -2 /tmp/tok.err 2>/dev/null | sed 's/^/  err: /'
    if [ -f spt3g.tgz ]; then
        got=$(md5sum spt3g.tgz | cut -d' ' -f1)
        echo "md5=$got (expect $EXPECT_MD5)"
        [ "$got" = "$EXPECT_MD5" ] && echo "READ_OK yes" || echo "READ_OK NO"
    fi
else
    echo "skipped -- no token"
fi

say PUT_WITH_TOKEN
if [ -n "$TOK" ]; then
    # The amundsen NFS mount of /sptgrid is read-only, so the target dir may not
    # exist; WebDAV MKCOL creates it (207/405 if it already exists is fine).
    mkcol=$(curl -sS --capath "$CADIR" -X MKCOL \
             -H "Authorization: Bearer $(cat "$TOK")" \
             -o /dev/null -w '%{http_code}' "$DCACHE/sptgrid/user/cailmdaley/osg_test" 2>/dev/null)
    echo "MKCOL osg_test -> HTTP ${mkcol:-none}"
    echo "probe5 from $(hostname) at $(date -u +%FT%TZ)" > payload.txt
    dst="/sptgrid/user/cailmdaley/osg_test/probe5_$(hostname)_$$.txt"
    code=$(curl -sS --capath "$CADIR" -T payload.txt \
             -H "Authorization: Bearer $(cat "$TOK")" \
             -o /dev/null -w '%{http_code}' "$DCACHE$dst" 2>/tmp/put.err)
    echo "PUT $dst -> HTTP ${code:-none}"
    head -2 /tmp/put.err 2>/dev/null | sed 's/^/  err: /'
else
    echo "skipped -- no token"
fi

say DONE
